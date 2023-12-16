# coding: UTF-8
# Copyright (c) Facebook, Inc. and its affiliates.
import itertools
import logging
import math
import os
import random
from typing import Dict, List, Tuple, Union
from math import e
import numpy as np
import torch
import torch.distributions as dists
from detectron2.config import configurable
from detectron2.layers import (ShapeSpec, batched_nms, cat, cross_entropy,
                               nonzero_tuple)
#from detectron2.modeling.box_regression import Box2BoxTransform
from detectron2.modeling.box_regression import Box2BoxTransform, _dense_box_regression_loss
from detectron2.modeling.roi_heads.fast_rcnn import (FastRCNNOutputLayers,
                                                     _log_classification_stats)
from detectron2.structures import Boxes, Instances, pairwise_iou
from detectron2.structures.boxes import matched_boxlist_iou

#  fast_rcnn_inference)
from detectron2.utils import comm
from detectron2.utils.events import get_event_storage
from detectron2.utils.registry import Registry
from fvcore.nn import giou_loss, smooth_l1_loss
from torch import nn
from torch.nn import functional as F

from ..layers import MLP
from ..losses import ICLoss, UPLoss
from typing import Optional
ROI_BOX_OUTPUT_LAYERS_REGISTRY = Registry("ROI_BOX_OUTPUT_LAYERS")
ROI_BOX_OUTPUT_LAYERS_REGISTRY.__doc__ = """
ROI_BOX_OUTPUT_LAYERS
"""


def fast_rcnn_inference(
        boxes: List[torch.Tensor],
        scores: List[torch.Tensor],
        uncertainty:List[torch.Tensor],
        image_shapes: List[Tuple[int, int]],
        score_thresh: float,
        nms_thresh: float,
        topk_per_image: int,
        vis_iou_thr: float = 1.0,
        uncertainty_thr: float=0.5,
):
    result_per_image = [
        fast_rcnn_inference_single_image(
            boxes_per_image, scores_per_image, uncertainty_per_image, image_shape, score_thresh, nms_thresh, topk_per_image, vis_iou_thr,uncertainty_thr
        )
        for scores_per_image, boxes_per_image, uncertainty_per_image, image_shape in zip(scores, boxes,uncertainty, image_shapes)
    ]

    return [x[0] for x in result_per_image], [x[1] for x in result_per_image]


def fast_rcnn_inference_single_image(
        boxes,
        scores,
        uncertainty,
        image_shape: Tuple[int, int],
        score_thresh: float=0.03,
        nms_thresh: float,
        topk_per_image: int,
        vis_iou_thr: float,
        uncertainty_thr: float,
):
    valid_mask = torch.isfinite(boxes).all(
        dim=1) & torch.isfinite(scores).all(dim=1)
        boxes = boxes[valid_mask]
        scores = scores[valid_mask]
    scores = scores[:, :-1]
    num_bbox_reg_classes = boxes.shape[1] // 4
    # Convert to Boxes to use the `clip` function ...
    boxes = Boxes(boxes.reshape(-1, 4))
    boxes.clip(image_shape)
    boxes = boxes.tensor.view(-1, num_bbox_reg_classes, 4)  # R x C x 4

    # 1. Filter results based on detection scores. It can make NMS more efficient
    #    by filtering out low-confidence detections.
    filter_mask = scores > score_thresh

    # R' x 2. First column contains indices of the R predictions;
    # Second column contains indices of classes.
    filter_inds = filter_mask.nonzero()
    if num_bbox_reg_classes == 1:
        boxes = boxes[filter_inds[:, 0], 0]
    else:
        boxes = boxes[filter_mask]
    scores = scores[filter_mask]
    uncertainty = uncertainty[filter_inds[:, 0]]

    # 2. Apply NMS for each class independently.
    keep = batched_nms(boxes, scores, filter_inds[:, 1], nms_thresh)

    if topk_per_image >= 0:

        keep = keep[:topk_per_image]

    boxes, scores, filter_inds = boxes[keep], scores[keep], filter_inds[
        keep]
    uncertainty = uncertainty[keep]


    unknown = uncertainty > uncertainty_thr
    unknown_nonzero = unknown.nonzero()
    filter_inds[unknown_nonzero, 1] = 80
    scores[unknown_nonzero]=(15*uncertainty[unknown_nonzero]+scores[unknown_nonzero])/2+score_thresh
    

    
    unknown = (uncertainty > (uncertainty_thr/2)) * (uncertainty <= uncertainty_thr)*(scores <= score_thresh)
    unknown_nonzero = unknown.nonzero()
    filter_inds[unknown_nonzero, 1] = 80
    scores[unknown_nonzero]=(15*uncertainty[unknown_nonzero]+scores[unknown_nonzero])/2+score_thresh
    
    # apply nms between known classes and unknown class for visualization.
    '''
    if vis_iou_thr < 1.0:
        boxes, scores, filter_inds = unknown_aware_nms(
            boxes, scores, filter_inds, iou_thr=vis_iou_thr)
    '''
    # 2. Apply NMS for each class independently.
    keep = batched_nms(boxes, scores, filter_inds[:, 1], nms_thresh)

    if topk_per_image >= 0:

        keep = keep[:topk_per_image]

    boxes, scores, filter_inds = boxes[keep], scores[keep], filter_inds[
        keep]
    filter_mask = scores > score_thresh
    boxes, scores, filter_inds = boxes[filter_mask], scores[filter_mask], filter_inds[filter_mask]
    #
    result = Instances(image_shape)
    result.pred_boxes = Boxes(boxes)
    result.scores = scores
    result.pred_classes = filter_inds[:, 1]
    return result, filter_inds[:, 0]

def unknown_aware_nms(boxes, scores, labels, ukn_class_id=80, iou_thr=0.9):
    u_inds = labels[:, 1] == ukn_class_id
    k_inds = ~u_inds
    if k_inds.sum() == 0 or u_inds.sum() == 0:
        return boxes, scores, labels

    k_boxes, k_scores, k_labels = boxes[k_inds], scores[k_inds], labels[k_inds]
    u_boxes, u_scores, u_labels = boxes[u_inds], scores[u_inds], labels[u_inds]

    ious = pairwise_iou(Boxes(k_boxes), Boxes(u_boxes))
    mask = torch.ones((ious.size(0), ious.size(1), 2), device=ious.device)
    inds = (ious > iou_thr).nonzero()
    if not inds.numel():
        return boxes, scores, labels

    for [ind_x, ind_y] in inds:
        if k_scores[ind_x] >= u_scores[ind_y]:
            mask[ind_x, ind_y, 1] = 0
        else:
            mask[ind_x, ind_y, 0] = 0

    k_inds = mask[..., 0].mean(dim=1) == 1
    u_inds = mask[..., 1].mean(dim=0) == 1

    k_boxes, k_scores, k_labels = k_boxes[k_inds], k_scores[k_inds], k_labels[k_inds]
    u_boxes, u_scores, u_labels = u_boxes[u_inds], u_scores[u_inds], u_labels[u_inds]

    boxes = torch.cat([k_boxes, u_boxes])
    scores = torch.cat([k_scores, u_scores])
    labels = torch.cat([k_labels, u_labels])

    return boxes, scores, labels


logger = logging.getLogger(__name__)


def build_roi_box_output_layers(cfg, input_shape):
    """
    Build ROIHeads defined by `cfg.MODEL.ROI_HEADS.NAME`.
    """
    name = cfg.MODEL.ROI_BOX_HEAD.OUTPUT_LAYERS
    return ROI_BOX_OUTPUT_LAYERS_REGISTRY.get(name)(cfg, input_shape)


@ROI_BOX_OUTPUT_LAYERS_REGISTRY.register()
class EvidentialOutputLayers(nn.Module):
    """
    Two linear layers for predicting Fast R-CNN outputs:

    1. proposal-to-detection box regression deltas
    2. classification scores
    """

    @configurable
    def __init__(
            self,
            input_shape: ShapeSpec,
            *,
            box2box_transform,

            num_known_classes: int,
            test_score_thresh: float = 0.0,
            test_nms_thresh: float = 0.5,
            test_topk_per_image: int = 100,
            cls_agnostic_bbox_reg: bool = False,
            smooth_l1_beta: float = 0.0,
            box_reg_loss_type: str = "smooth_l1",
            loss_weight: Union[float, Dict[str, float]] = 1.0,
            vis_iou_thr: float = 1.0,
            max_iters,
            uncertainty_thr: float,
            evidence_loss_type:str,
            ic_loss_out_dim,
            ic_loss_queue_size,
            ic_loss_in_queue_size,
            ic_loss_batch_iou_thr,
            ic_loss_queue_iou_thr,
            ic_loss_queue_tau,
            ic_loss_weight,
    ):
        """
        NOTE: this interface is experimental.

        Args:
            input_shape (ShapeSpec): shape of the input feature to this module
            box2box_transform (Box2BoxTransform or Box2BoxTransformRotated):
            num_classes (int): number of foreground classes
            test_score_thresh (float): threshold to filter predictions results.
            test_nms_thresh (float): NMS threshold for prediction results.
            test_topk_per_image (int): number of top predictions to produce per image.
            cls_agnostic_bbox_reg (bool): whether to use class agnostic for bbox regression
            smooth_l1_beta (float): transition point from L1 to L2 loss. Only used if
                `box_reg_loss_type` is "smooth_l1"
            box_reg_loss_type (str): Box regression loss type. One of: "smooth_l1", "giou",
                "diou", "ciou"
            loss_weight (float|dict): weights to use for losses. Can be single float for weighting
                all losses, or a dict of individual weightings. Valid dict keys are:
                    * "loss_cls": applied to classification loss
                    * "loss_box_reg": applied to box regression loss
            use_fed_loss (bool): whether to use federated loss which samples additional negative
                classes to calculate the loss
            use_sigmoid_ce (bool): whether to calculate the loss using weighted average of binary
                cross entropy with logits. This could be used together with federated loss
            get_fed_loss_cls_weights (Callable): a callable which takes dataset name and frequency
                weight power, and returns the probabilities to sample negative classes for
                federated loss. The implementation can be found in
                detectron2/data/detection_utils.py
            fed_loss_num_classes (int): number of federated classes to keep in total
        """
        super().__init__()
        if isinstance(input_shape, int):  # some backward compatibility
            input_shape = ShapeSpec(channels=input_shape)

        self.num_known_classes = num_known_classes
        input_size = input_shape.channels * (input_shape.width or 1) * (input_shape.height or 1)
        # prediction layer for num_classes foreground classes and one background class (hence + 1)
        # self.cls_score = nn.Linear(input_size, num_classes + 1)
        self.cls_score = nn.Linear(input_size, num_known_classes + 1)
        #self.relu = nn.ReLU(inplace=True)
        # num_bbox_reg_classes = 1 if cls_agnostic_bbox_reg else num_classes
        num_bbox_reg_classes = 1 if cls_agnostic_bbox_reg else num_known_classes

        box_dim = len(box2box_transform.weights)
        self.bbox_pred = nn.Linear(input_size, num_bbox_reg_classes * box_dim)

        nn.init.normal_(self.cls_score.weight, std=0.01)
        nn.init.normal_(self.bbox_pred.weight, std=0.001)
        for l in [self.cls_score, self.bbox_pred]:
            nn.init.constant_(l.bias, 0)

        self.box2box_transform = box2box_transform
        self.smooth_l1_beta = smooth_l1_beta
        self.test_score_thresh = test_score_thresh
        self.test_nms_thresh = test_nms_thresh
        self.test_topk_per_image = test_topk_per_image
        self.box_reg_loss_type = box_reg_loss_type
        if isinstance(loss_weight, float):
            loss_weight = {"loss_cls": loss_weight, "loss_box_reg": loss_weight}
        self.loss_weight = loss_weight

        self.max_iters = max_iters
        self.vis_iou_thr = vis_iou_thr
        self.uncertainty_thr=uncertainty_thr
        self.evidence_loss_type=evidence_loss_type
        

        self.encoder = MLP(input_size, ic_loss_out_dim)
        self.ic_loss_loss = ICLoss(tau=ic_loss_queue_tau)
        self.ic_loss_out_dim = ic_loss_out_dim
        self.ic_loss_queue_size = ic_loss_queue_size
        self.ic_loss_in_queue_size = ic_loss_in_queue_size
        self.ic_loss_batch_iou_thr = ic_loss_batch_iou_thr
        self.ic_loss_queue_iou_thr = ic_loss_queue_iou_thr
        self.ic_loss_weight = ic_loss_weight

        self.register_buffer('queue', torch.zeros(
            self.num_known_classes, ic_loss_queue_size, ic_loss_out_dim))
        self.register_buffer('queue_label', torch.empty(
            self.num_known_classes, ic_loss_queue_size).fill_(-1).long())
        self.register_buffer('queue_ptr', torch.zeros(
            self.num_known_classes, dtype=torch.long))
               

        self.para = torch.nn.Parameter(torch.tensor([1.0]), requires_grad=True)
        

        self.clip_encoder = MLP(512, ic_loss_out_dim)
    @classmethod
    def from_config(cls, cfg, input_shape):
        return {
            "input_shape": input_shape,
            "box2box_transform": Box2BoxTransform(weights=cfg.MODEL.ROI_BOX_HEAD.BBOX_REG_WEIGHTS),

            'num_known_classes': cfg.MODEL.ROI_HEADS.NUM_KNOWN_CLASSES,
            "max_iters": cfg.SOLVER.MAX_ITER,
            'vis_iou_thr': cfg.MODEL.ROI_HEADS.VIS_IOU_THRESH,
            "cls_agnostic_bbox_reg": cfg.MODEL.ROI_BOX_HEAD.CLS_AGNOSTIC_BBOX_REG,
            "smooth_l1_beta": cfg.MODEL.ROI_BOX_HEAD.SMOOTH_L1_BETA,
            "test_score_thresh": cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST,
            "test_nms_thresh": cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST,
            "test_topk_per_image": cfg.TEST.DETECTIONS_PER_IMAGE,
            "box_reg_loss_type": cfg.MODEL.ROI_BOX_HEAD.BBOX_REG_LOSS_TYPE,
            "loss_weight": {"loss_box_reg": cfg.MODEL.ROI_BOX_HEAD.BBOX_REG_LOSS_WEIGHT},
            'uncertainty_thr': cfg.MODEL.ROI_HEADS.UNCERTAINTY_THR,
            "evidence_loss_type":cfg.MODEL.ROI_HEADS.EVIDENCE_LOSS_TYPE,

            

            "ic_loss_out_dim": cfg.ICLOSS.OUT_DIM,
            "ic_loss_queue_size": cfg.ICLOSS.QUEUE_SIZE,
            "ic_loss_in_queue_size": cfg.ICLOSS.IN_QUEUE_SIZE,
            "ic_loss_batch_iou_thr": cfg.ICLOSS.BATCH_IOU_THRESH,
            "ic_loss_queue_iou_thr": cfg.ICLOSS.QUEUE_IOU_THRESH,
            "ic_loss_queue_tau": cfg.ICLOSS.TEMPERATURE,
            "ic_loss_weight": cfg.ICLOSS.WEIGHT,

            
        }

    def forward(self, x,clip_gt_classes:Optional=None,proposals_clip_feature:Optional=None):
        """
        Args:
            x: per-region features of shape (N, ...) for N bounding boxes to predict.

        Returns:
            (Tensor, Tensor):
            First tensor: shape (N,K+1), scores for each of the N box. Each row contains the
            scores for K object categories and 1 background class.

            Second tensor: bounding box regression deltas for each box. Shape is shape (N,Kx4),
            or (N,4) for class-agnostic regression.
        """
        if isinstance(x, tuple):
            reg_x, cls_x = x
        else:
            reg_x = cls_x = x
        if reg_x.dim() > 2:
            reg_x = torch.flatten(reg_x, start_dim=1)
            cls_x = torch.flatten(cls_x, start_dim=1)
            
        pre_scores = self.cls_score(cls_x)

        scores=torch.exp(torch.clamp(pre_scores, -10, 10))
        proposal_deltas = self.bbox_pred(reg_x)
        

        # encode feature with MLP
        mlp_feat = self.encoder(cls_x)

        if proposals_clip_feature!=None:
            clp_feat=self.clip_encoder(proposals_clip_feature)
            return scores, proposal_deltas,mlp_feat,clip_gt_classes,clp_feat
        return scores, proposal_deltas,mlp_feat
        
    def get_ic_loss(self, feat, gt_classes, ious):
        # select foreground and iou > thr instance in a mini-batch
        pos_inds = (ious > self.ic_loss_batch_iou_thr) & (
            gt_classes != self.num_known_classes)
        feat, gt_classes = feat[pos_inds], gt_classes[pos_inds]

        queue = self.queue.reshape(-1, self.ic_loss_out_dim)
        queue_label = self.queue_label.reshape(-1)
        queue_inds = queue_label != -1  # filter empty queue
        queue, queue_label = queue[queue_inds], queue_label[queue_inds]

        loss_ic_loss = self.ic_loss_loss(feat, gt_classes, queue, queue_label)#return loss
        # annealing coefficient
        storage = get_event_storage()
        decay_weight = 1.0 - storage.iter / self.max_iters
        return {"loss_cls_ic": self.ic_loss_weight * decay_weight * loss_ic_loss}

    def get_clip_loss(self, feat, gt_classes):


        queue = self.queue.reshape(-1, self.ic_loss_out_dim)
        queue_label = self.queue_label.reshape(-1)
        queue_inds = queue_label != -1  # filter empty queue
        queue, queue_label = queue[queue_inds], queue_label[queue_inds]

        loss_clip_loss = self.ic_loss_loss(feat, gt_classes, queue, queue_label)#return loss
        # annealing coefficient
        storage = get_event_storage()
        decay_weight = 1.0 - storage.iter / self.max_iters
        return {"loss_clip": self.ic_loss_weight * decay_weight * loss_clip_loss}

    @torch.no_grad()
    def _dequeue_and_enqueue(self, feat, gt_classes, ious, iou_thr=0.7):
        # 1. gather variable
        feat = self.concat_all_gather(feat)
        gt_classes = self.concat_all_gather(gt_classes)
        ious = self.concat_all_gather(ious)
        # 2. filter by iou and obj, remove bg
        keep = (ious > iou_thr) & (gt_classes != self.num_known_classes)
        feat, gt_classes = feat[keep], gt_classes[keep]

        for i in range(self.num_known_classes):
            ptr = int(self.queue_ptr[i])
            cls_ind = gt_classes == i
            cls_feat, cls_gt_classes = feat[cls_ind], gt_classes[cls_ind]
            # 3. sort by similarity, low sim ranks first
            cls_queue = self.queue[i, self.queue_label[i] != -1]

            _, sim_inds = F.cosine_similarity(
                cls_feat[:, None], cls_queue[None, :], dim=-1).mean(dim=1).sort()
            top_sim_inds = sim_inds[:self.ic_loss_in_queue_size]
            cls_feat, cls_gt_classes = cls_feat[top_sim_inds], cls_gt_classes[top_sim_inds]
            # 4. in queue
            batch_size = cls_feat.size(
                0) if ptr + cls_feat.size(0) <= self.ic_loss_queue_size else self.ic_loss_queue_size - ptr
            self.queue[i, ptr:ptr+batch_size] = cls_feat[:batch_size]
            self.queue_label[i, ptr:ptr + batch_size] = cls_gt_classes[:batch_size]

            ptr = ptr + batch_size if ptr + batch_size < self.ic_loss_queue_size else 0
            self.queue_ptr[i] = ptr
            
    @torch.no_grad()
    def _clip_dequeue_and_enqueue(self, feat, gt_classes):

        for i in range(self.num_known_classes):
            ptr = int(self.queue_ptr[i])
            cls_ind = gt_classes == i
            cls_feat, cls_gt_classes = feat[cls_ind], gt_classes[cls_ind]
            # 3. sort by similarity, low sim ranks first
            cls_queue = self.queue[i, self.queue_label[i] != -1]

            _, sim_inds = F.cosine_similarity(
                cls_feat[:, None], cls_queue[None, :], dim=-1).mean(dim=1).sort()
            top_sim_inds = sim_inds[:self.ic_loss_in_queue_size]
            cls_feat, cls_gt_classes = cls_feat[top_sim_inds], cls_gt_classes[top_sim_inds]

            # 4. in queue
            batch_size = cls_feat.size(
                0) if ptr + cls_feat.size(0) <= self.ic_loss_queue_size else self.ic_loss_queue_size - ptr
            self.queue[i, ptr:ptr+batch_size] = cls_feat[:batch_size]
            self.queue_label[i, ptr:ptr + batch_size] = cls_gt_classes[:batch_size]
      
            ptr = ptr + batch_size if ptr + batch_size < self.ic_loss_queue_size else 0
            self.queue_ptr[i] = ptr


    @torch.no_grad()
    def concat_all_gather(self, tensor):
        world_size = comm.get_world_size()
        # single GPU, directly return the tensor
        if world_size == 1:
            return tensor
        # multiple GPUs, gather tensors
        tensors_gather = [torch.ones_like(tensor) for _ in range(world_size)]
        torch.distributed.all_gather(tensors_gather, tensor, async_op=False)
        output = torch.cat(tensors_gather, dim=0)
        return output
        
    def losses(self, predictions, proposals):
        """
        Args:
            predictions: return values of :meth:`forward()`.
            proposals (list[Instances]): proposals that match the features that were used
                to compute predictions. The fields ``proposal_boxes``, ``gt_boxes``,
                ``gt_classes`` are expected.

        Returns:
            Dict[str, Tensor]: dict of losses
        """

        scores, proposal_deltas,mlp_feat,clip_gt_classes,clp_feat = predictions
        
        # parse classification outputs
        gt_classes = (
            cat([p.gt_classes for p in proposals], dim=0) if len(proposals) else torch.empty(0)
        )
        _log_classification_stats(scores, gt_classes)  # Log the classification metrics to EventStorage.
        # parse box regression outputs
        if len(proposals):
            proposal_boxes = cat([p.proposal_boxes.tensor for p in proposals], dim=0)  # Nx4
            assert not proposal_boxes.requires_grad, "Proposals should not require gradients!"
            # If "gt_boxes" does not exist, the proposals must be all negative and
            # should not be included in regression loss computation.
            # Here we just use proposal_boxes as an arbitrary placeholder because its
            # value won't be used in self.box_reg_loss().
            gt_boxes = cat(
                [(p.gt_boxes if p.has("gt_boxes") else p.proposal_boxes).tensor for p in proposals],
                dim=0,
            )
        else:
            proposal_boxes = gt_boxes = torch.empty((0, 4), device=proposal_deltas.device)

        loss_cls = self.evidence_loss(scores, gt_classes,self.evidence_loss_type)
        print("loss_cls",loss_cls)
        losses = {
            "loss_cls": loss_cls,
            "loss_box_reg": self.box_reg_loss(
                proposal_boxes, gt_boxes, proposal_deltas, gt_classes
            ),
        }
        

        ious = cat([p.iou for p in proposals], dim=0)
        # we first store feats in the queue, then cmopute loss
        losses.update(self.get_ic_loss(mlp_feat, gt_classes, ious))
        self._dequeue_and_enqueue(
            mlp_feat, gt_classes, ious, iou_thr=self.ic_loss_queue_iou_thr)
        #losses.update(self.get_ic_loss(mlp_feat, gt_classes, ious))
        losses.update(self.get_clip_loss(clp_feat, clip_gt_classes))
        self._clip_dequeue_and_enqueue(clp_feat, clip_gt_classes)
        losses.update(self.get_clip_loss(clp_feat, clip_gt_classes))

        

        
        return {k: v * self.loss_weight.get(k, 1.0) for k, v in losses.items()}

    def evidence_loss(self, evidence, target, losstype, *, reduction="mean", **kwargs):

        device = evidence.device

        target[target==81]=20
        target=target.to(device)

        if target.numel() == 0 and reduction == "mean":
            return evidence.sum() * 0.0  # connect the gradient
        n = evidence.size(0);c = evidence.size(1)
        alpha = evidence + 1  # Dirichlet distribution parameters
        S = torch.sum(alpha, dim=1)  # Dirichlet strength
        y = torch.zeros(n, c).to(device).scatter_(1, target.unsqueeze(dim=1), 1)
        modified_alpha=y+(1-y)*alpha
        
        storage = get_event_storage()
        #decay_weight = storage.iter / self.max_iters

        regularization_item = 0.05*torch.mean(self.kl_divergence_v2(modified_alpha,21))
        #regularization_item = 0.005*torch.mean(self.kl_divergence(modified_alpha,21))
        
        

        


        if losstype == "ML":

            log_S = torch.unsqueeze(torch.log(S), dim=1).repeat(1, c)
            log_alpha = torch.log(alpha)
            loss = torch.sum(y * (log_S - log_alpha)) / n
            return loss+regularization_item
            

        if losstype == "CE":

            diga_S = torch.unsqueeze(torch.digamma(S), dim=1).repeat(1, c)
            diga_alpha = torch.digamma(alpha)
            loss = torch.sum(y * (diga_S - diga_alpha)) / n
            return loss#+regularization_item
            

        if losstype == "SQ":

            S = torch.unsqueeze(S, dim=1).repeat(1, c)
            p = alpha / S  # expected probability for the k-th singleton
            loss = torch.sum((y - p).pow(2) + p * (1 - p) / (S + 1)) / n
            return loss+regularization_item
            

        if losstype == "Hybrid":
            diga_S = torch.unsqueeze(torch.digamma(S), dim=1).repeat(1, c)
            diga_alpha = torch.digamma(alpha)
            ce_loss = torch.sum(y * (diga_S - diga_alpha)) / n

            
            S = torch.unsqueeze(S, dim=1).repeat(1, c)
            p = alpha / S  # expected probability for the k-th singleton
            mse_loss = torch.sum((y - p).pow(2) + p * (1 - p) / (S + 1)) / n
            
            storage = get_event_storage()
            lamada = torch.min(torch.tensor(1.0, dtype=torch.float32),torch.tensor(storage.iter / 25000,dtype=torch.float32))

            return (1-lamada)*ce_loss+lamada*mse_loss+regularization_item

        if losstype == "focalCE":
            p = alpha / torch.unsqueeze(S, dim=1).repeat(1, c)  # expected probability for the k-th singleton
            diga_S = torch.unsqueeze(torch.digamma(S), dim=1).repeat(1, c)
            diga_alpha = torch.digamma(alpha)  # (n,c)
            #focal_bg=torch.full_like(target, 0.25)
            #focal_fg=torch.full_like(target, 0.75)
            #focal_alpha=torch.where(target==20,focal_bg,focal_fg)
            #focal_alpha=torch.unsqueeze(focal_alpha, dim=1).repeat(1, c)
            loss = torch.sum((1-p+1/torch.unsqueeze(S, dim=1).repeat(1, c))*y * (diga_S - diga_alpha)) / n
            #loss = torch.sum(1*y * (diga_S - diga_alpha)) / n


            return loss#+regularization_item
            
    def kl_divergence(self,modified_alpha, num_classes):
        device = modified_alpha.device
        ones = torch.ones([1, num_classes], dtype=torch.float32, device=device)
        sum_alpha = torch.sum(modified_alpha, dim=1, keepdim=True)

        first_term = (
                torch.lgamma(sum_alpha)
                - torch.lgamma(modified_alpha).sum(dim=1, keepdim=True)
                + torch.lgamma(ones).sum(dim=1, keepdim=True)
                - torch.lgamma(ones.sum(dim=1, keepdim=True))
        )

        second_term = (
            (modified_alpha - ones)
                .mul(torch.digamma(modified_alpha) - torch.digamma(sum_alpha))
                .sum(dim=1, keepdim=True)
        )
        kl = first_term + second_term
        return kl
        
    def kl_divergence_v2(self,modified_alpha, num_classes,thre=e**10):
        device = modified_alpha.device
        mask=(modified_alpha<thre).float()
        mask_modified_alpha=mask+(1-mask)*modified_alpha
        ones = torch.ones([1, num_classes], dtype=torch.float32, device=device)
        sum_alpha = torch.sum(mask_modified_alpha, dim=1, keepdim=True)

        first_term = (
                torch.lgamma(sum_alpha)
                - torch.lgamma(mask_modified_alpha).sum(dim=1, keepdim=True)
                + torch.lgamma(ones).sum(dim=1, keepdim=True)
                - torch.lgamma(ones.sum(dim=1, keepdim=True))
        )

        second_term = (
            (mask_modified_alpha - ones)
                .mul(torch.digamma(mask_modified_alpha) - torch.digamma(sum_alpha))
                .sum(dim=1, keepdim=True)
        )
        kl = first_term + second_term
        return kl      
          
    def new_kl_divergence(self,alpha, y,num_classes):
        device = alpha.device
        sum_alpha = torch.sum(alpha, dim=1, keepdim=True)
        n = alpha.size(0);c = alpha.size(1)
        #y = torch.zeros(n, c).to(device).scatter_(1, target.unsqueeze(dim=1), 1)
        temp=(y*alpha)*0.95+(y*alpha).max(dim=1).values.reshape((n,1))*0.05
        regularization_alpha=torch.clamp(temp,1,)
        first_term = (
                torch.lgamma(sum_alpha)
                - torch.lgamma(alpha).sum(dim=1, keepdim=True)
                + torch.lgamma(torch.sum(regularization_alpha, dim=1, keepdim=True))
                - torch.lgamma(regularization_alpha.sum(dim=1, keepdim=True))
        )

        second_term = (
            (alpha - regularization_alpha)
                .mul(torch.digamma(alpha) - torch.digamma(sum_alpha))
                .sum(dim=1, keepdim=True)
        )
        kl = first_term + second_term
        return kl
        
    def box_reg_loss(self, proposal_boxes, gt_boxes, pred_deltas, gt_classes):
        """
        Args:
            proposal_boxes/gt_boxes are tensors with the same shape (R, 4 or 5).
            pred_deltas has shape (R, 4 or 5), or (R, num_classes * (4 or 5)).
            gt_classes is a long tensor of shape R, the gt class label of each proposal.
            R shall be the number of proposals.
        """
        gt_classes[gt_classes==81]=20
        box_dim = proposal_boxes.shape[1]  # 4 or 5
        # Regression loss is only computed for foreground proposals (those matched to a GT)
        # fg_inds = nonzero_tuple((gt_classes >= 0) & (gt_classes < self.num_classes))[0]
        fg_inds = nonzero_tuple((gt_classes >= 0) & (gt_classes < self.num_known_classes))[
            0]
        if pred_deltas.shape[1] == box_dim:  # cls-agnostic regression
            fg_pred_deltas = pred_deltas[fg_inds]
        else:
            # fg_pred_deltas = pred_deltas.view(-1, self.num_classes, box_dim)[
            #    fg_inds, gt_classes[fg_inds]
            fg_pred_deltas = pred_deltas.view(-1, self.num_known_classes, box_dim)[
                fg_inds, gt_classes[fg_inds]
            ]

        loss_box_reg = _dense_box_regression_loss(
            [proposal_boxes[fg_inds]],
            self.box2box_transform,
            [fg_pred_deltas.unsqueeze(0)],
            [gt_boxes[fg_inds]],
            ...,
            self.box_reg_loss_type,
            self.smooth_l1_beta,
        )

        # The reg loss is normalized using the total number of regions (R), not the number
        # of foreground regions even though the box regression loss is only defined on
        # foreground regions. Why? Because doing so gives equal training influence to
        # each foreground example. To see how, consider two different minibatches:
        #  (1) Contains a single foreground region
        #  (2) Contains 100 foreground regions
        # If we normalize by the number of foreground regions, the single example in
        # minibatch (1) will be given 100 times as much influence as each foreground
        # example in minibatch (2). Normalizing by the total number of regions, R,
        # means that the single example in minibatch (1) and each of the 100 examples
        # in minibatch (2) are given equal influence.
        return loss_box_reg / max(gt_classes.numel(), 1.0)  # return 0 if empty

    def inference(self, predictions: Tuple[torch.Tensor, torch.Tensor],proposals: List[Instances]):
        boxes = self.predict_boxes(predictions, proposals)
        scores = self.predict_probs(predictions, proposals)
        uncertainty = self.predict_uncertainty(predictions, proposals)
        image_shapes = [x.image_size for x in proposals]
        return fast_rcnn_inference(
            boxes,
            scores,
            uncertainty,
            image_shapes,
            self.test_score_thresh,
            self.test_nms_thresh,
            self.test_topk_per_image,
            self.vis_iou_thr,
            self.uncertainty_thr,
        )

    def predict_boxes_for_gt_classes(self, predictions, proposals):

        """
        Args:
            predictions: return values of :meth:`forward()`.
            proposals (list[Instances]): proposals that match the features that were used
                to compute predictions. The fields ``proposal_boxes``, ``gt_classes`` are expected.

        Returns:
            list[Tensor]:
                A list of Tensors of predicted boxes for GT classes in case of
                class-specific box head. Element i of the list has shape (Ri, B), where Ri is
                the number of proposals for image i and B is the box dimension (4 or 5)
        """
        if not len(proposals):
            return []
        scores, proposal_deltas,mlp_feat = predictions
        proposal_boxes = cat([p.proposal_boxes.tensor for p in proposals], dim=0)
        N, B = proposal_boxes.shape
        predict_boxes = self.box2box_transform.apply_deltas(
            proposal_deltas, proposal_boxes
        )

        K = predict_boxes.shape[1] // B
        if K > 1:
            gt_classes = torch.cat([p.gt_classes for p in proposals], dim=0)
            # Some proposals are ignored or have a background class. Their gt_classes
            # cannot be used as index.
            gt_classes = gt_classes.clamp_(0, K - 1)

            predict_boxes = predict_boxes.view(N, K, B)[
                torch.arange(N, dtype=torch.long, device=predict_boxes.device), gt_classes
            ]
        num_prop_per_image = [len(p) for p in proposals]
        return predict_boxes.split(num_prop_per_image)

    def predict_boxes(self, predictions: Tuple[torch.Tensor, torch.Tensor], proposals: List[Instances]):
        """
        Args:
            predictions: return values of :meth:`forward()`.
            proposals (list[Instances]): proposals that match the features that were
                used to compute predictions. The ``proposal_boxes`` field is expected.

        Returns:
            list[Tensor]:
                A list of Tensors of predicted class-specific or class-agnostic boxes
                for each image. Element i has shape (Ri, K * B) or (Ri, B), where Ri is
                the number of proposals for image i and B is the box dimension (4 or 5)
        """
        if not len(proposals):
            return []
        _, proposal_deltas,mlp_feat = predictions
        num_prop_per_image = [len(p) for p in proposals]
        proposal_boxes = cat([p.proposal_boxes.tensor for p in proposals], dim=0)
        predict_boxes = self.box2box_transform.apply_deltas(
            proposal_deltas,
            proposal_boxes,
        )
        return predict_boxes.split(num_prop_per_image)



    def predict_probs(self, predictions: Tuple[torch.Tensor, torch.Tensor], proposals: List[Instances]):


        #20(known)+60(coco)+1(unknown)+1(background)
        num_extra = 60
        scores, _ ,mlp_feat= predictions
        device = scores.device
        num_inst_per_image = [len(p) for p in proposals]
        n = scores.size(0);c = scores.size(1)
        alpha = scores + 1  # Dirichlet distribution parameters
        S = torch.sum(alpha, dim=1)  # Dirichlet strength
        S = torch.unsqueeze(S, dim=1).repeat(1, c)  # (n,c)
        probs = alpha / S  # expected probability for the k-th singleton
        extra_probs = torch.zeros(n, num_extra + 1).to(device)  # 60(coco)+1(unknown)
        probs = torch.cat((probs[:, :c - 1], extra_probs, probs[:, -1:]), 1)
        return probs.split(num_inst_per_image, dim=0)

    def predict_uncertainty(self,predictions,proposals):
        scores, _,mlp_feat = predictions
        num_inst_per_image = [len(p) for p in proposals]
        n = scores.size(0);c = scores.size(1)
        alpha = scores + 1  # Dirichlet distribution parameters
        S = torch.sum(alpha, dim=1)  # Dirichlet strength
        uncertainty= c/S
        return uncertainty.split(num_inst_per_image, dim=0)