# Copyright (c) Facebook, Inc. and its affiliates.
import logging
from typing import Dict, List

import numpy as np
import torch
import torch.nn.functional as F
from detectron2.config import configurable
from detectron2.layers import ShapeSpec
from detectron2.modeling.poolers import ROIPooler
from detectron2.modeling.roi_heads.box_head import build_box_head
from detectron2.modeling.roi_heads.roi_heads import (
    ROI_HEADS_REGISTRY, StandardROIHeads, add_ground_truth_to_proposals)
from detectron2.structures import Boxes, Instances, pairwise_iou
from detectron2.utils.events import get_event_storage
from detectron2.utils.registry import Registry
from torch import nn
from detectron2.utils.visualizer import Visualizer
from .fast_rcnn_clip import build_roi_box_output_layers
import torchvision.transforms as transforms
import clip
from typing import Optional


logger = logging.getLogger(__name__)


@ROI_HEADS_REGISTRY.register()
class OpenSetStandardROIHeads(StandardROIHeads):

    @torch.no_grad()
    def label_and_sample_proposals(self, proposals: List[Instances], targets: List[Instances]) -> List[Instances]:
        if self.proposal_append_gt:
            proposals = add_ground_truth_to_proposals(targets, proposals)

        proposals_with_gt = []

        num_fg_samples = []
        num_bg_samples = []
        for proposals_per_image, targets_per_image in zip(proposals, targets):
            has_gt = len(targets_per_image) > 0
            match_quality_matrix = pairwise_iou(
                targets_per_image.gt_boxes, proposals_per_image.proposal_boxes
            )
            matched_idxs, matched_labels = self.proposal_matcher(
                match_quality_matrix)
            sampled_idxs, gt_classes = self._sample_proposals(
                matched_idxs, matched_labels, targets_per_image.gt_classes
            )

            # Set target attributes of the sampled proposals:
            proposals_per_image = proposals_per_image[sampled_idxs]
            proposals_per_image.gt_classes = gt_classes
            print("gt_classes",gt_classes)
            # NOTE: add iou of each proposal
            ious, _ = match_quality_matrix.max(dim=0)
            proposals_per_image.iou = ious[sampled_idxs]

            if has_gt:
                sampled_targets = matched_idxs[sampled_idxs]
                for (trg_name, trg_value) in targets_per_image.get_fields().items():
                    if trg_name.startswith("gt_") and not proposals_per_image.has(trg_name):
                        proposals_per_image.set(
                            trg_name, trg_value[sampled_targets])

            num_bg_samples.append(
                (gt_classes == self.num_classes).sum().item())
            num_fg_samples.append(gt_classes.numel() - num_bg_samples[-1])
            proposals_with_gt.append(proposals_per_image)

        # Log the number of fg/bg samples that are selected for training ROI heads
        storage = get_event_storage()
        storage.put_scalar("roi_head/num_fg_samples", np.mean(num_fg_samples))
        print("roi_head/num_fg_samples", np.mean(num_fg_samples))
        storage.put_scalar("roi_head/num_bg_samples", np.mean(num_bg_samples))
        print("roi_head/num_bg_samples", np.mean(num_bg_samples))

        return proposals_with_gt

    @classmethod
    def _init_box_head(cls, cfg, input_shape):
        # fmt: off
        in_features = cfg.MODEL.ROI_HEADS.IN_FEATURES
        pooler_resolution = cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION
        pooler_scales = tuple(1.0 / input_shape[k].stride for k in in_features)
        sampling_ratio = cfg.MODEL.ROI_BOX_HEAD.POOLER_SAMPLING_RATIO
        pooler_type = cfg.MODEL.ROI_BOX_HEAD.POOLER_TYPE
        # fmt: on

        # If StandardROIHeads is applied on multiple feature maps (as in FPN),
        # then we share the same predictors and therefore the channel counts must be the same
        in_channels = [input_shape[f].channels for f in in_features]
        # Check all channel counts are equal
        assert len(set(in_channels)) == 1, in_channels
        in_channels = in_channels[0]

        box_pooler = ROIPooler(
            output_size=pooler_resolution,
            scales=pooler_scales,
            sampling_ratio=sampling_ratio,
            pooler_type=pooler_type,
        )

        box_head = build_box_head(
            cfg, ShapeSpec(channels=in_channels,
                           height=pooler_resolution, width=pooler_resolution)
        )
        # register output layers
        box_predictor = build_roi_box_output_layers(cfg, box_head.output_shape)
        return {
            "box_in_features": in_features,
            "box_pooler": box_pooler,
            "box_head": box_head,
            "box_predictor": box_predictor,
        }

    def get_clip_feature(
        self,
        images,#: ImageList
        targets#: Optional[List[Instances]]
    ):

        proposals_clip_feature=[]
        clip_gt_classes=[]

        for i in range(len(images)):

            temp_image=images[i]
            clip_model, _ = clip.load("ViT-B/32", device=temp_image.device)
            mean=torch.tensor([103.530, 116.280, 123.675], device=temp_image.device)
            temp_image=(temp_image+mean[:,None, None])/255
            temp_image=temp_image[ [2, 1, 0],...]
                

            normalize = transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711])
            for j in range(len(targets[i])):

                x0,y0,x1,y1=targets[i][j].get('gt_boxes').tensor.squeeze()

                gt_class=targets[i][j].get('gt_classes')
                if gt_class==81:
                    continue
                instance_image=temp_image[:,int(y0.item()):int(y1.item()),int(x0.item()):int(x1.item())]

                instance_image= F.interpolate(instance_image.unsqueeze(0), size=(clip_model.visual.input_resolution, clip_model.visual.input_resolution), mode='bicubic', align_corners=False)#resize
                normalized_tensor = normalize(instance_image)
                with torch.no_grad():
                    instance_feature = clip_model.encode_image(normalized_tensor)
                    print("instance_feature",instance_feature.shape)
                clip_gt_classes.append(gt_class)
                proposals_clip_feature.append(instance_feature.cuda())
            del clip_model
        return torch.cat(clip_gt_classes,dim=0),torch.cat(proposals_clip_feature,dim=0)

    def forward(
        self,
        images,#: ImageList
        features: Dict[str, torch.Tensor],
        proposals: List[Instances],
        targets:Optional=None,#: Optional[List[Instances]] = None
    ) :#-> Tuple[List[Instances], Dict[str, torch.Tensor]]
        """
        See :class:`ROIHeads.forward`.
        """
        
        if self.training:
            assert targets, "'targets' argument is required during training"
            proposals = self.label_and_sample_proposals(proposals, targets)
            clip_gt_classes,proposals_clip_feature=self.get_clip_feature(images,targets)#list
            

                    
        del images
        del targets

        if self.training:
            losses = self._forward_box(features, proposals,clip_gt_classes,proposals_clip_feature)
            # Usually the original proposals used by the box head are used by the mask, keypoint
            # heads. But when `self.train_on_pred_boxes is True`, proposals will contain boxes
            # predicted by the box head.
            losses.update(self._forward_mask(features, proposals))
            losses.update(self._forward_keypoint(features, proposals))
            return proposals, losses
        else:
            pred_instances = self._forward_box(features, proposals)
            # During inference cascaded prediction is used: the mask and keypoints heads are only
            # applied to the top scoring box detections.
            pred_instances = self.forward_with_given_boxes(features, pred_instances)
            return pred_instances, {}

    def _forward_box(self, features: Dict[str, torch.Tensor], proposals: List[Instances],clip_gt_classes:Optional=None,proposals_clip_feature:Optional=None):
        """
        Forward logic of the box prediction branch. If `self.train_on_pred_boxes is True`,
            the function puts predicted boxes in the `proposal_boxes` field of `proposals` argument.

        Args:
            features (dict[str, Tensor]): mapping from feature map names to tensor.
                Same as in :meth:`ROIHeads.forward`.
            proposals (list[Instances]): the per-image object proposals with
                their matching ground truth.
                Each has fields "proposal_boxes", and "objectness_logits",
                "gt_classes", "gt_boxes".

        Returns:
            In training, a dict of losses.
            In inference, a list of `Instances`, the predicted instances.
        """
        features = [features[f] for f in self.box_in_features]
        box_features = self.box_pooler(features, [x.proposal_boxes for x in proposals])
        box_features = self.box_head(box_features)
        if self.training:
            predictions = self.box_predictor(box_features,clip_gt_classes,proposals_clip_feature)
        else:
            predictions = self.box_predictor(box_features,clip_gt_classes)
        del box_features

        if self.training:

            losses = self.box_predictor.losses(predictions, proposals)
            # proposals is modified in-place below, so losses must be computed first.
            if self.train_on_pred_boxes:
                with torch.no_grad():
                    pred_boxes = self.box_predictor.predict_boxes_for_gt_classes(
                        predictions, proposals
                    )
                    for proposals_per_image, pred_boxes_per_image in zip(proposals, pred_boxes):
                        proposals_per_image.proposal_boxes = Boxes(pred_boxes_per_image)
            return losses
        else:
            pred_instances, _ = self.box_predictor.inference(predictions, proposals)
            return pred_instances

@ROI_HEADS_REGISTRY.register()
class DropoutStandardROIHeads(OpenSetStandardROIHeads):
    @configurable
    def __init__(self, *args, **kwargs,):
        super().__init__(*args, **kwargs)
        # num of sampling
        self.num_sample = 30

    def _forward_box(self, features: Dict[str, torch.Tensor], proposals: List[Instances], targets=None):

        features = [features[f] for f in self.box_in_features]
        box_features = self.box_pooler(
            features, [x.proposal_boxes for x in proposals])
        box_features = self.box_head(box_features)

        # if testing, we run multiple inference for dropout sampling
        if self.training:
            predictions = self.box_predictor(box_features)
        else:
            predictions = [self.box_predictor(
                box_features, testing=True) for _ in range(self.num_sample)]

        del box_features

        if self.training:
            losses = self.box_predictor.losses(predictions, proposals)
            # proposals is modified in-place below, so losses must be computed first.
            if self.train_on_pred_boxes:
                with torch.no_grad():
                    pred_boxes = self.box_predictor.predict_boxes_for_gt_classes(
                        predictions, proposals
                    )
                    for proposals_per_image, pred_boxes_per_image in zip(proposals, pred_boxes):
                        proposals_per_image.proposal_boxes = Boxes(
                            pred_boxes_per_image)
            return losses
        else:
            pred_instances, _ = self.box_predictor.inference(
                predictions, proposals)
            return pred_instances
