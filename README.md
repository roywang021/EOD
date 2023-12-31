# EOD

Code for AAAI2024 paper: Towards Evidential and Class Separable Open Set Object Detection


The Evidential Object Detector (EOD) is implemented using the [Detectron2](https://github.com/facebookresearch/detectron2) object detection framework and follows the open-set configuration of [Opendet](https://github.com/csuhan/opendet2). Gratitude is extended to the original authors for their valuable contributions and commitment to open source.

### Basic Setup
The fundamental setup tasks (e.g., installation and dataset preparation) can be easily accomplished by referring to the guidelines provided in the two aforementioned projects.

### Usage
The essential code from the paper is provided. This enables straightforward reproduction through the following steps.

1. Identify the modules that need to be replaced in the Faster-RCNN according to the detection framework and open-set settings.

2. Copy the corresponding files or folders from this repository to the appropriate locations in the project.

3. Make modifications and adaptations as needed.

### Train and Test
#### Training
The training process is the same as Detectron2 and Opendet.
```bash
python tools/train_net.py --num-gpus 8 --config-file configs/faster_rcnn_R_50_FPN_3x_EOD.yaml
```
#### Testing
Run the following command for testing.
```bash
python tools/train_net.py --num-gpus 8 --config-file configs/faster_rcnn_R_50_FPN_3x_EOD.yaml --eval-only MODEL.WEIGHTS output/faster_rcnn_R_50_FPN_3x_EOD/model_test.pth
```
