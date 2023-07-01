# Fuzzy-NMS
## Project purpose
Implementation of the paper "Fuzzy-NMS: Improving 3D Object Detection with Fuzzy Classification in NMS" on Openpcdet.

![fig1](D:\Desktop\Fuzzy-NMS\Fuzzy-NMS\figure\fig1.png)

## Installation
### Requirements

All the codes are tested in the following environment:

- Linux (tested on Ubuntu 14.04/16.04/18.04/20.04/21.04)
- Python 3.6+
- PyTorch 1.1 or higher (tested on PyTorch 1.1, 1,3, 1,5~1.10)
- CUDA 9.0 or higher (PyTorch 1.3+ needs CUDA 9.2+)
- [`spconv v1.0 (commit 8da6f96)`](https://github.com/traveller59/spconv/tree/8da6f967fb9a054d8870c3515b1b44eca2103634) or [`spconv v1.2`](https://github.com/traveller59/spconv) or [`spconv v2.x`](https://github.com/traveller59/spconv)

Please refer to the official installation steps of Openpcdet.



## Introduction to Core Documentation
```
pcdet
├── datasets
├── models
│   ├── backbones_2d
│   ├── backbones_3d
│   ├── dense_heads
│   ├── detectors
│		├── detector3d_template.py #A part is added in lines 283 to 294, and the dictionary composed of nms  │                                   results is counted, and a return value nms_dicts is added to the          │                 ┇                 post_processing function.
│		         
│		└── pointpillar.py #A return value nms_dicts is added to the forward function, and the same is true │                           for other baseline detector files.		          
│   ├── model_utils
│		├── fuzzy_code
│       	├── cpp_fuzzy.py #Loads a dynamic link library to classify objects by size and density.
│       	├── DBSCAN_plot.py #Plot the DBSCAN process.
│       	├── libfuzzy.so #C++ dynamic link library for fuzzy classification.
│		├── fuzzy_nms
│       	├── model_nms_utils_cpp.py #If it is a single run, you need to replace the code in this file 	│										with model_nms_utils.py in the upper directory.
│		├── model_nms_utils.py #The class_agnostic_nms function has been rewritten and a return value has   │                               been added.
├── ops
│   ├── iou3d_nms 
│		└──iou3d_nms_utils.py #Added soft-nms and Diou-nms.
tools
├── eval_utils
│   ├── eval_utils_fuzzy.py #It needs to be used when traversing to find the optimal parameters.
├── test.py #The eval_single_ckpt function needs to be modified when selecting a single run or a traversal    │            parameter.
```




## Getting Started

Please refer to the official documents of Openpcdet to prepare the Kitti and Waymo datasets.

## Test 

```
cd tools
CUDA_VISIBLE_DEVICES=1 python test.py --cfg_file cfgs/kitti_models/pointpillar.yaml --batch_size 4 --ckpt pointpillar.pth --extra_tag nms_test
```

## Results 

### KITTI 3D Object Detection Baselines

Selected supported methods are shown in the below table. The results are the 3D detection performance of moderate difficulty on the *test* set of KITTI dataset.

|              |  Car  | Pedestrian | Cyclist |
| ------------ | :---: | :--------: | :-----: |
| PointPillars | 73.13 |   34.16    |  54.68  |
| PV-RCNN      | 78.71 |   40.01    |  62.05  |
| IA-SSD       | 78.93 |   39.95    |  60.28  |
| GD-MAE       | 76.03 |   36.46    |  54.99  |
| BiProDet     | 81.77 |   45.71    |  64.01  |

### KITTI BEV Object Detection Baselines

Selected supported methods are shown in the below table. The results are the BEV detection performance of moderate difficulty on the *test* set of KITTI dataset.

|              |  Car  | Pedestrian | Cyclist |
| ------------ | :---: | :--------: | :-----: |
| PointPillars | 86.79 |   39.74    |  62.20  |
| PV-RCNN      | 87.71 |   45.37    |  66.12  |
| IA-SSD       | 89.01 |   45.68    |  67.72  |
| GD-MAE       | 88.38 |   41.32    |  62.46  |
| BiProDet     | 89.02 |   50.97    |  70.34  |

### Waymo Open Dataset Baselines

The following three baselines are all tested on *val* set of KITTI dataset.

|              |   Vec_L1    |   Vec_L2    |   Ped_L1    |   Ped_L2    |   Cyc_L1    |   Cyc_L2    |
| ------------ | :---------: | :---------: | :---------: | :---------: | :---------: | :---------: |
| PointPillars | 70.15/69.53 | 62.05/61.50 | 67.61/46.87 | 60.31/41.74 | 57.78/54.02 | 56.19/52.53 |
| SECOND       | 69.07/68.44 | 60.85/60.29 | 65.15/53.73 | 58.29/47.98 | 53.72/52.24 | 52.35/50.91 |
| M3DETR       | 76.58/75.92 | 69.24/68.60 | 67.10/57.73 | 59.00/50.68 | 68.12/66.72 | 66.46/65.09 |

