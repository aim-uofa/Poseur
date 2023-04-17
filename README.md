# Poseur: Direct Human Pose Regression with Transformers


> [**Poseur: Direct Human Pose Regression with Transformers**](https://arxiv.org/pdf/2201.07412.pdf),            
> Weian Mao\*, Yongtao Ge\*, Chunhua Shen, Zhi Tian, Xinlong Wang, Zhibin Wang, Anton van den Hengel  
> In: European Conference on Computer Vision (ECCV), 2022   
> *arXiv preprint ([arXiv 2201.07412](https://arxiv.org/pdf/2201.07412))*  
> (\* equal contribution)

## News :triangular_flag_on_post:
[2023/04/17] Release a naive version of Poseur based on ViT backbone. Please see [poseur_vit_base_coco_256x192](configs/poseur/coco/poseur_vit_base_coco_256x192.py).

[2023/04/17] Release a naive version of Poseur trained on COCO-Wholebody dataset. Please see [poseur_coco_wholebody](configs/poseur/coco_wholebody/).

# Introduction
This project is bulit upon [MMPose](https://github.com/open-mmlab/mmpose) with commit ID [eeebc652842a9724259ed345c00112641d8ee06d](https://github.com/open-mmlab/mmpose/commit/eeebc652842a9724259ed345c00112641d8ee06d).

# Installation & Quick Start
1. Install following packages
```
pip install easydict einops
```
2. Follow the [MMPose instruction](mmpose_README.md) to install the project and set up the datasets (MS-COCO).

For training on COCO, run:
```
./tools/dist_train.sh \
configs/poseur/coco/poseur_r50_coco_256x192.py 8 \
--work-dir work_dirs/poseur_r50_coco_256x192
```

For evaluating on COCO, run the following command lines:
```
wget https://cloudstor.aarnet.edu.au/plus/s/UXr1Dn9w6ja4fM9/download -O poseur_256x192_res50_6dec_coco.pth
./tools/dist_test.sh configs/poseur/coco/poseur_res50_coco_256x192.py \
    poseur_256x192_r50_6dec_coco.pth 4 \
    --eval mAP \
    --cfg-options model.filp_fuse_type=\'type2\'
```

For visualizing on COCO, run the following command lines:
```
python demo/top_down_img_demo.py \
    configs/poseur/coco/poseur_res50_coco_256x192.py \
    poseur_256x192_res50_6dec_coco.pth \
    --img-root tests/data/coco/ --json-file tests/data/coco/test_coco.json \
    --out-img-root vis_results_poseur
```

## COCO Keypoint Detection

Name | AP | AP.5| AP.75 |download link
--- |:---:|:---:|:---:|:---:
[poseur_mobilenetv2_coco_256x192](configs/poseur/coco/poseur_mobilenetv2_coco_256x192.py)| 71.9  | 88.9 |78.6 | [model](https://cloudstor.aarnet.edu.au/plus/s/L198TFFqwWYsSop/download)
[poseur_mobilenetv2_coco_256x192_12dec](configs/poseur/coco/poseur_mobilenetv2_coco_256x192_12dec.py)| 72.3  | 88.9 |78.9 | [model](https://cloudstor.aarnet.edu.au/plus/s/sw0II7qSQDjJ88h/download)
[poseur_res50_coco_256x192](configs/poseur/coco/poseur_res50_coco_256x192.py)| 75.5  | 90.7 |82.6 | [model](https://cloudstor.aarnet.edu.au/plus/s/UXr1Dn9w6ja4fM9/download)
[poseur_hrnet_w32_coco_256x192](configs/poseur/coco/poseur_hrnet_w32_coco_256x192.py)| 76.8  | 91.0 |83.5 | [model](https://cloudstor.aarnet.edu.au/plus/s/xMvCnp5lb2MR7S4/download)
[poseur_hrnet_w48_coco_384x288](configs/poseur/coco/poseur_hrnet_w48_coco_384x288.py)| 78.7  | 91.6 |85.1 | [model](https://cloudstor.aarnet.edu.au/plus/s/IGXy98TZlJYerNc/download)
[poseur_hrformer_tiny_coco_256x192_3dec](configs/poseur/coco/poseur_hrformer_tiny_coco_256x192_3dec.py)| 74.2  | 90.1 |81.4 | [model](https://cloudstor.aarnet.edu.au/plus/s/CpGYghZQX3mv32i/download)
[poseur_hrformer_small_coco_256x192_3dec](configs/poseur/coco/poseur_hrformer_small_coco_256x192_3dec.py)| 76.6  | 91.0 |83.4 | [model](https://cloudstor.aarnet.edu.au/plus/s/rK2s3fdrpeP9k6l/download)
[poseur_hrformer_big_coco_256x192](configs/poseur/coco/poseur_hrformer_big_coco_256x192.py)| 78.9  | 91.9 |85.6 | [model](https://cloudstor.aarnet.edu.au/plus/s/34udjbTr9p9Aigo/download)
[poseur_hrformer_big_coco_384x288](configs/poseur/coco/poseur_hrformer_big_coco_384x288.py)| 79.6  | 92.1 |85.9 | [model](https://cloudstor.aarnet.edu.au/plus/s/KST3aSAlGd8PJpQ/download)
[poseur_vit_base_coco_256x192](configs/poseur/coco/poseur_vit_base_coco_256x192.py)| 76.7  | 90.6 |83.5 | [model](https://cloudstor.aarnet.edu.au/plus/s/46foUsIwzmHiVmb/download)


## COCO-WholeBody Benchmark (V0.5)

Compare Whole-body pose estimation results with other methods.

|Method           |  body |       | foot  |       | face  |       |  hand |       | whole |       |
|-----------------| ------| ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- | 
|                 |  AP   | AR    | AP    | AR    |  AP   | AR    | AP    | AR    | AP    | AR    |
|OpenPose [1]     | 0.563 | 0.612 | 0.532 | 0.645 | 0.482 | 0.626 | 0.198 | 0.342 | 0.338 | 0.449 |
|HRNet [2]        | 0.659 | 0.709 | 0.314 | 0.424 | 0.523 | 0.582 | 0.300 | 0.363 | 0.432 | 0.520 |
|HRNet-body [2]   | 0.758 | 0.809 |   -   |   -   |   -   |   -   |   -   |   -   |   -   |   -   |
|ZoomNet [3]      | 0.743 | 0.802 | 0.798 | 0.869 | 0.623 | 0.701 | 0.401 | 0.498 | 0.541 | 0.658 |
|ZoomNas [4]      | 0.740 |  -     | 0.617 |   -    | 0.889 |    -   | 0.625 |   -    | 0.654 |  -   |
|RTMPose [5]      | 0.730 |   -    | 0.734 |   -    | 0.898 |    -   | 0.587 |   -    | 0.669 |  -   |
|Poseur_ResNet50  | 0.655 | 0.732 | 0.615 | 0.742 | 0.844 | 0.900 | 0.560 | 0.673 | 0.587 | 0.681 |
|Poseur_HRNet_W32 | 0.680 | 0.753 | 0.668 | 0.780 | 0.863 | 0.912 | 0.604 | 0.706 | 0.620 | 0.707 |
|Poseur_HRNet_W48 | 0.692 | 0.766 | 0.689 | 0.799 | 0.861 | 0.911 | 0.621 | 0.721 | 0.633 | 0.721 |

### COCO-WholeBody Pretrain Models

Name | AP | AP.5| AP.75 |download link
--- |:---:|:---:|:---:|:---:
[poseur_res50_coco_wholebody_256x192](configs/poseur/coco_wholebody/res50_coco_wholebody_256x192_poseur.py)| 65.5 | 85.0 | 71.8 | [model](https://cloudstor.aarnet.edu.au/plus/s/pLNFGWavdFAji5J/download)
[poseur_hrnet_w32_coco_wholebody_256x192](configs/poseur/coco_wholebody/hrnet_w32_coco_wholebody_256x192_poseur.py)| 68.0  | 85.8 | 74.4 | [model](https://cloudstor.aarnet.edu.au/plus/s/eGfWFWYasRtoFo5/download)
[poseur_hrnet_w48_coco_wholebody_256x192](configs/poseur/coco_wholebody/hrnet_w48_coco_wholebody_256x192_poseur.py)| 69.2  | 86.0 | 75.7 | [model](https://cloudstor.aarnet.edu.au/plus/s/LBokqDr1DK7s7C4/download)


*Disclaimer:*

- Due to the update of MMPose, the results are slightly different from our original paper.
- We use the official HRFormer implement from [here](https://github.com/HRNet/HRFormer/tree/main/pose), the implementation in mmpose has not been verified by us.

# Citations
Please consider citing our papers in your publications if the project helps your research. BibTeX reference is as follows.
```BibTeX
@inproceedings{mao2022poseur,
  title={Poseur: Direct human pose regression with transformers},
  author={Mao, Weian and Ge, Yongtao and Shen, Chunhua and Tian, Zhi and Wang, Xinlong and Wang, Zhibin and Hengel, Anton van den},
  journal = {Proceedings of the European Conference on Computer Vision {(ECCV)}},
  month = {October},
  year={2022}
}
```

## Reference
```
[1] Andriluka, M., Pishchulin, L., Gehler, P., Schiele, B.: 2d human pose estimation: New benchmark and state of the art analysis. In: Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR) (2014)
[2] Sun, K., Xiao, B., Liu, D., Wang, J.: Deep high-resolution representation learning for human pose estimation. arXiv preprint arXiv:1902.09212 (2019)
[3] Sheng Jin, Lumin Xu, Jin Xu, Can Wang, Wentao Liu, Chen Qian, Wanli Ouyang, Ping Luo. Whole-Body Human Pose Estimation in the Wild. (ECCV) (2020)
[4] Lumin Xu, Sheng Jin, Wentao Liu, Chen Qian, Wanli Ouyang, Ping Luo, Xiaogang Wang: ZoomNAS: Searching for Whole-body Human Pose Estimation in the Wild In: IEEE Transactions on Pattern Analysis and Machine Intelligence (TPAMI) (2022)
[5] Tao Jiang, Peng Lu, Li Zhang, Ningsheng Ma, Rui Han, Chengqi Lyu, Yining Li, Kai Chen. RTMPose: Real-Time Multi-Person Pose Estimation based on MMPose. arXiv preprint arXiv:2303.07399 (2023)
```

## License

For commercial use, please contact [Chunhua Shen](mailto:chhshen@gmail.com).