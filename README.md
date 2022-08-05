# Poseur: Direct Human Pose Regression with Transformers


> [**Poseur: Direct Human Pose Regression with Transformers**](https://arxiv.org/pdf/2201.07412.pdf),            
> Weian Mao\*, Yongtao Ge\*, Chunhua Shen, Zhi Tian, Xinlong Wang, Zhibin Wang, Anton van den Hengel  
> In: European Conference on Computer Vision (ECCV), 2022   
> *arXiv preprint ([arXiv 2201.07412](https://arxiv.org/pdf/2201.07412))*  
> (\* equal contribution)

# Introduction
This is a preview for Poseur, which currently including Poseur with R-50 backbone for both training and inference. More models with various backbones will be released soon. This project is bulit upon [MMPose](https://github.com/open-mmlab/mmpose) with commit ID [eeebc652842a9724259ed345c00112641d8ee06d](https://github.com/open-mmlab/mmpose/commit/eeebc652842a9724259ed345c00112641d8ee06d).

# Installation & Quick Start
1. Install following packages
```
pip install easydict, einops
```
2. Follow the [MMPose instruction](mmpose_README.md) to install the project and set up the datasets (MS-COCO).

For training on COCO, run:
```
./tools/dist_train.sh \
configs/body/2d_kpt_sview_rgb_img/poseur/coco/poseur_r50_coco_256x192.py 8 \
--work-dir work_dirs/poseur_r50_coco_256x192
```

For evaluating on COCO, run the following command lines:
```
wget https://cloudstor.aarnet.edu.au/plus/s/UXr1Dn9w6ja4fM9/download -O poseur_256x192_r50_6dec_coco.pth
./tools/dist_test.sh configs/body/2d_kpt_sview_rgb_img/poseur/coco/poseur_r50_coco_256x192.py \
    poseur_256x192_r50_6dec_coco.pth 4 \
    --eval mAP \
    --cfg-options model.filp_fuse_type=\'type2\'
```

For visualizing on COCO, run the following command lines:
```
python demo/top_down_img_demo.py \
    configs/body/2d_kpt_sview_rgb_img/poseur/coco/poseur_r50_coco_256x192.py \
    poseur_256x192_r50_6dec_coco.pth \
    --img-root tests/data/coco/ --json-file tests/data/coco/test_coco.json \
    --out-img-root vis_results_poseur
```

## Models
### COCO Keypoint Detection Results

Name | AP | AP.5| AP.75 |download
--- |:---:|:---:|:---:|:---:
[Poseur_R50_COCO_256x192](configs/body/2d_kpt_sview_rgb_img/poseur/coco/poseur_r50_coco_256x192.py)| 75.5  | 90.7 |82.6 | [model](https://cloudstor.aarnet.edu.au/plus/s/UXr1Dn9w6ja4fM9/download)


*Disclaimer:*

- Due to the update of MMPose, the result of R50 is slightly higher than our original paper.

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

## License

For academic use, this project is licensed under the 2-clause BSD License. For commercial use, please contact [Chunhua Shen](mailto:chhshen@gmail.com).