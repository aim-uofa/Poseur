# CUDA_VISIBLE_DEVICES=2 python demo/top_down_img_demo.py \
#     configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/hrnet_w48_coco_256x192.py \
#     https://download.openmmlab.com/mmpose/top_down/hrnet/hrnet_w48_coco_256x192-b9e0b3ab_20200708.pth \
#     --img-root tests/data/coco/ --json-file tests/data/coco/test_coco.json \
#     --out-img-root vis_results

CUDA_VISIBLE_DEVICES=2 python demo/top_down_img_demo.py \
    configs/body/2d_kpt_sview_rgb_img/poseur/coco/poseur_r50_coco_256x192.py \
    ckpts/poseur_256x192_r50_6dec_coco.pth \
    --img-root tests/data/coco/ --json-file tests/data/coco/test_coco.json \
    --out-img-root vis_results_poseur