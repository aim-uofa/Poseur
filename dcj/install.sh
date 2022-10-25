conda create --name openmmlab python=3.8 -y
conda activate openmmlab
pip install https://download.pytorch.org/whl/cu101/torch-1.8.1%2Bcu101-cp38-cp38-linux_x86_64.whl
pip install https://download.pytorch.org/whl/cu101/torchvision-0.9.1%2Bcu101-cp38-cp38-linux_x86_64.whl
pip install -U openmim
mim install mmcv-full==1.6.0
pip install easydict einops
pip install timm
pip install future tensorboard
mkdir checkpoints
mkdir outputs
mkdir data 