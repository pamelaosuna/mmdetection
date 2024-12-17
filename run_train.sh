#!/bin/bash

#SBATCH --job-name=mmdet
#SBATCH --output=out_sbatch/%j.out
#SBATCH --partition=gpu2
#SBATCH --gres=gpu:1
#SBATCH --time=3-00:00:00

module load nvidia/cuda/12.3.0
echo "load conda environment"
eval "$(/scratch/dldevel/osuna/miniconda3/bin/conda shell.bash hook)"
conda activate openmmlab
which python3
echo "loaded conda environment"
echo "start training..."
date

WHICH_DATA=S1A2
export DATAROOT=/scratch/dldevel/osuna/spines/trackformer/data/spine_${WHICH_DATA}/

# python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.version.cuda)"
# python -c "import mmcv; print(mmcv.__version__)"
python3 -u tools/train.py projects/CO-DETR/configs/codino/co_dino_5scale_r50_8xb2_1x_coco_SPINE.py \
 --work-dir work_dirs/co_dino_${WHICH_DATA}

echo "job finished"
date