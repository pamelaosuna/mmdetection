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

TRAIN_DATA=S1A2
EVAL_DATA=S1A2
export DATAROOT=/scratch/dldevel/osuna/spines/trackformer/data/spine_${EVAL_DATA}/
EPOCH=15

python3 -u tools/test.py projects/CO-DETR/configs/codino/co_dino_5scale_r50_8xb2_1x_coco_SPINE.py \
 work_dirs/co_dino_${TRAIN_DATA}/best_coco_spine_f1_epoch_${EPOCH}.pth \
 --work-dir results/co_dino_${TRAIN_DATA} \
 --out results/co_dino_${TRAIN_DATA}/valid_${EVAL_DATA}.pkl
#  --show --show-dir /scratch/dldevel/osuna/mmdetection/results

echo "job finished"
date