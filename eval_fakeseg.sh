#!/bin/bash

CUDA_VISIBLE_DEVICES="3" python eval_fakenews.py \
  --version ./runs/FAKESEG/hg_model \
  --dataset_dir /home/hpclp/disk/Graphgpt/dataset \
  --vision-tower /home/hpclp/disk/q/models/clip-vit-large-patch14-336 \
  --vision_pretrained /home/hpclp/disk/q/models/sam_vit_h_4b8939/sam_vit_h_4b8939.pth \
  --save_pred_masks \
  --save_metrics_csv \
  --precision bf16 \
  --image_size 1024 \
  --conv_type llava_v1 \