python build_prior_maps_casia2_clip.py \
  --root /home/hpclp/disk/Graphgpt/dataset/CASIA2 \
  --clip-encoder-path /home/hpclp/disk/Graphgpt/sesame/model/llava/model/multimodal_encoder/clip_encoder.py \
  --vision-tower /home/hpclp/disk/q/models/clip-vit-large-patch14-336 \
  --patchcore-src src \
  --intermediate-layer 15 \
  --device cuda \
  --include-tp-bg
