#!/bin/bash

# AnimeGANv2 models
declare -a models=("generator_Hayao_weight" "generator_Shinkai_weight" "generator_Paprika_weight")

# Input directory containing the images
input_dir="/data/dataset/input/animeganv2"

# Output directory to save the transformed images
output_dir="/data/dataset/output/animeganv2"

# Loop over all models
for model in "${models[@]}"; do
  # Prepare directory for this model's output
  model_output_dir="${output_dir}/${model}"
  mkdir -p "${model_output_dir}"

  # Run AnimeGANv2 on the input images
  python test.py --checkpoint_dir "checkpoint/${model}" --test_dir "${input_dir}" --save_dir "${model_output_dir}"
done
