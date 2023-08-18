#!/bin/bash

stage=0
spk_id=015

cd ../

if [ $stage -le 0 ]; then

  python3 -W ignore preprocess_wtimit.py \
    --data_directory raw_data/wtimit_train \
    --preprocessed_data_directory wtimit_preprocessed \
    --speaker_ids 015 \
    --config config.json \
    || exit 1
fi

if [ $stage -le 1 ]; then
  if [[ "$OSTYPE" == "darwin"* ]]; then
    export PYTORCH_ENABLE_MPS_FALLBACK=1
  fi
  python3 -W ignore::FutureWarning train.py \
    --corpus wtimit \
    --speaker_A_id ${spk_id} \
    --speaker_B_id ${spk_id} \
    --preprocessed_data_dir wtimit_preprocessed \
    --checkpoint_path checkpoints/model_${spk_id} \
    --config config.json \
    --training_epochs 2000 \
    --stdout_interval 50 \
    --checkpoint_interval 5000 \
    --summary_interval 100 \
    --validation_interval 1000 \
    || exit 1
fi


if [ $stage -le 2 ]; then
  load_iter=00050000
  python3 -W ignore::FutureWarning inference.py \
    --input_wavs_dir raw_data/wtimit_test \
    --speaker_id ${spk_id} \
    --output_dir generated_audio/model_${spk_id}_${load_iter} \
    --checkpoint_file checkpoints/model_${spk_id}/ga_${load_iter} \
    || exit 1
fi

exit 0
