#!/bin/bash

/home/lwong/Toolkits/Anaconda3/envs/dev_torch2/bin/python -um guild.op_main main --skip-posttrain-inference -- \
  --controller_cfg.cp_load_dir None \
  --controller_cfg.debug_mode 1 \
  --controller_cfg.debug_validation "" \
  --controller_cfg.id_list example_patient_split.ini \
  --controller_cfg.id_list_val None \
  --controller_cfg.sequence "T2WFS" \
  --controller_cfg.version "v1.0" \
  --data_loader_cfg.force_augment "" \
  --data_loader_cfg.id_globber "[a-zA-Z]{0,5}\d+" \
  --data_loader_cfg.input_dir ./DataDir/60.Large-Study/v1-All-Data/Normalized_2/T2WFS_TRA/01.NyulNormalized/ \
  --data_loader_cfg.probmap_dir ./DataDir/60.Large-Study/v1-All-Data/Normalized_2/T2WFS_TRA/00.HuangMask/ \
  --data_loader_cfg.target_dir ./DataDir/0B.Segmentations/T2WFS_TRA/03.AI_Generated_ManualCorrected/ \
  --solver_cfg.batch_size 40 \
  --solver_cfg.batch_size_val 40 \
  --solver_cfg.class_weights "0.01 1.0" \
  --solver_cfg.decay_init_epoch 0 \
  --solver_cfg.decay_on_plateau "" \
  --solver_cfg.decay_rate_LR 1 \
  --solver_cfg.early_stop loss_reference \
  --solver_cfg.early_stop_kwargs "{'warmup':5,'patience':15}" \
  --solver_cfg.init_lr 1e-05 \
  --solver_cfg.num_of_epochs 100 \
  --solver_cfg.optimizer Adam \
  --solver_cfg.sigmoid_params.cap 0.2 \
  --solver_cfg.sigmoid_params.delay 5 \
  --solver_cfg.sigmoid_params.stretch 1