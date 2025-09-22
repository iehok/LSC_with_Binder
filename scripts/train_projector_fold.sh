#!/bin/bash

METHOD='projector_fold'
RUN_NAME='projector-model=mlp-span=19602000-fold=0'

python -B -m src.train_projector_fold \
    --accumulate_grad_batches 1 \
    --batch_size 16 \
    --check_val_every_n_epoch 1 \
    --embedding_model bert-base-uncased \
    --fold 0 \
    --gpus 1 \
    --log_every_n_steps 1 \
    --lr 1e-3 \
    --max_epochs 100 \
    --method $METHOD \
    --num_nodes 1 \
    --precision 16 \
    --projector_model mlp \
    --run_name $RUN_NAME \
    --strategy deepspeed_stage_2 \
    --target_span 1960-2000 \
    --val_check_interval 1.0 \
    --wandb
