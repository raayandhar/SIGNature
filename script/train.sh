#!/bin/bash

DATA_PATH="/data/edward/SIGNature-data"

# deepfake - Single GPU training
# python train_classifier.py --device_num 1 --per_gpu_batch_size 32 --total_epoch 50 --lr 2e-5 --warmup_steps 2000\
#     --model_name princeton-nlp/unsup-simcse-roberta-base --dataset deepfake --path ${DATA_PATH}/Deepfake/cross_domains_cross_models \
#     --name deepfake-roberta-base --freeze_embedding_layer --database_name train --test_dataset_name test \
#     --use_signature --wandb_project "signature-deepfake"

# # TuringBench - Single GPU training
# python train_classifier.py --device_num 1 --per_gpu_batch_size 32 --total_epoch 50 --lr 2e-5 --warmup_steps 2000\
#     --model_name princeton-nlp/unsup-simcse-roberta-base --dataset TuringBench --path ${DATA_PATH}/TuringBench/AA \
#     --name TuringBench-roberta-base --freeze_embedding_layer --database_name train --test_dataset_name test \
#     --use_signature --wandb_project "signature-turingbench"

# M4-monolingual - Single GPU training
# python train_classifier.py --device_num 1 --per_gpu_batch_size 32 --total_epoch 50 --lr 2e-5 --warmup_steps 2000\
#     --model_name princeton-nlp/unsup-simcse-roberta-base --dataset M4 --path ${DATA_PATH}/SemEval2024-M4/SubtaskA \
#     --name M4-monolingual-roberta-base --freeze_embedding_layer --database_name monolingual_train --test_dataset_name monolingual_test \
#     --use_signature --wandb_project "signature-m4-mono"

# # M4-multilingual - Single GPU training
python train_classifier.py --device_num 1 --per_gpu_batch_size 32 --total_epoch 50 --lr 2e-5 --warmup_steps 2000\
    --model_name princeton-nlp/unsup-simcse-roberta-base --dataset M4 --path ${DATA_PATH}/SemEval2024-M4/SubtaskA \
    --name M4-multilingual-roberta-base --freeze_embedding_layer --database_name multilingual_train --test_dataset_name multilingual_test \
    --use_signature --wandb_project "signature-m4-multi"

# # OUTFOX - Single GPU training
# python train_classifier.py --device_num 1 --per_gpu_batch_size 32 --total_epoch 50 --lr 2e-5 --warmup_steps 2000\
#     --model_name princeton-nlp/unsup-simcse-roberta-base --dataset OUTFOX --path ${DATA_PATH}/OUTFOX \
#     --name OUTFOX-roberta-base --freeze_embedding_layer --database_name train --test_dataset_name test \
#     --use_signature --wandb_project "signature-outfox"
