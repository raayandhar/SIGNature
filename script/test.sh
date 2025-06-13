DATA_PATH="/data/edward/SIGNature-data"

Model_PATH="/data/edward/SIGNature/runs/deepfake-roberta-base_v0/model_best.pth"
# Model_PATH="/data/edward/SIGNature/runs/TuringBench-roberta-base_v4/model_best.pth" #PATH for the model ckpt

# deepfake
# python test_knn.py --device_num 1 --batch_size 128 --max_K 5 --model_name princeton-nlp/unsup-simcse-roberta-base \
#                    --mode deepfake --database_path ${DATA_PATH}/Deepfake/cross_domains_cross_models --database_name 'train' \
#                    --test_dataset_path ${DATA_PATH}/Deepfake/cross_domains_cross_models --test_dataset_name 'test'\
#                    --model_path ${Model_PATH} --save_database --save_path database/deepfake

# # TuringBench
# python test_knn.py --device_num 1 --batch_size 128 --max_K 5 --model_name princeton-nlp/unsup-simcse-roberta-base \
#                    --mode Turing --database_path ${DATA_PATH}/TuringBench/AA --database_name 'train' \
#                    --test_dataset_path ${DATA_PATH}/TuringBench/AA --test_dataset_name 'test'\
#                    --model_path ${Model_PATH} --save_database --save_path database/TuringBench

# # M4-monolingual
# python test_knn.py --device_num 1 --batch_size 128 --max_K 5 --model_name princeton-nlp/unsup-simcse-roberta-base \
#                    --mode M4 --database_path ${DATA_PATH}/SemEval2024-M4/SubtaskA --database_name 'monolingual_train' \
#                    --test_dataset_path ${DATA_PATH}/SemEval2024-M4/SubtaskA --test_dataset_name 'monolingual_test'\
#                    --model_path ${Model_PATH} --save_database --save_path database/M4-monolingual

# # M4-multilingual
# python test_knn.py --device_num 1 --batch_size 128 --max_K 5 --model_name princeton-nlp/unsup-simcse-roberta-base \
#                    --mode M4 --database_path ${DATA_PATH}/SemEval2024-M4/SubtaskA --database_name 'multilingual_train' \
#                    --test_dataset_path ${DATA_PATH}/SemEval2024-M4/SubtaskA --test_dataset_name 'multilingual_test'\
#                    --model_path ${Model_PATH} --save_database --save_path database/M4-multilingual

# # OUTFOX,attack:none,outfox,dipper
# python test_knn.py --device_num 8 --batch_size 128 --max_K 51 --model_name princeton-nlp/unsup-simcse-roberta-base \
#                    --mode OUTFOX --attack dipper --database_path ${DATA_PATH}/OUTFOX --database_name 'train' \
#                    --test_dataset_path ${DATA_PATH}/OUTFOX --test_dataset_name 'test'\
#                    --model_path ${Model_PATH} --save_database --save_path database/OUTFOX

# OOD eval
# ==============================================
# python test_knn.py --device_num 1 --batch_size 128 --max_K 5 --model_name princeton-nlp/unsup-simcse-roberta-base \
#                    --mode deepfake --database_path ${DATA_PATH}/Deepfake/cross_domains_cross_models --database_name 'train' \
#                    --test_dataset_path ${DATA_PATH}/Deepfake/unseen_models/unseen_model_GLM130B --test_dataset_name 'test_ood'\
#                    --model_path ${Model_PATH} --save_database --save_path database/deepfake

# python test_knn.py --device_num 1 --batch_size 128 --max_K 5 --model_name princeton-nlp/unsup-simcse-roberta-base \
#                    --mode deepfake --database_path ${DATA_PATH}/Deepfake/cross_domains_cross_models --database_name 'train' \
#                    --test_dataset_path ${DATA_PATH}/Deepfake/cross_domains_cross_models --test_dataset_name 'test_ood_gpt_para'\
#                    --model_path ${Model_PATH} --save_database --save_path database/deepfake

# CS162 dev set
# ============================================== arxiv 
# Deepfake Train, 162 M4 test arxiv chat
# python test_knn.py --device_num 1 --batch_size 128 --max_K 5 --model_name princeton-nlp/unsup-simcse-roberta-base \
#                    --mode 162DeepFake --database_path ${DATA_PATH}/Deepfake/cross_domains_cross_models --database_name 'train' \
#                    --test_dataset_path ${DATA_PATH}/CS162-dev/arxiv --test_dataset_name 'test_chatGPT' \
#                    --model_path ${Model_PATH} --save_database --save_path database/deepfake_arxiv_chat

# Deepfake Train, 162 M4 test arxiv cohere
# python test_knn.py --device_num 1 --batch_size 128 --max_K 5 --model_name princeton-nlp/unsup-simcse-roberta-base \
#                    --mode 162DeepFake --database_path ${DATA_PATH}/Deepfake/cross_domains_cross_models --database_name 'train' \
#                    --test_dataset_path ${DATA_PATH}/CS162-dev/arxiv --test_dataset_name 'test_cohere' \
#                    --model_path ${Model_PATH} --save_database --save_path database/deepfake_arxiv_cohere

# ============================================== reddit 
# Deepfake Train, 162 M4 test reddit chat 
# python test_knn.py --device_num 1 --batch_size 128 --max_K 5 --model_name princeton-nlp/unsup-simcse-roberta-base \
#                    --mode 162DeepFake --database_path ${DATA_PATH}/Deepfake/cross_domains_cross_models --database_name 'train' \
#                    --test_dataset_path ${DATA_PATH}/CS162-dev/reddit --test_dataset_name 'test_chatGPT' \
#                    --model_path ${Model_PATH} --save_database --save_path database/deepfake_reddit_chat

# Deepfake Train, 162 M4 test reddit cohere
# python test_knn.py --device_num 1 --batch_size 128 --max_K 5 --model_name princeton-nlp/unsup-simcse-roberta-base \
#                    --mode 162DeepFake --database_path ${DATA_PATH}/Deepfake/cross_domains_cross_models --database_name 'train' \
#                    --test_dataset_path ${DATA_PATH}/CS162-dev/reddit --test_dataset_name 'test_cohere' \
#                    --model_path ${Model_PATH} --save_database --save_path database/deepfake_reddit_cohere

# CS162 ethics dev set
# ============================================== ethics 
# Deepfake Train, 162 ethics test german 
# python test_knn.py --device_num 1 --batch_size 128 --max_K 5 --model_name princeton-nlp/unsup-simcse-roberta-base \
#                    --mode 162DeepFake --database_path ${DATA_PATH}/Deepfake/cross_domains_cross_models --database_name 'train' \
#                    --test_dataset_path ${DATA_PATH}/CS162-dev/german --test_dataset_name 'test_wikipedia' \
#                    --model_path ${Model_PATH} --save_database --save_path database/deepfake_german

# Deepfake Train, 162 ethics test toefl 
python test_knn.py --device_num 1 --batch_size 128 --max_K 5 --model_name princeton-nlp/unsup-simcse-roberta-base \
                   --mode 162DeepFake --database_path ${DATA_PATH}/Deepfake/cross_domains_cross_models --database_name 'train' \
                   --test_dataset_path ${DATA_PATH}/CS162-dev/toefl --test_dataset_name 'test' \
                   --model_path ${Model_PATH} --save_database --save_path database/deepfake_toefl

# Deepfake Train, 162 ethics test hewlett 
# python test_knn.py --device_num 1 --batch_size 128 --max_K 5 --model_name princeton-nlp/unsup-simcse-roberta-base \
#                    --mode 162DeepFake --database_path ${DATA_PATH}/Deepfake/cross_domains_cross_models --database_name 'train' \
#                    --test_dataset_path ${DATA_PATH}/CS162-dev/hewlett --test_dataset_name 'test_hewlett' \
#                    --model_path ${Model_PATH} --save_database --save_path database/deepfake_hewlett
