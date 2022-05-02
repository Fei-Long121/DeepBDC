gpuid=0

DATA_ROOT=/path/cub # path to the json file of CUB
MODEL_PATH=./checkpoints/cub/ResNet18_good_embed_distill_born11/last_model.tar
cd ../../../

echo "============= meta-test 1-shot ============="
python test.py --dataset cub --data_path $DATA_ROOT --model ResNet18 --method good_embed --image_size 224 --gpu ${gpuid} --n_shot 1 --model_path $MODEL_PATH --test_task_nums 5 --penalty_C 0.1


echo "============= meta-test 5-shot ============="
python test.py --dataset cub --data_path $DATA_ROOT --model ResNet18 --method good_embed --image_size 224 --gpu ${gpuid} --n_shot 5 --model_path $MODEL_PATH --test_task_nums 5 --penalty_C 2