gpuid=0

DATA_ROOT=/path/cub # path to the json file of CUB
MODEL_PATH=./checkpoints/cub/ResNet18_protonet_pretrain/best_model.tar
cd ../../../


echo "============= meta-train 1-shot ============="
python meta_train.py --dataset cub --data_path $DATA_ROOT --model ResNet18 --method protonet --image_size 224 --gpu ${gpuid} --lr 1e-3 --epoch 100 --milestones 40 80 --n_shot 1 --train_n_episode 600 --val_n_episode 300 --pretrain_path $MODEL_PATH

echo "============= meta-train 5-shot ============="
python meta_train.py --dataset cub --data_path $DATA_ROOT --model ResNet18 --method protonet --image_size 224 --gpu ${gpuid} --lr 1e-3 --epoch 100 --milestones 40 80 --n_shot 5 --train_n_episode 600 --val_n_episode 300 --pretrain_path $MODEL_PATH
