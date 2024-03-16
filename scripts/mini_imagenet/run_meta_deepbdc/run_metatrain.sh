gpuid=0

DATA_ROOT=F:/File/GitHub/miniImageNet
MODEL_PATH=./checkpoints/mini_imagenet/ResNet12_meta_deepbdc_pretrain/best_model.tar
cd ../../../

echo "============= meta-train 1-shot ============="
python meta_train.py --dataset mini_imagenet --data_path $DATA_ROOT --model ResNet12 --method meta_deepbdc --image_size 84 --gpu ${gpuid} --lr 1e-4 --epoch 100 --milestones 40 80 --n_shot 1 --train_n_episode 1000 --val_n_episode 600 --reduce_dim 640 --pretrain_path $MODEL_PATH

echo "============= meta-train 5-shot ============="
python meta_train.py --dataset mini_imagenet --data_path $DATA_ROOT --model ResNet12 --method meta_deepbdc --image_size 84 --gpu ${gpuid} --lr 1e-4 --epoch 100 --milestones 40 80 --n_shot 5 --train_n_episode 600 --val_n_episode 600 --reduce_dim 640 --pretrain_path $MODEL_PATH