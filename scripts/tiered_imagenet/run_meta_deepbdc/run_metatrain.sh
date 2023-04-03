gpuid=0

DATA_ROOT=/path/tiered_imagenet
MODEL_PATH=./checkpoints/tiered_imagenet/ResNet12_meta_deepbdc_pretrain/best_model.tar
cd ../../../

echo "============= meta-train 1-shot ============="
python meta_train.py --dataset tiered_imagenet --data_path $DATA_ROOT --model ResNet12 --method meta_deepbdc --image_size 84 --gpu ${gpuid} --lr 5e-5 --epoch 100 --milestones 70 --n_shot 1 --train_n_episode 1000 --val_n_episode 1000 --reduce_dim 256 --pretrain_path $MODEL_PATH

echo "============= meta-train 5-shot ============="
python meta_train.py --dataset tiered_imagenet --data_path $DATA_ROOT --model ResNet12 --method meta_deepbdc --image_size 84 --gpu ${gpuid} --lr 5e-5 --epoch 100 --milestones 70 --n_shot 5 --train_n_episode 1000 --val_n_episode 1000 --reduce_dim 256 --pretrain_path $MODEL_PATH

