gpuid=0

DATA_ROOT=/path/tiered_imagenet
cd ../../../

echo "============= pre-train ============="
python pretrain.py --dataset tiered_imagenet --data_path $DATA_ROOT --model ResNet12 --method protonet --image_size 84 --gpu ${gpuid} --lr 5e-2 --wd 1e-4 --epoch 100 --milestones 40 70 --save_freq 50 --val meta --val_n_episode 600 --n_shot 5