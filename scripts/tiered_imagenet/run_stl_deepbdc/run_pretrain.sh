gpuid=0

DATA_ROOT=/path/tiered_imagenet
cd ../../../

echo "============= pre-train ============="
python pretrain.py --dataset tiered_imagenet --data_path $DATA_ROOT --model ResNet12 --method stl_deepbdc --image_size 84 --gpu ${gpuid} --lr 5e-2 --t_lr 1e-3 --epoch 90 --milestones 40 70 --save_freq 100 --reduce_dim 128 --dropout_rate 0.5 --val last