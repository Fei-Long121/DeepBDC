gpuid=0

DATA_ROOT=/path/cub # path to the json file of CUB
cd ../../../

echo "============= pre-train ============="
python pretrain.py --dataset cub --data_path $DATA_ROOT --model ResNet18 --method meta_deepbdc --image_size 224 --gpu ${gpuid} --lr 5e-2 --t_lr 5e-2 --epoch 400 --milestones 401 --save_freq 100 --reduce_dim 256 --dropout_rate 0.6 --val last