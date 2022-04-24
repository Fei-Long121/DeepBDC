gpuid=0

DATA_ROOT=/path/mini_imagenet
MODEL_PATH=./checkpoints/mini_imagenet/ResNet12_stl_deepbdc_pretrain/last_model.tar
cd ../../../

echo "============= distill born 1 ============="
python distillation.py --dataset mini_imagenet --data_path $DATA_ROOT --model ResNet12 --method stl_deepbdc --image_size 84 --gpu ${gpuid} --lr 5e-2 --t_lr 1e-3 --epoch 170 --milestones 100 150 --save_freq 100 --teacher_path ./checkpoints/mini_imagenet/ResNet12_stl_deepbdc_pretrain/last_model.tar --trial 1 --reduce_dim 128 --dropout_rate 0.5 --val last

echo "============= distill born 2 ============="
python distillation.py --dataset mini_imagenet --data_path $DATA_ROOT --model ResNet12 --method stl_deepbdc --image_size 84 --gpu ${gpuid} --lr 5e-2 --t_lr 1e-3 --epoch 170 --milestones 100 150 --save_freq 100 --teacher_path ./checkpoints/mini_imagenet/ResNet12_stl_deepbdc_distill_born1/last_model.tar --trial 2 --reduce_dim 128 --dropout_rate 0.5 --val last

echo "============= distill born 3 ============="
python distillation.py --dataset mini_imagenet --data_path $DATA_ROOT --model ResNet12 --method stl_deepbdc --image_size 84 --gpu ${gpuid} --lr 5e-2 --t_lr 1e-3 --epoch 170 --milestones 100 150 --save_freq 100 --teacher_path ./checkpoints/mini_imagenet/ResNet12_stl_deepbdc_distill_born2/last_model.tar --trial 3 --reduce_dim 128 --dropout_rate 0.5 --val last

# echo "============= distill born 4 ============="
# python distillation.py --dataset mini_imagenet --data_path $DATA_ROOT --model ResNet12 --method stl_deepbdc --image_size 84 --gpu ${gpuid} --lr 5e-2 --t_lr 1e-3 --epoch 170 --milestones 100 150 --save_freq 100 --teacher_path ./checkpoints/mini_imagenet/ResNet12_stl_deepbdc_distill_born3/last_model.tar --trial 4 --reduce_dim 128 --dropout_rate 0.5 --val last

# echo "============= distill born 5 ============="
# python distillation.py --dataset mini_imagenet --data_path $DATA_ROOT --model ResNet12 --method stl_deepbdc --image_size 84 --gpu ${gpuid} --lr 5e-2 --t_lr 1e-3 --epoch 170 --milestones 100 150 --save_freq 100 --teacher_path ./checkpoints/mini_imagenet/ResNet12_stl_deepbdc_distill_born4/last_model.tar --trial 5 --reduce_dim 128 --dropout_rate 0.5 --val last