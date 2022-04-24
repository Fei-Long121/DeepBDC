gpuid=0

DATA_ROOT=/path/mini_imagenet
MODEL_PATH=./checkpoints/mini_imagenet/ResNet12_good_embed_pretrain/last_model.tar
cd ../../../

 echo "============= distill born 1 ============="
 python distillation.py --dataset mini_imagenet --data_path $DATA_ROOT --model ResNet12 --method good_embed --image_size 84 --gpu ${gpuid} --lr 5e-2 --epoch 170 --milestones 100 150 --save_freq 100 --teacher_path ./checkpoints/mini_imagenet/ResNet12_good_embed_pretrain/last_model.tar --trial 1 --val last

echo "============= distill born 2 ============="
python distillation.py --dataset mini_imagenet --data_path $DATA_ROOT --model ResNet12 --method good_embed --image_size 84 --gpu ${gpuid} --lr 5e-2 --epoch 170 --milestones 100 150 --save_freq 100 --teacher_path ./checkpoints/mini_imagenet/ResNet12_good_embed_distill_born1/last_model.tar --trial 2 --val last