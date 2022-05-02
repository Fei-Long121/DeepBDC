gpuid=0

DATA_ROOT=/path/cub # path to the json file of CUB
MODEL_PATH=./checkpoints/cub/ResNet18_good_embed_pretrain/last_model.tar
cd ../../../

echo "============= distill born 1 ============="
python distillation.py --dataset cub --data_path $DATA_ROOT --model ResNet18 --method good_embed --image_size 224 --gpu ${gpuid} --lr 5e-2 --epoch 220 --milestones 120 170 --save_freq 100  --teacher_path $MODEL_PATH --trial 1 --val last

for i in {2..11}
do
let j=$i-1
echo "=====born $i last born $j ====="
python distillation.py --dataset cub --data_path $DATA_ROOT --model ResNet18 --method good_embed --image_size 224 --gpu ${gpuid} --lr 5e-2 --epoch 220 --milestones 120 170 --save_freq 100 --teacher_path ./checkpoints/cub/ResNet18_good_embed_distill_born$j/last_model.tar --trial $i --val last
done