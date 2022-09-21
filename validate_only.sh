DATA_ROOT=/data/VOCdevkit/VOC2012
DATASET=voc
TASK=15-1 # [15-1, 10-1, 19-1, 15-5, 5-3, 5-1, 2-1, 2-2]

python3 validate_only_main.py --crop_val --overlap --data_root ${DATA_ROOT} --model deeplabv3_resnet101 --gpu_id 0 --dataset ${DATASET} --task ${TASK} --amp
