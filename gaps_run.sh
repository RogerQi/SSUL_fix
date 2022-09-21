DATA_ROOT=/data/VOCdevkit/VOC2012
DATASET=voc
TASK=15-1 # [15-1, 10-1, 19-1, 15-5, 5-3, 5-1, 2-1, 2-2]
EPOCH=50
BATCH=16
LOSS=bce_loss
LR=0.01
THRESH=0.7
MEMORY=300 # [0 (for SSUL), 100 (for SSUL-M)]

python3 gaps_main.py --crop_val --overlap --data_root ${DATA_ROOT} --model deeplabv3_resnet101 --gpu_id 0 --lr ${LR} --batch_size ${BATCH} --train_epoch ${EPOCH} --loss_type ${LOSS} --dataset ${DATASET} --task ${TASK} --lr_policy poly --pseudo --pseudo_thresh ${THRESH} --freeze --bn_freeze --unknown --w_transfer --amp --mem_size ${MEMORY}
