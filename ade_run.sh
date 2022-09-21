DATA_ROOT=/root/autodl-tmp/ADEChallengeData2016
DATASET=ade
TASK=100-5 # [100-5, 100-10, 100-50, 50-50]
EPOCH=100
BATCH=24
LOSS=bce_loss
LR=0.05
THRESH=0.7
MEMORY=300 # [0 (for SSUL), 300 (for SSUL-M)]

python3 main.py --data_root ${DATA_ROOT} --model deeplabv3_resnet101 --gpu_id 0 --crop_val --lr ${LR} --batch_size ${BATCH} --train_epoch ${EPOCH} --loss_type ${LOSS} --dataset ${DATASET} --task ${TASK} --overlap --lr_policy warm_poly --pseudo --pseudo_thresh ${THRESH} --freeze --bn_freeze --unknown --w_transfer --amp --mem_size ${MEMORY}
