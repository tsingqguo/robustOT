# trained model can be found in $PYOTP_EXP/lrr_saves

python $PYOTP_PATH/libs/LRR/tools/train.py \
    --config $PYOTP_PATH/configs/LRR/train/stir.yml \
    --name stir \
    --trial 0
