# trained model can be found in $PYOTP_EXP/lrr_saves

python $PYOTP_PATH/libs/LRR/tools/train_rsn.py \
    -c $PYOTP_PATH/configs/LRR/train/lrr.yml \
    -s $STIR_saves/epoch-best.pth \
    -n lrr \
    --trial 0 \
    --rsn-type cnn \
    --layers 256,32
