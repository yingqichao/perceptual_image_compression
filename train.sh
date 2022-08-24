CUDA_VISIBLE_DEVICES=0 python ./train_compress.py -compression_rate 32 \
                                                  -cls_weight 0.005 \
                                                  -thresh 29 \
                                                  -checkpoint 0