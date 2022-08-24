CUDA_VISIBLE_DEVICES=0 python ./train_compress.py -compression_rate 24 \
                                                  -cls_weight 0.005 \
                                                  -thresh 28 \
                                                  -checkpoint 0