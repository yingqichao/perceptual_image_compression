CUDA_VISIBLE_DEVICES=1 python ./train_compress.py -compression_rate 24 \
                                                  -cls_weight 0.005 \
                                                  -thresh 28 \
                                                  -batch 64 \
                                                  -checkpoint 0