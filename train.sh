CUDA_VISIBLE_DEVICES=0 python ./train_compress.py -compression_rate 64 \
                                                  -cls_weight 0.005 \
                                                  -thresh 29 \
                                                  -batch 12 \
                                                  -checkpoint 0 \
                                                  -epoch_thresh 3 \
                                                  -dataset Caltech \
                                                  -original_scale 128