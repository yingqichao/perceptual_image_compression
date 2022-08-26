CUDA_VISIBLE_DEVICES=1 python ./train_compress.py -compression_rate 72 \
                                                  -cls_weight 0.005 \
                                                  -thresh 27 \
                                                  -batch 32 \
                                                  -checkpoint 0 \
                                                  -epoch_thresh 0 \
                                                  -dataset Caltech \
                                                  -original_scale 96