CUDA_VISIBLE_DEVICES=0 python ./train_compress.py -compression_rate 96 \
                                                  -cls_weight 0.01 \
                                                  -thresh 27 \
                                                  -batch 12 \
                                                  -checkpoint 0 \
                                                  -epoch_thresh 0 \
                                                  -dataset Caltech \
                                                  -original_scale 128