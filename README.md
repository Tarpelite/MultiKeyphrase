# MultiKeyphrase
Multi Document Keyphrase Generation

## Introduction


## Dataset

We use kp20k dataset of scitifical papers (about 53w) to perform k-means clustering.
Each paper is associated with several keyphrases, so the paper is represeneted by the union vector of its all keyphrases during clustering.

After filtering the over dense clusters and the single instances not involved in any clusters, two versions of our MK dataset has been built.

### MK - 3K

 | # sources | # size | 
 | --------- |  ----- |
 |          2|   1342 |
 |          3|    646 |
 |          4|    399 |
 |          5|    241 |


##  Models

UniLM is used as backbone for these series of experiments


## Scripts

for training
```
#!/bin/bash
DATA_DIR=/data/MK_2w/train
OUTPUT_DIR=/data/MK_title_l1_model
MODEL_RECOVER_PATH=/data/unilmv1-large-cased.bin
export CUDA_VISIBLE_DEVICES=0,1,2,3
python biunilm/run_MK.py --do_train --fp16 --amp --num_workers 24 \
  --bert_model bert-large-cased --new_segment_ids \
  --data_dir ${DATA_DIR} --src_file train.src.18k --tgt_file train.tgt.18k \
  --output_dir ${OUTPUT_DIR}/bert_save \
  --log_dir ${OUTPUT_DIR}/bert_log \
  --model_recover_path ${MODEL_RECOVER_PATH} \
  --max_seq_length 192 --max_position_embeddings 192 \
  --trunc_seg a --always_truncate_tail --max_len_b 16 \
  --mask_prob 0.7 --max_pred 16 \
  --train_batch_size 64 --gradient_accumulation_steps 1 \
  --learning_rate 0.00001 --warmup_proportion 0.1 --label_smoothing 0.1 \
  --num_train_epochs 3 --experiment title-l1

```


for decoding
```
#!/bin/bash
# run decoding
DATA_DIR=/data/MK_2w/test
MODEL_RECOVER_PATH=/MK_title_l1_model/bert_save/model.3.bin
EVAL_SPLIT=test


python biunilm/evaluate_1pair.py --fp16 --amp --bert_model bert-large-cased --new_segment_ids --mode s2s  \
  --input_file $DATA_DIR/test.src.2k --split ${EVAL_SPLIT} --output_file model_title_l1.out\
  --model_recover_path ${MODEL_RECOVER_PATH} \
  --max_seq_length 192 --max_tgt_length 16 --do_predict --do_evaluate --label_file $DATA_DIR/test.tgt.2k --experiment title-l1 \
  --batch_size 4 --beam_size 200 --length_penalty 0 --topk 10\
  --forbid_duplicate_ngrams --forbid_ignore_word "."

```
