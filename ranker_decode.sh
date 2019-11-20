#!/bin/bash
# run decoding
DATA_DIR=/data/MK_2w/test
MODEL_RECOVER_PATH=/data/MK_title_first_model/bert_save/model.3.bin
RANK_MODEL=/data/MK_ranker_title_first_model/bert_save/model.3.bin
EVAL_SPLIT=test
export PYTORCH_PRETRAINED_BERT_CACHE=/{tmp_folder}/bert-cased-pretrained-cache
# run decoding
python biunilm/evaluate_1pair.py --fp16 --amp --bert_model bert-large-cased --new_segment_ids --mode s2s  \
  --input_file $DATA_DIR/test.src.2k --split ${EVAL_SPLIT} --output_file model_rank_single.out\
  --model_recover_path ${MODEL_RECOVER_PATH} --ranker_recover_path ${RANK_MODEL} --ranker_max_len 64 --ranker_batch_size 4 \
  --max_seq_length 512 --max_tgt_length 16 --do_predict --do_evaluate --label_file $DATA_DIR/test.tgt.2k --experiment title-first-rank \
  --batch_size 4 --beam_size 50 --length_penalty 0 --top_k 10\
  --forbid_duplicate_ngrams --forbid_ignore_word "."

