"""BERT finetuning runner."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import logging
import glob
import json
import argparse
import math
from tqdm import tqdm, trange
from tensorboardX import SummaryWriter
import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
import random
import pickle
from collections import Counter

from pytorch_pretrained_bert.tokenization import BertTokenizer, WhitespaceTokenizer
from pytorch_pretrained_bert.modeling import BertForSeq2SeqDecoder, BertForSentenceRanker
from pytorch_pretrained_bert.optimization import BertAdam, warmup_linear

from nn.data_parallel import DataParallelImbalance

import biunilm.seq2seq_loader as seq2seq_loader
from loader_utils import batch_list_to_batch_tensors
from utils_concat import EvalDataset, ScoreEvalDataset, Preprocess4Seq2cls, Preprocess4SegSepDecoder, EvalRankDataset

from nltk.stem import PorterStemmer
from nltk import word_tokenize, sent_tokenize

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

SPECIAL_TOKEN = ["[UNK]", "[PAD]", "[CLS]", "[MASK]"]
stemmer = PorterStemmer()

def IsMatch(keyphrase1, keyphrase2):
    keyphrase1_stemmed = [stemmer.stem(x) for x in word_tokenize(keyphrase1)]
    keyphrase2_stemmed = [stemmer.stem(x) for x in word_tokenize(keyphrase2)]
    if len(keyphrase1_stemmed) != len(keyphrase2_stemmed):
        return False
    for k1,k2 in zip(keyphrase1_stemmed,keyphrase2_stemmed):
        if k1 != k2:
            return False
    return True

def acc_score(predict, labels):
    hit_cnt = 0
    for kk in predict:
        for jj in labels:
            if IsMatch(kk, jj):
                hit_cnt += 1
                break

    return hit_cnt*1.0000/len(predict)

def recall_score(predict, labels):
    hit_cnt = 0
    for jj in labels:
        for kk in predict:
            if IsMatch(kk, jj):
                hit_cnt += 1
                break
    return hit_cnt*1.0000/len(labels)

def f1_score(acc_score, recall_score):
    if acc_score == 0 or recall_score == 0:
        return 0
    return 2.0000*(acc_score * recall_score)/(acc_score + recall_score)

def detokenize(tk_list):
    r_list = []
    for tk in tk_list:
        if tk.startswith('##') and len(r_list) > 0:
            r_list[-1] = r_list[-1] + tk[2:]
        else:
            r_list.append(tk)
    return r_list


def ascii_print(text):
    text = text.encode("ascii", "ignore")
    print(text)


def main():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument("--bert_model", default=None, type=str, required=True,
                        help="Bert pre-trained model selected in the list: bert-base-uncased, "
                             "bert-large-uncased, bert-base-cased, bert-base-multilingual, bert-base-chinese.")
    parser.add_argument("--model_recover_path", default=None, type=str,
                        help="The file of fine-tuned pretraining model.")
    parser.add_argument("--max_seq_length", default=512, type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")

    # decoding parameters
    parser.add_argument('--fp16', action='store_true',
                        help="Whether to use 16-bit float precision instead of 32-bit")
    parser.add_argument('--amp', action='store_true',
                        help="Whether to use amp for fp16")
    parser.add_argument("--input_file", type=str, help="Input file")
    parser.add_argument("--output_file", type=str, help="output file")
    parser.add_argument("--split", type=str, default="",
                        help="Data split (train/val/test).")
    parser.add_argument('--tokenized_input', action='store_true',
                        help="Whether the input is tokenized.")
    parser.add_argument('--seed', type=int, default=123,
                        help="random seed for initialization")
    parser.add_argument("--do_lower_case", action='store_true',
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument('--new_segment_ids', action='store_true',
                        help="Use new segment ids for bi-uni-directional LM.")
    parser.add_argument('--batch_size', type=int, default=4,
                        help="Batch size for decoding.")
    parser.add_argument('--beam_size', type=int, default=1,
                        help="Beam size for searching")
    parser.add_argument('--top_k', type=int, default=1,
                        help="Top k for output")
    parser.add_argument('--top_kk', type=int, default=0,
                        help="Top k sample method for output")
    parser.add_argument('--length_penalty', type=float, default=0,
                        help="Length penalty for beam search")

    parser.add_argument('--forbid_duplicate_ngrams', action='store_true')
    parser.add_argument('--forbid_ignore_word', type=str, default=None,
                        help="Forbid the word during forbid_duplicate_ngrams")
    parser.add_argument("--min_len", default=None, type=int)
    parser.add_argument('--need_score_traces', action='store_true')
    parser.add_argument('--ngram_size', type=int, default=3)
    parser.add_argument('--mode', default="s2s",
                        choices=["s2s", "l2r", "both"])
    parser.add_argument('--max_tgt_length', type=int, default=128,
                        help="maximum length of target sequence")
    
    # evaluate parameters
    parser.add_argument('--do_predict', action='store_true', help="do_predict")
    parser.add_argument("--do_evaluate", action="store_true", help="caculate the scores if have label file")
    parser.add_argument("--label_file", type=str, default="")
    parser.add_argument("--experiment", type=str, default="full", help="full/title/title-l1/hierachical/title-first/title-first-rouge")

    # ranker parameters
    parser.add_argument("--ranker_recover_path", type=str, help="ranker model for extract sentence")
    parser.add_argument("--ranker_max_len", type=int, default=192, help ="max length of the ranker input")
    parser.add_argument("--ranker_batch_size", type=int, default=128)

    args = parser.parse_args()

    if args.need_score_traces and args.beam_size <= 1:
        raise ValueError(
            "Score trace is only available for beam search with beam size > 1.")
    if args.max_tgt_length >= args.max_seq_length - 2:
        raise ValueError("Maximum tgt length exceeds max seq length - 2.")

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu")
    n_gpu = torch.cuda.device_count()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    tokenizer = BertTokenizer.from_pretrained(
        args.bert_model, do_lower_case=args.do_lower_case)

    tokenizer.max_len = args.max_seq_length

    pair_num_relation = 0
    bi_uni_pipeline = []
    if args.mode == "s2s" or args.mode == "both":
        bi_uni_pipeline.append(seq2seq_loader.Preprocess4Seq2seqDecoder(list(
            tokenizer.vocab.keys()), tokenizer.convert_tokens_to_ids, args.max_seq_length, max_tgt_length=args.max_tgt_length, new_segment_ids=args.new_segment_ids, mode="s2s"))
    if args.mode == "l2r" or args.mode == "both":
        bi_uni_pipeline.append(seq2seq_loader.Preprocess4Seq2seqDecoder(list(
            tokenizer.vocab.keys()), tokenizer.convert_tokens_to_ids, args.max_seq_length, max_tgt_length=args.max_tgt_length, new_segment_ids=args.new_segment_ids, mode="l2r"))
    
    if args.experiment == "segsep":
        bi_uni_pipeline = []
        bi_uni_pipeline.append(Preprocess4SegSepDecoder(list(
            tokenizer.vocab.keys()), tokenizer.convert_tokens_to_ids, args.max_seq_length, max_tgt_length=args.max_tgt_length, new_segment_ids=args.new_segment_ids, mode="s2s"))

    amp_handle = None
    if args.fp16 and args.amp:
        from apex import amp
        amp_handle = amp.init(enable_caching=True)
        logger.info("enable fp16 with amp")

    # Prepare model
    cls_num_labels = 2
    type_vocab_size = 6 if args.new_segment_ids else 2
    if args.experiment == "segsep":
        type_vocab_size = 11
    mask_word_id, eos_word_ids = tokenizer.convert_tokens_to_ids(
        ["[MASK]", "[SEP]"])
    forbid_ignore_set = None
    if args.forbid_ignore_word:
        w_list = []
        for w in args.forbid_ignore_word.split('|'):
            if w.startswith('[') and w.endswith(']'):
                w_list.append(w.upper())
            else:
                w_list.append(w)
        forbid_ignore_set = set(tokenizer.convert_tokens_to_ids(w_list))
    print(args.model_recover_path)
    if args.do_predict:
        for model_recover_path in glob.glob(args.model_recover_path.strip()):
            logger.info("***** Recover model: %s *****", model_recover_path)
            model_recover = torch.load(model_recover_path)
            model = BertForSeq2SeqDecoder.from_pretrained(args.bert_model, state_dict=model_recover, num_labels=cls_num_labels, num_rel=pair_num_relation, type_vocab_size=type_vocab_size, task_idx=3, mask_word_id=mask_word_id, search_beam_size=args.beam_size,
                                                        length_penalty=args.length_penalty, eos_id=eos_word_ids, forbid_duplicate_ngrams=args.forbid_duplicate_ngrams, forbid_ignore_set=forbid_ignore_set, ngram_size=args.ngram_size, min_len=args.min_len, mode=args.mode, max_position_embeddings=args.max_seq_length)
            del model_recover

            if args.fp16:
                model.half()
            model.to(device)
            if n_gpu > 1:
                model = torch.nn.DataParallel(model)

            torch.cuda.empty_cache()
            model.eval()
            next_i = 0
            max_src_length = args.max_seq_length - 2 - args.max_tgt_length

            if args.experiment in ["full", "title", "title-l1"]:
                input_lines = EvalDataset(args.input_file, args.experiment).proc()
            elif args.experiment == "single":
                input_lines, map_dict = EvalDataset(args.input_file, args.experiment).proc()
            elif args.experiment == "title-first":
                input_lines = EvalDataset(args.input_file, args.experiment, tokenizer, args.max_seq_length, args.max_seq_length).proc()
            elif args.experiment == "segsep":
                input_lines = EvalDataset(args.input_file, args.experiment, tokenizer, args.max_seq_length, args.max_seq_length).proc()
            elif args.experiment == "heirachical":
                logger.info("***** Recover rank model: %s *****", args.ranker_recover_path)
                # extract sentences before load data
                # load rank model
                rank_model_recover = torch.load(args.ranker_recover_path, map_location="cpu")
                global_step = 0
                rank_model = BertForSentenceRanker.from_pretrained(args.bert_model, state_dict=rank_model_recover, num_labels=2)
                
                # set model for multi GPUs or multi nodes
                if args.fp16:
                    rank_model.half()
                
                rank_model.to(device)

                if n_gpu > 1:
                    rank_model = DataParallelImbalance(rank_model)
                
                DatasetFunc = ScoreEvalDataset
                
                # Load title + sentence pair
                print ("Loading Rank Dataset from ", args.input_file)
                data_tokenizer = WhitespaceTokenizer() if args.tokenized_input else tokenizer
                max_pred = 16
                mask_prob = 0.7
                rank_bi_uni_pipeline = [Preprocess4Seq2cls(max_pred, mask_prob, list(tokenizer.vocab.keys()), tokenizer.convert_tokens_to_ids, args.ranker_max_len, new_segment_ids=args.new_segment_ids, truncate_config={'max_len_a': 64, 'max_len_b': 16, 'trunc_seg': 'a', 'always_truncate_tail': True}, mask_source_words=False, skipgram_prb=0.0, skipgram_size=1, mask_whole_word=False, mode="s2s", has_oracle=False, num_qkv=0, s2s_special_token=False, s2s_add_segment=False, s2s_share_segment=False, pos_shift=False, eval=True)]
                fn_src = args.input_file
                fn_tgt = None
                eval_dataset = DatasetFunc(
                     fn_src, fn_tgt, args.ranker_batch_size, data_tokenizer, args.ranker_max_len, bi_uni_pipeline=rank_bi_uni_pipeline
                )

                eval_sampler = SequentialSampler(eval_dataset)
                _batch_size = args.ranker_batch_size

                eval_dataloader = torch.utils.data.DataLoader(eval_dataset, batch_size=_batch_size, sampler=eval_sampler, num_workers=24, collate_fn=seq2seq_loader.batch_list_to_batch_tensors, pin_memory=False)


                logger.info("***** CUDA.empty_cache() *****")
                torch.cuda.empty_cache()
                logger.info("***** Runinning ranker *****")
                logger.info("   Batch size = %d", _batch_size)
                logger.info("   Num steps = %d", int(len(eval_dataset)/ args.ranker_batch_size))

                rank_model.to(device)
                rank_model.eval()

                iter_bar = tqdm(eval_dataloader, desc = "Iter: ")
                num_rank_labels = 2
                all_labels = []
                for step, batch in enumerate(iter_bar):
                    batch = [t.to(device) if t is not None else None for t in batch]
                    input_ids, segment_ids, input_mask, mask_qkv, lm_label_ids, masked_pos, masked_weights, is_next, task_idx = batch
                    logits = rank_model(input_ids, task_idx=task_idx, mask_qkv=mask_qkv)
                    labels = torch.argmax(logits.view(-1, num_rank_labels), dim=-1)
                    all_labels.append(labels)
                
                all_labels_results = []
                for label in all_labels:
                    all_labels_results.extend(label.detach().cpu().numpy())
                
                # collect results
                logger.info("**** Collect results ******")
                clu2doc_dict, doc2sent_dict, all_titles, all_sents = eval_dataset.get_maps()
                all_docs = []
                for i, doc in enumerate(doc2sent_dict):
                    text = all_titles[i]
                    sent_idx = doc2sent_dict[doc]
                    for idx in sent_idx:
                        if all_labels_results[idx] == 1:
                            text += ". " + all_sents[idx]
                    all_docs.append(text)
                
                input_lines = []
                for clu in tqdm(clu2doc_dict):
                    doc_idx = clu2doc_dict[clu]
                    input_line  = ""
                    for idx in doc_idx:
                        input_line += all_docs[idx]
                    input_lines.append(input_line)

            elif args.experiment == "title-first-rank":
                logger.info("***** Recover rank model: %s *****", args.ranker_recover_path)
                # extract sentences before load data
                # load rank model
                rank_model_recover = torch.load(args.ranker_recover_path, map_location="cpu")
                global_step = 0
                rank_model = BertForSentenceRanker.from_pretrained(args.bert_model, state_dict=rank_model_recover, num_labels=2)
                
                # set model for multi GPUs or multi nodes
                if args.fp16:
                    rank_model.half()
                
                rank_model.to(device)

                if n_gpu > 1:
                    rank_model = DataParallelImbalance(rank_model)
                
                DatasetFunc = EvalRankDataset
                
                # Load title + sentence pair
                print ("Loading Rank Dataset from ", args.input_file)
                data_tokenizer = WhitespaceTokenizer() if args.tokenized_input else tokenizer
                max_pred = 16
                mask_prob = 0.7
                rank_bi_uni_pipeline = [Preprocess4Seq2cls(max_pred, mask_prob, list(tokenizer.vocab.keys()), tokenizer.convert_tokens_to_ids, args.max_seq_length, new_segment_ids=args.new_segment_ids, truncate_config={'max_len_a': 512, 'max_len_b': 16, 'trunc_seg': 'a', 'always_truncate_tail': True}, mask_source_words=False, skipgram_prb=0.0, skipgram_size=1, mask_whole_word=False, mode="s2s", has_oracle=False, num_qkv=0, s2s_special_token=False, s2s_add_segment=False, s2s_share_segment=False, pos_shift=False, eval=True)]
                fn_src = args.input_file
                fn_tgt = None
                eval_dataset = DatasetFunc(
                     fn_src, fn_tgt, args.ranker_batch_size, data_tokenizer, args.max_seq_length, bi_uni_pipeline=rank_bi_uni_pipeline
                )

                eval_sampler = SequentialSampler(eval_dataset)
                _batch_size = args.ranker_batch_size

                eval_dataloader = torch.utils.data.DataLoader(eval_dataset, batch_size=_batch_size, sampler=eval_sampler, num_workers=24, collate_fn=seq2seq_loader.batch_list_to_batch_tensors, pin_memory=False)

                logger.info("***** CUDA.empty_cache() *****")
                torch.cuda.empty_cache()
                logger.info("***** Runinning ranker *****")
                logger.info("   Batch size = %d", _batch_size)
                logger.info("   Num steps = %d", int(len(eval_dataset)/ args.ranker_batch_size))

                rank_model.to(device)
                rank_model.eval()

                iter_bar = tqdm(eval_dataloader, desc = "Iter: ")
                num_rank_labels = 2
                all_labels = []
                for step, batch in enumerate(iter_bar):
                    batch = [t.to(device) if t is not None else None for t in batch]
                    input_ids, segment_ids, input_mask, mask_qkv, lm_label_ids, masked_pos, masked_weights, is_next, task_idx = batch
                    # print("input_ids", len(input_ids[0]), "segment_ids", len(segment_ids[0]))
                    with torch.no_grad():
                        logits = rank_model(input_ids, task_idx=task_idx, mask_qkv=mask_qkv)
                    labels = logits.view(-1)
                    all_labels.append(labels)

                
                all_labels_results = []
                for label in all_labels:
                    all_labels_results.extend(label.detach().cpu().numpy())
                
                print("test label results")
                print(all_labels_results[0])
                # collect results
                logger.info("**** Collect results ******")
                clu2sent_dict, all_sents, all_titles= eval_dataset.get_maps()
                all_clusters = []
                input_lines = []
                for i, clu_id in enumerate(clu2sent_dict):
                    text = all_titles[clu_id]
                    sent_idx = clu2sent_dict[clu_id]
                    sents_collect = []
                    for idx in sent_idx:
                        sents_collect.append([all_sents[idx], all_labels_results[idx]])
                    sents_collect_sort = sorted(sents_collect, key=lambda x:x[1])

                    sents_collect = [x[0] for x in sents_collect_sort]

                    text_tk = tokenizer.tokenize(text)
                    j = 0
                    while j < len(sents_collect) and len(text_tk) + len(tokenizer.tokenize(sents_collect[j])) <= args.max_seq_length:
                        text += " " + sents_collect[j]
                        j += 1
                    
                    input_lines.append(text)                
            else:
                input_lines = []
            
            data_tokenizer = WhitespaceTokenizer() if args.tokenized_input else tokenizer
            input_lines = [data_tokenizer.tokenize(
                x)[:max_src_length] for x in input_lines]
            input_lines = sorted(list(enumerate(input_lines)),
                                key=lambda x: -len(x[1]))
            output_lines = [""] * len(input_lines)
            score_trace_list = [None] * len(input_lines)
            total_batch = math.ceil(len(input_lines) / args.batch_size)

            with tqdm(total=total_batch) as pbar:
                while next_i < len(input_lines):
                    _chunk = input_lines[next_i:next_i + args.batch_size]
                    buf_id = [x[0] for x in _chunk]
                    buf = [x[1] for x in _chunk]
                    next_i += args.batch_size
                    max_a_len = max([len(x) for x in buf])
                    instances = []
                    for instance in [(x, max_a_len) for x in buf]:
                        for proc in bi_uni_pipeline:
                            instances.append(proc(instance))
                    with torch.no_grad():
                        batch = seq2seq_loader.batch_list_to_batch_tensors(
                            instances)
                        # print("batch")
                        # print(batch)
                        # print(len(batch))
                        batch = [t.to(device) for t in batch if t is not None]
                        input_ids, token_type_ids, position_ids, input_mask, task_idx = batch
                        traces = model(input_ids, token_type_ids,
                                    position_ids, input_mask, task_idx=task_idx)
                        if args.beam_size > 1:
                            traces = {k: v.tolist() for k, v in traces.items()}
                            output_ids = traces['pred_seq']
                        else:
                            output_ids = traces.tolist()
                        for i in range(len(buf)):
                            scores = traces['scores'][i]
                            wids_list = traces['wids'][i]
                            ptrs = traces['ptrs'][i]
                            eos_id = 102
                            top_k = args.top_k
                            # first we need to find the eos frame where all symbols are eos
                            # any frames after the eos frame are invalid
                            last_frame_id = len(scores) - 1
                            for _i, wids in enumerate(wids_list):
                                if all(wid == eos_id for wid in wids):
                                    last_frame_id = _i
                                    break
                            frame_id = -1
                            pos_in_frame = -1
                            seqs = []
                            for fid in range(last_frame_id + 1):
                                for _i, wid in enumerate(wids_list[fid]):
                                    if wid == eos_id or fid == last_frame_id:
                                        s = scores[fid][_i]

                                        frame_id = fid
                                        pos_in_frame = _i

                                        if frame_id != -1 and s < 0:
                                            seq = [wids_list[frame_id][pos_in_frame]]
                                            for _fid in range(frame_id, 0, -1):
                                                pos_in_frame = ptrs[_fid][pos_in_frame]
                                                seq.append(wids_list[_fid - 1][pos_in_frame])
                                            seq.reverse()
                                            seqs.append([seq, s])
                            seqs = sorted(seqs, key= lambda x:x[1], reverse=True)
                            w_idss = [seq[0] for seq in seqs[:top_k]]
                            output_sequences = []
                            for w_ids in w_idss:
                                output_buf = tokenizer.convert_ids_to_tokens(w_ids)
                                output_tokens = []
                                for t in output_buf:
                                    if t in ("[SEP]", "[PAD]"):
                                        break
                                    output_tokens.append(t)
                                output_sequence = ' '.join(detokenize(output_tokens))
                                output_sequences.append(output_sequence)
                            output_lines[buf_id[i]] = output_sequences
                            if args.need_score_traces:
                                score_trace_list[buf_id[i]] = {
                                    'scores': traces['scores'][i], 'wids': traces['wids'][i], 'ptrs': traces['ptrs'][i]}
                    pbar.update(1)
            # collect instances after split
            results = []
            if args.experiment == "single":
                for clu in map_dict:
                    record = []
                    clu_ixs = map_dict[clu]
                    for i in clu_ixs:
                        record.extend(output_lines[i])
                    record_top10 = Counter(record).most_common(10)
                    record_top10 = [x[0] for x in record_top10]
                    results.append(record_top10)

                output_lines = results

            if args.output_file:
                fn_out = args.output_file
            else:
                fn_out = model_recover_path+'.'+args.split
            with open(fn_out, "w", encoding="utf-8") as fout:
                for l in output_lines:
                    fout.write('\t'.join(l))
                    fout.write("\n")

            if args.need_score_traces:
                with open(fn_out + ".trace.pickle", "wb") as fout_trace:
                    pickle.dump(
                        {"version": 0.0, "num_samples": len(input_lines)}, fout_trace)
                    for x in score_trace_list:
                        pickle.dump(x, fout_trace)
        
        # Evaluate !
        if args.do_evaluate:
            labels = []
            if not os.path.exists(args.label_file):
                raise ValueError("Label file not exists")
            print("Loading label file from {}".format(args.label_file))
            with open(args.label_file) as f:
                for line in tqdm(f.readlines()):
                    line = line.strip().split("\t")
                    labels.append(line)
            results = output_lines

            ks = [1, 5, 10]
            results_dict = {}
            for k in ks:
                acc_cul = 0
                r_cul = 0
                f1_cul = 0
                cnt = 0
                for predict, true_label in zip(tqdm(results), tqdm(labels)):
                    predict = predict[:k]
                    true_label = true_label[:k]
                    if len(predict) > 0 and len(true_label) > 0:
                        acc_cul += acc_score(predict, true_label)
                        r_cul += recall_score(predict, true_label)
                        f1_cul += f1_score(acc_score(predict, true_label), recall_score(predict, true_label))
                        cnt += 1
                    
                results_dict["P@{}".format(k)] = acc_cul*1.000 / cnt
                results_dict["R@{}".format(k)] = r_cul*1.000 / cnt
                results_dict["F1@{}".format(k)] = f1_cul*1.000 / cnt
            
            print(results_dict)
if __name__ == "__main__":
    main()
