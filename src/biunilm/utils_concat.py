import json
import sys
import copy
import nltk
import torch


from seq2seq_loader import *
from nltk import sent_tokenize, word_tokenize
from nltk.stem import PorterStemmer
import pickle
import numpy as np
from random import shuffle
# from transformers import BertTokenizer

from tqdm import *
import os
from random import randint, shuffle, choice
from random import random as rand
from rouge_score import rouge_scorer
from multiprocessing import cpu_count, Pool

def batch_list_to_batch_tensors(batch):
    batch_tensors = []
    # print("bacth")
    # print(len(batch))  # == 8
    # print(len(batch[0]))# == 10
    print("bacth label raw")
    print(batch[0][-1])
    for x in zip(*batch):
        print("check the x ")
        # print(torch.tensor(x).shape)
        print(x)
        if x[0] is None:
            batch_tensors.append(None)
        elif isinstance(x[0], torch.Tensor):
            batch_tensors.append(torch.stack(x))
        elif isinstance(x[0], list):
            if isinstance(x[0][0], float):
                print("show me the x")
                print(x)
                batch_tensors.append(torch.tensor(x))
            else:
                batch_tensors.append(torch.tensor(x, dtype=torch.long))
        else:
            batch_tensors.append(torch.tensor(x, dtype=torch.long))
    return batch_tensors

class InputExample(object):
    """

    A single training/test example for multi-keyphrase generation 

    based on One2One pattern (1 input document cluster to 1 keyphrase pertime)

    """
    def __init__(self, guid, cluster_text, keyphrase=None):
        self.guid = guid
        self.cluster_text = cluster_text
        self.keyphrase = keyphrase
    
    def __repr__(self):
        return str(self.to_json_string())
    
    def to_dict(self):
        """Serializes this instance to a python dictionary. """
        output = copy.deepcopy(self.__dict__)
        return output
    
    def to_json_string(self):
        """Serialize this instance to a JSON string. """
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"
    

class InputFeatures(object):
    """
    A single set of features of data.
    Args:
        input_ids: Indices of input sequence tokens in the vocabulary.
        attention_mask: Mask to avoid performing attention on padding token indices.
            Mask values selected in ``[0, 1]``:
            Usually  ``1`` for tokens that are NOT MASKED, ``0`` for MASKED (padded) tokens.
        token_type_ids: Segment token indices to indicate first and second portions of the inputs.
        keyphrase: the target keyphrase of the input sequence
    """

    def __init__(self, input_ids, segment_ids, input_mask, mask_qkv, masked_ids, masked_pos, masked_weights, is_next, task_idx):
        self.input_ids = input_ids
        self.segment_ids = segment_ids
        self.input_mask = input_mask
        self.mask_qkv = mask_qkv
        self.masked_ids = masked_ids
        self.masked_pos = masked_pos
        self.masked_weights = masked_weights
        self.is_next = is_next
        self.task_idx = task_idx
    
    def __repr__(self):
        return str(self.to_json_string)
    
    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"

class DataProcessor(object):
    """Base class for data converters for keyphrase generation data sets."""
    def get_example_from_tensor_dict(self, tensor_dict):
        """Gets an example from a dict with tensorflow tensors
        Args:
            tensor_dict: Keys and values should match the corresponding Glue
                tensorflow_dataset examples.
        """
        raise NotImplementedError()

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with open(input_file, "r", encoding="utf-8-sig") as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            lines = []
            for line in reader:
                if sys.version_info[0] == 2:
                    line = list(unicode(cell, 'utf-8') for cell in line)
                lines.append(line)
            return lines


class ConcatDataset(Seq2SeqDataset):
    
    def __init__(self, file_src, file_tgt, batch_size, tokenizer, max_len, short_sampling_prob=0.1, sent_reverse_order=False, bi_uni_pipeline=[]):
        super().__init__(file_src, file_tgt, batch_size, tokenizer, max_len)
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.short_sampling_prob = short_sampling_prob
        self.bi_uni_pipeline = bi_uni_pipeline
        self.batch_size = batch_size
        self.sent_reverse_order = sent_reverse_order

        self.cached = False
        if os.path.exists("cached_dataset.pl") :
            self.cached = True
        
        # if cached, load cache
        if self.cached:
            with open("cached_dataset.pl", "rb") as f:
                self.ex_list = pickle.load(f)
        else:
            # read the file into memory
            self.ex_list = []
            with open(file_src, "r", encoding="utf-8") as f_src, open(file_tgt, "r", encoding="utf-8") as f_tgt:
                for src, tgt in zip(tqdm(f_src.readlines()), tqdm(f_tgt.readlines())):
                    docs = [json.loads(x) for x in src.strip().strip("\n").split("\t")]
                    docs = [x["title"] + " " + x["abstract"] for x in docs]
                    keywords = tgt.strip().strip("\n").split("\t")
                    src_tk = tokenizer.tokenize(" ".join(docs))
                    for kk in keywords:
                        tgt_tk = tokenizer.tokenize(kk)
                        if len(src_tk) > 0 and len(tgt_tk) > 0:
                            self.ex_list.append((src_tk, tgt_tk))
            
            with open("cached_dataset.pl", "wb") as f:
                pickle.dump(self.ex_list, f)

        
        print('Load {0} documents'.format(len(self.ex_list)))
        # caculate statistics
        src_tk_lens = [len(x[0]) for x in self.ex_list]
        tgt_tk_lens = [len(x[1]) for x in self.ex_list]
        print("Statistics:\nsrc_tokens: max:{0}  min:{1}  avg:{2}\ntgt_tokens: max:{3} min:{4} avg:{5}".format(max(src_tk_lens), min(src_tk_lens), sum(src_tk_lens)/len(self.ex_list), max(tgt_tk_lens), min(tgt_tk_lens), sum(tgt_tk_lens)/len(tgt_tk_lens)))
                


class TitleDataset(Seq2SeqDataset):

    def __init__(self, file_src, file_tgt, batch_size, tokenizer, max_len, short_sampling_prob=0.1, sent_reverse_order=False, bi_uni_pipeline=[]):
        super().__init__(file_src, file_tgt, batch_size, tokenizer, max_len)
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.short_sampling_prob = short_sampling_prob
        self.bi_uni_pipeline = bi_uni_pipeline
        self.batch_size = batch_size
        self.sent_reverse_order = sent_reverse_order

        self.cached = False
        if os.path.exists("cached_dataset.pl") :
            self.cached = True
        
        # if cached, load cache
        if self.cached:
            with open("cached_dataset.pl", "rb") as f:
                self.ex_list = pickle.load(f)
        else:
            # read the file into memory
            self.ex_list = []
            with open(file_src, "r", encoding="utf-8") as f_src, open(file_tgt, "r", encoding="utf-8") as f_tgt:
                for src, tgt in zip(tqdm(f_src.readlines()), tqdm(f_tgt.readlines())):
                    docs = [json.loads(x) for x in src.strip().strip("\n").split("\t")]
                    docs = [x["title"]  for x in docs]
                    keywords = tgt.strip().strip("\n").split("\t")
                    src_tk = tokenizer.tokenize(" ".join(docs))
                    for kk in keywords:
                        tgt_tk = tokenizer.tokenize(kk)
                        if len(src_tk) > 0 and len(tgt_tk) > 0:
                            self.ex_list.append((src_tk, tgt_tk))
            
            with open("cached_dataset.pl", "wb") as f:
                pickle.dump(self.ex_list, f)

        
        print('Load {0} documents'.format(len(self.ex_list)))
        # caculate statistics
        src_tk_lens = [len(x[0]) for x in self.ex_list]
        tgt_tk_lens = [len(x[1]) for x in self.ex_list]
        print("Statistics:\nsrc_tokens: max:{0}  min:{1}  avg:{2}\ntgt_tokens: max:{3} min:{4} avg:{5}".format(max(src_tk_lens), min(src_tk_lens), sum(src_tk_lens)/len(self.ex_list), max(tgt_tk_lens), min(tgt_tk_lens), sum(tgt_tk_lens)/len(tgt_tk_lens)))



class TitleLead1Dataset(Seq2SeqDataset):
    
    def __init__(self, file_src, file_tgt, batch_size, tokenizer, max_len, short_sampling_prob=0.1, sent_reverse_order=False, bi_uni_pipeline=[]):
        super().__init__(file_src, file_tgt, batch_size, tokenizer, max_len)
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.short_sampling_prob = short_sampling_prob
        self.bi_uni_pipeline = bi_uni_pipeline
        self.batch_size = batch_size
        self.sent_reverse_order = sent_reverse_order

        self.cached = False
        if os.path.exists("cached_dataset.pl") :
            self.cached = True
        
        # if cached, load cache
        if self.cached:
            with open("cached_dataset.pl", "rb") as f:
                self.ex_list = pickle.load(f)
        else:
            # read the file into memory
            self.ex_list = []
            with open(file_src, "r", encoding="utf-8") as f_src, open(file_tgt, "r", encoding="utf-8") as f_tgt:
                for src, tgt in zip(tqdm(f_src.readlines()), tqdm(f_tgt.readlines())):
                    docs = [json.loads(x) for x in src.strip().strip("\n").split("\t")]
                    titles = [x["title"]  for x in docs]
                    abstracts = [x["abstract"] for x in docs]
                    abstracts_l1 = [nltk.sent_tokenize(x)[0] for x in abstracts]
                    
                    try:
                        assert len(titles) == len(abstracts_l1)
                    except Exception as e:
                        print("title", len(titles))
                        print("abstrac_l1", len(abstracts_l1))

                    docs = []
                    for title, l1_line in zip(titles, abstracts_l1):
                        docs.append(title)
                        docs.append(l1_line)
                    
                    keywords = tgt.strip().strip("\n").split("\t")
                    src_tk = tokenizer.tokenize(" ".join(docs))
                    for kk in keywords:
                        tgt_tk = tokenizer.tokenize(kk)
                        if len(src_tk) > 0 and len(tgt_tk) > 0:
                            self.ex_list.append((src_tk, tgt_tk))
            
            with open("cached_dataset.pl", "wb") as f:
                pickle.dump(self.ex_list, f)

        
        print('Load {0} documents'.format(len(self.ex_list)))
        # caculate statistics
        src_tk_lens = [len(x[0]) for x in self.ex_list]
        tgt_tk_lens = [len(x[1]) for x in self.ex_list]
        print("Statistics:\nsrc_tokens: max:{0}  min:{1}  avg:{2}\ntgt_tokens: max:{3} min:{4} avg:{5}".format(max(src_tk_lens), min(src_tk_lens), sum(src_tk_lens)/len(self.ex_list), max(tgt_tk_lens), min(tgt_tk_lens), sum(tgt_tk_lens)/len(tgt_tk_lens)))

class TitleFirstDataset(Seq2SeqDataset):

    def __init__(self, file_src, file_tgt, batch_size, tokenizer, max_len, max_len_b, short_sampling_prob=0.1, sent_reverse_order=False, bi_uni_pipeline=[]):
        
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.short_sampling_prob = short_sampling_prob
        self.bi_uni_pipeline = bi_uni_pipeline
        self.batch_size = batch_size
        self.sent_reverse_order = sent_reverse_order
        self.max_len_a = max_len - max_len_b - 3 # leave 3 tokens for [CLS] [SEP] symbol

        self.cached = False
        if os.path.exists("cached_dataset.pl") :
            self.cached = True
        
        # if cached, load cache
        if self.cached:
            with open("cached_dataset.pl", "rb") as f:
                self.ex_list = pickle.load(f)
        else:
            # read the file into memory
            self.ex_list = []
            with open(file_src, "r", encoding="utf-8") as f_src, open(file_tgt, "r", encoding="utf-8") as f_tgt:
                for src, tgt in zip(tqdm(f_src.readlines()), tqdm(f_tgt.readlines())):
                    docs = [json.loads(x) for x in src.strip().strip("\n").split("\t")]
                    titles = [x["title"]  for x in docs]
                    abstracts = [x["abstract"] for x in docs]
                    keywords = tgt.strip().strip("\n").split("\t")
                    sents = []
                    for piece in abstracts:
                        piece_sents = sent_tokenize(piece)
                        sents.extend(piece_sents)
                    

                    src_tk = self.tokenizer.tokenize(" ".join(titles))
                    sent_idx = 0
                    while sent_idx < len(sents) and len(src_tk) + len(self.tokenizer.tokenize(" ".join(sents[sent_idx]))) <= self.max_len_a:
                        src_tk += self.tokenizer.tokenize(" ".join(sents[sent_idx]))
                        sent_idx += 1

                    for kk in keywords:
                        tgt_tk = tokenizer.tokenize(kk)
                        if len(src_tk) > 0 and len(tgt_tk) > 0:
                            self.ex_list.append((src_tk, tgt_tk))
            
            with open("cached_dataset.pl", "wb") as f:
                pickle.dump(self.ex_list, f)

        
        print('Load {0} documents'.format(len(self.ex_list)))
        # caculate statistics
        src_tk_lens = [len(x[0]) for x in self.ex_list]
        tgt_tk_lens = [len(x[1]) for x in self.ex_list]
        print("Statistics:\nsrc_tokens: max:{0}  min:{1}  avg:{2}\ntgt_tokens: max:{3} min:{4} avg:{5}".format(max(src_tk_lens), min(src_tk_lens), sum(src_tk_lens)/len(self.ex_list), max(tgt_tk_lens), min(tgt_tk_lens), sum(tgt_tk_lens)/len(tgt_tk_lens)))

class SingleTrainingDataset(Seq2SeqDataset):
    
    def __init__(self, file_src, file_tgt, batch_size, tokenizer, max_len, short_sampling_prob=0.1, sent_reverse_order=False, bi_uni_pipeline=[]):
        super().__init__(file_src, file_tgt, batch_size, tokenizer, max_len)
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.short_sampling_prob = short_sampling_prob
        self.bi_uni_pipeline = bi_uni_pipeline
        self.batch_size = batch_size
        self.sent_reverse_order = sent_reverse_order

        self.cached = False
        if os.path.exists("cached_dataset.pl") :
            self.cached = True
        
        # if cached, load cache
        if self.cached:
            with open("cached_dataset.pl", "rb") as f:
                self.ex_list = pickle.load(f)
        else:
            # read the file into memory
            self.ex_list = []
            with open(file_src, "r", encoding="utf-8") as f_src, open(file_tgt, "r", encoding="utf-8") as f_tgt:
                for src in tqdm(f_src.readlines()):
                    json_docs = [json.loads(x) for x in src.strip().strip("\n").split("\t")]
                    for json_doc in json_docs:
                        doc = json_doc["title"] + " " + json_doc["abstract"]
                        keywords = json_doc["keyword"].split(";")
                        src_tk = tokenizer.tokenize(doc)
                        for kk in keywords:
                            tgt_tk = tokenizer.tokenize(kk)
                            if len(src_tk) > 0 and len(tgt_tk) > 0:
                                self.ex_list.append((src_tk, tgt_tk))
            
            with open("cached_dataset.pl", "wb") as f:
                pickle.dump(self.ex_list, f)

        
        print('Load {0} documents'.format(len(self.ex_list)))
        # caculate statistics
        src_tk_lens = [len(x[0]) for x in self.ex_list]
        tgt_tk_lens = [len(x[1]) for x in self.ex_list]
        print("Statistics:\nsrc_tokens: max:{0}  min:{1}  avg:{2}\ntgt_tokens: max:{3} min:{4} avg:{5}".format(max(src_tk_lens), min(src_tk_lens), sum(src_tk_lens)/len(self.ex_list), max(tgt_tk_lens), min(tgt_tk_lens), sum(tgt_tk_lens)/len(tgt_tk_lens)))

class EvalDataset(object):
    def __init__(self,input_file, experiment, tokenizer=None, max_seq_length=None, max_tgt_length=None):
        self.input_file = input_file
        self.experiment = experiment
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.max_tgt_length = max_tgt_length
        self.max_len = max_seq_length - max_tgt_length - 3
    
    def load_full(self, x):
        return x["title"] + " " + x["abstract"]
    
    def load_title(self, x):
        return x["title"]
    
    def load_title_l1(self, x):
        return x["title"] + " " + sent_tokenize(x["abstract"])[0]
    
    def proc(self):
        input_lines = []
        if self.experiment in ["full", "title", "title-l1"]:
            if self.experiment == "full":
                load_func = self.load_full
            elif self.experiment == "title":
                load_func = self.load_title
            else:
                load_func = self.load_title_l1
            
            with open(self.input_file) as fin:
                for line in tqdm(fin.readlines()):
                    line = line.strip().split("\t")
                    docs = [load_func(json.loads(doc)) for doc in line]
                    input_lines.append(" ".join(docs))
            return input_lines
        elif self.experiment == "title-first":
            
            with open(self.input_file) as fin:
                input_lines = []
                for line in tqdm(fin.readlines()):
                    line = line.strip().split("\t")
                    docs = [json.loads(doc) for doc in line]
                    titles = [doc["title"] for doc in docs]
                    abstracts = [doc["abstract"] for doc in docs]
                    titles_seq = " ".join(titles)
                    sents = []

                    for piece in abstracts:
                        piece_sents = sent_tokenize(piece)
                        sents.extend(piece_sents)
                    input_line = titles_seq
                    input_line_tk = self.tokenizer.tokenize(titles_seq)
                    sents_idx = 0
                    while sents_idx < len(sents) and len(input_line_tk) + len(self.tokenizer.tokenize(sents[sents_idx])) <= self.max_len :
                        input_line += sents[sents_idx]
                        input_line_tk += self.tokenizer.tokenize(sents[sents_idx])
                        sents_idx += 1
                    input_lines.append(input_line)
                return input_lines
                    
        elif self.experiment == "single":
            clu_cnt = 0
            glo_cnt = 0
            map_dict = {}
            with open(self.input_file) as fin:
                for line in tqdm(fin.readlines()):
                    line = line.strip().split("\t")
                    docs = [self.load_full(json.loads(doc)) for doc in line]
                    for doc in docs:
                        input_lines.append(doc)
                        if clu_cnt not in map_dict:
                            map_dict[clu_cnt] = [glo_cnt]
                        else:
                            map_dict[clu_cnt].append(glo_cnt)
                        glo_cnt += 1
                    clu_cnt += 1
            return input_lines, map_dict       
        return input_lines
    



class ScoreDataset(Seq2SeqDataset):
    
    def __init__(self, file_src, file_tgt, batch_size, tokenizer, max_len, short_sampling_prob=0.1, sent_reverse_order=False, bi_uni_pipeline=[]):
        # super().__init__(file_src, file_tgt, batch_size, tokenizer, max_len)
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.short_sampling_prob = short_sampling_prob
        self.bi_uni_pipeline = bi_uni_pipeline
        self.batch_size = batch_size
        self.sent_reverse_order = sent_reverse_order
        self.cached = False
        self.stemmer = PorterStemmer()

        if os.path.exists("cached_dataset.pl"):
            self.cached = True
        
        if self.cached:
            with open("cached_dataset.pl", "rb") as f:
                self.ex_list = pickle.load(f)
        else:
            self.ex_list = []
            with open(file_src, "r", encoding="utf-8") as f_src, open(file_tgt, "r", encoding="utf-8") as f_tgt:
                for src, tgt in zip(tqdm(f_src.readlines()), tqdm(f_tgt.readlines())):
                    docs = [json.loads(x) for x in src.strip().strip("\n").split("\t")]
                    titles = [x["title"]  for x in docs]
                    abstracts = [x["title"] + ". " + x["abstract"] for x in docs]
                    keywords = tgt.strip().strip("\n").split()
                    keywords_stemmed = [self.stem(x) for x in keywords]

                    for doc in docs:
                        title = doc["title"]
                        abstract = doc["abstract"]
                        sents = sent_tokenize(abstract)
                        for sent in sents:
                            label = 0
                            sent_stemmed = self.stem(sent)
                            for kk in keywords_stemmed:
                                if kk in sent_stemmed:
                                    label = 1
                                    break
                            src_tk = tokenizer.tokenize(title)
                            tgt_tk = tokenizer.tokenize(sent)
                            if len(src_tk) > 0 and len(tgt_tk) > 0:
                                self.ex_list.append((src_tk, tgt_tk, label))
            with open("cached_dataset.pl", "wb") as f:
                pickle.dump(self.ex_list, f)

        
        print('Load {0} documents'.format(len(self.ex_list)))
        # caculate statistics
        src_tk_lens = [len(x[0]) for x in self.ex_list]
        tgt_tk_lens = [len(x[1]) for x in self.ex_list]
        print("Statistics:\nsrc_tokens: max:{0}  min:{1}  avg:{2}\ntgt_tokens: max:{3} min:{4} avg:{5}".format(max(src_tk_lens), min(src_tk_lens), sum(src_tk_lens)/len(self.ex_list), max(tgt_tk_lens), min(tgt_tk_lens), sum(tgt_tk_lens)/len(tgt_tk_lens)))


    def stem(self, x):
        return " ".join([self.stemmer.stem(w) for w in word_tokenize(x)])


class ScoreRougeDataset(Seq2SeqDataset):

    def __init__(self, file_src, file_tgt, batch_size, tokenizer, max_len, short_sampling_prob=0.1, sent_reverse_order=False, bi_uni_pipeline=[]):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.short_sampling_prob = short_sampling_prob
        self.bi_uni_pipeline = bi_uni_pipeline
        self.batch_size = batch_size
        self.sent_reverse_order = sent_reverse_order
        self.cached = False
        self.scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)

        if os.path.exists("cached_rank_dataset.pl"):
            self.cached = True
        
        if self.cached:
            with open("cached_rank_dataset.pl", "rb") as f:
                self.ex_list = pickle.load(f)
        
        else:
            self.ex_list = []
            instances = []
            results = []
            with open(file_src, "r", encoding="utf-8") as f_src, open(file_tgt, "r", encoding="utf-8") as f_tgt:
                for src, tgt in zip(f_src.readlines(), f_tgt.readlines()):
                    instances.append([src, tgt])
            with Pool(cpu_count()) as p:
                results = list(tqdm(p.imap(self.solve, instances), total=len(instances)))
            for res in results:
                if len(res) > 0:
                    self.ex_list.extend(res)     

            with open("cached_rank_dataset.pl", "wb") as f:
                pickle.dump(self.ex_list, f)   
        
        test_ins = self.ex_list
                    
    def get_score(self, sent, keywords):
        scores = 0
        for kk in keywords:
            score_t = self.scorer.score(sent, kk)
            score_t = score_t['rougeL'].fmeasure
            scores += score_t
        scores = scores*1.0000 / 10

        return scores
    
    def solve(self, instance):
        results = []
        src, tgt = instance
        docs = [json.loads(x) for x in src.strip().strip("\n").split("\t")]
        titles = [x["title"] for x in docs]
        abstracts = [x["abstract"] for x in docs]
        keywords = list(tgt.strip().split("\t"))
        sents = []
        for piece in abstracts:
            piece_sents = sent_tokenize(piece)
            sents.extend(piece_sents)
        
        title_concat = " ".join(titles)
        for sent in sents:
            src_seq = title_concat 
            tgt_seq = sent
            label_score = self.get_score(src_seq, tgt_seq)
            src_tk = self.tokenizer.tokenize(src_seq)
            tgt_tk = self.tokenizer.tokenize(tgt_seq)
            if len(src_tk) > 0 and len(tgt_tk) > 0:
                results.append((src_tk, tgt_tk, label_score))
        return results



class ScoreEvalDataset(Seq2SeqDataset):
    def __init__(self, file_src, file_tgt, batch_size, tokenizer, max_len, short_sampling_prob=0.1, sent_reverse_order=False, bi_uni_pipeline=[]):
        # super().__init__(file_src, file_tgt, batch_size, tokenizer, max_len)
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.short_sampling_prob = short_sampling_prob
        self.bi_uni_pipeline = bi_uni_pipeline
        self.batch_size = batch_size
        self.sent_reverse_order = sent_reverse_order
        self.cached = False
        self.stemmer = PorterStemmer()
        self.ex_list = []
        self.clu2doc_dict = {}
        self.doc2sent_dict = {}
        self.all_titles = []
        self.all_sents = []

        if os.path.exists("eval_cached_dataset.pl"):
            self.cached = True
        
        if self.cached:
            with open("eval_cached_dataset.pl", "rb") as f:
                self.ex_list, self.clu2doc_dict, self.doc2sent_dict, self.all_titles, self.all_sents = pickle.load(f)
        else:
            
            clu_cnt = 0
            doc_cnt = 0
            sent_cnt = 0
            with open(file_src, "r", encoding="utf-8") as f_src:
                for src in tqdm(f_src.readlines()):
                    docs = [json.loads(x) for x in src.strip().strip("\n").split("\t")]
                    titles = [x["title"]  for x in docs]
                    abstracts = [x["title"] + ". " + x["abstract"] for x in docs]

                    for doc in docs:
                        title = doc["title"]
                        abstract = doc["abstract"]
                        sents = sent_tokenize(abstract)
                        self.all_titles.append(title)
                        for sent in sents:
                            src_tk = tokenizer.tokenize(title)
                            tgt_tk = tokenizer.tokenize(sent)
                            if len(src_tk) > 0 and len(tgt_tk) > 0:
                                self.ex_list.append((src_tk, tgt_tk))
                                if doc_cnt not in self.doc2sent_dict:
                                    self.doc2sent_dict[doc_cnt] = [sent_cnt]
                                else:
                                    self.doc2sent_dict[doc_cnt].append(sent_cnt)
                                sent_cnt += 1
                            self.all_sents.append(sent)
                        if clu_cnt not in self.clu2doc_dict:
                            self.clu2doc_dict[clu_cnt] = [doc_cnt]
                        else:
                            self.clu2doc_dict[clu_cnt].append(doc_cnt)
                        doc_cnt += 1
                    
                    clu_cnt += 1
                
                    
            with open("eval_cached_dataset.pl", "wb") as f:
                pickle.dump([self.ex_list, self.clu2doc_dict, self.doc2sent_dict, self.all_titles, self.all_sents], f)

        
        print('Load {0} documents'.format(len(self.ex_list)))
        # caculate statistics
        src_tk_lens = [len(x[0]) for x in self.ex_list]
        tgt_tk_lens = [len(x[1]) for x in self.ex_list]
        print("Statistics:\nsrc_tokens: max:{0}  min:{1}  avg:{2}\ntgt_tokens: max:{3} min:{4} avg:{5}".format(max(src_tk_lens), min(src_tk_lens), sum(src_tk_lens)/len(self.ex_list), max(tgt_tk_lens), min(tgt_tk_lens), sum(tgt_tk_lens)/len(tgt_tk_lens)))


    def stem(self, x):
        return " ".join([self.stemmer.stem(w) for w in word_tokenize(x)])
    
    def get_maps(self):
        return self.clu2doc_dict, self.doc2sent_dict, self.all_titles, self.all_sents



class Preprocess4Seq2cls(Preprocess4Seq2seq):

    def __init__(self, max_pred, mask_prob, vocab_words, indexer, max_len=512, skipgram_prb=0, skipgram_size=0, block_mask=False, mask_whole_word=False, new_segment_ids=False, truncate_config={}, mask_source_words=False, mode="s2s", has_oracle=False, num_qkv=0, s2s_special_token=False, s2s_add_segment=False, s2s_share_segment=False, pos_shift=False, eval=False):
        super().__init__(max_pred, mask_prob, vocab_words, indexer, max_len=512, skipgram_prb=0, skipgram_size=0, block_mask=False, mask_whole_word=False, new_segment_ids=False, truncate_config={}, mask_source_words=False, mode="s2s", has_oracle=False, num_qkv=0, s2s_special_token=False, s2s_add_segment=False, s2s_share_segment=False, pos_shift=False)
        self.eval = eval
    def __call__(self, instance):
        if not self.eval:
            tokens_a, tokens_b, label = instance
        else:
            tokens_a, tokens_b = instance
        if self.pos_shift:
            tokens_b = ['[S2S_SOS]'] + tokens_b

        # -3  for special tokens [CLS], [SEP], [SEP]
        num_truncated_a, _ = truncate_tokens_pair(tokens_a, tokens_b, self.max_len - 3, max_len_a=self.max_len_a,
                                                  max_len_b=self.max_len_b, trunc_seg=self.trunc_seg, always_truncate_tail=self.always_truncate_tail)

        # Add Special Tokens
        if self.s2s_special_token:
            tokens = ['[S2S_CLS]'] + tokens_a + \
                ['[S2S_SEP]'] + tokens_b + ['[SEP]']
        else:
            tokens = ['[CLS]'] + tokens_a + ['[SEP]'] + tokens_b + ['[SEP]']

        if self.new_segment_ids:
            if self.mode == "s2s":
                if self.s2s_add_segment:
                    if self.s2s_share_segment:
                        segment_ids = [0] + [1] * \
                            (len(tokens_a)+1) + [5]*(len(tokens_b)+1)
                    else:
                        segment_ids = [4] + [6] * \
                            (len(tokens_a)+1) + [5]*(len(tokens_b)+1)
                else:
                    segment_ids = [4] * (len(tokens_a)+2) + \
                        [5]*(len(tokens_b)+1)
            else:
                segment_ids = [2] * (len(tokens))
        else:
            segment_ids = [0]*(len(tokens_a)+2) + [1]*(len(tokens_b)+1)

        if self.pos_shift:
            n_pred = min(self.max_pred, len(tokens_b))
            masked_pos = [len(tokens_a)+2+i for i in range(len(tokens_b))]
            masked_weights = [1]*n_pred
            masked_ids = self.indexer(tokens_b[1:]+['[SEP]'])
        else:
            # For masked Language Models
            # the number of prediction is sometimes less than max_pred when sequence is short
            effective_length = len(tokens_b)
            if self.mask_source_words:
                effective_length += len(tokens_a)
            n_pred = min(self.max_pred, max(
                1, int(round(effective_length*self.mask_prob))))
            # candidate positions of masked tokens
            cand_pos = []
            special_pos = set()
            for i, tk in enumerate(tokens):
                # only mask tokens_b (target sequence)
                # we will mask [SEP] as an ending symbol
                if (i >= len(tokens_a)+2) and (tk != '[CLS]'):
                    cand_pos.append(i)
                elif self.mask_source_words and (i < len(tokens_a)+2) and (tk != '[CLS]') and (not tk.startswith('[SEP')):
                    cand_pos.append(i)
                else:
                    special_pos.add(i)
            shuffle(cand_pos)

            masked_pos = set()
            max_cand_pos = max(cand_pos)
            for pos in cand_pos:
                if len(masked_pos) >= n_pred:
                    break
                if pos in masked_pos:
                    continue

                def _expand_whole_word(st, end):
                    new_st, new_end = st, end
                    while (new_st >= 0) and tokens[new_st].startswith('##'):
                        new_st -= 1
                    while (new_end < len(tokens)) and tokens[new_end].startswith('##'):
                        new_end += 1
                    return new_st, new_end

                if (self.skipgram_prb > 0) and (self.skipgram_size >= 2) and (rand() < self.skipgram_prb):
                    # ngram
                    cur_skipgram_size = randint(2, self.skipgram_size)
                    if self.mask_whole_word:
                        st_pos, end_pos = _expand_whole_word(
                            pos, pos + cur_skipgram_size)
                    else:
                        st_pos, end_pos = pos, pos + cur_skipgram_size
                else:
                    # directly mask
                    if self.mask_whole_word:
                        st_pos, end_pos = _expand_whole_word(pos, pos + 1)
                    else:
                        st_pos, end_pos = pos, pos + 1

                for mp in range(st_pos, end_pos):
                    if (0 < mp <= max_cand_pos) and (mp not in special_pos):
                        masked_pos.add(mp)
                    else:
                        break

            masked_pos = list(masked_pos)
            if len(masked_pos) > n_pred:
                shuffle(masked_pos)
                masked_pos = masked_pos[:n_pred]

            masked_tokens = [tokens[pos] for pos in masked_pos]
            for pos in masked_pos:
                if rand() < 0.8:  # 80%
                    tokens[pos] = '[MASK]'
                elif rand() < 0.5:  # 10%
                    tokens[pos] = get_random_word(self.vocab_words)
            # when n_pred < max_pred, we only calculate loss within n_pred
            masked_weights = [1]*len(masked_tokens)

            # Token Indexing
            masked_ids = self.indexer(masked_tokens)
        # Token Indexing
        input_ids = self.indexer(tokens)

        # Zero Padding
        n_pad = self.max_len - len(input_ids)
        input_ids.extend([0]*n_pad)
        segment_ids.extend([0]*n_pad)

        if self.num_qkv > 1:
            mask_qkv = [0]*(len(tokens_a)+2) + [1] * (len(tokens_b)+1)
            mask_qkv.extend([0]*n_pad)
        else:
            mask_qkv = None

        input_mask = torch.zeros(self.max_len, self.max_len, dtype=torch.long)
        if self.mode == "s2s":
            input_mask[:, :len(tokens_a)+2].fill_(1)
            second_st, second_end = len(
                tokens_a)+2, len(tokens_a)+len(tokens_b)+3
            input_mask[second_st:second_end, second_st:second_end].copy_(
                self._tril_matrix[:second_end-second_st, :second_end-second_st])
        else:
            st, end = 0, len(tokens_a) + len(tokens_b) + 3
            input_mask[st:end, st:end].copy_(self._tril_matrix[:end, :end])

        # Zero Padding for masked target
        if self.max_pred > n_pred:
            n_pad = self.max_pred - n_pred
            if masked_ids is not None:
                masked_ids.extend([0]*n_pad)
            if masked_pos is not None:
                masked_pos.extend([0]*n_pad)
            if masked_weights is not None:
                masked_weights.extend([0]*n_pad)

        oracle_pos = None
        oracle_weights = None
        oracle_labels = None
        if self.has_oracle:
            s_st, labls = instance[2:]
            oracle_pos = []
            oracle_labels = []
            for st, lb in zip(s_st, labls):
                st = st - num_truncated_a[0]
                if st > 0 and st < len(tokens_a):
                    oracle_pos.append(st)
                    oracle_labels.append(lb)
            oracle_pos = oracle_pos[:20]
            oracle_labels = oracle_labels[:20]
            oracle_weights = [1] * len(oracle_pos)
            if len(oracle_pos) < 20:
                x_pad = 20 - len(oracle_pos)
                oracle_pos.extend([0] * x_pad)
                oracle_labels.extend([0] * x_pad)
                oracle_weights.extend([0] * x_pad)

            return (input_ids, segment_ids, input_mask, mask_qkv, masked_ids,
                    masked_pos, masked_weights, -1, self.task_idx,
                    oracle_pos, oracle_weights, oracle_labels)
        if self.eval:
            return (input_ids, segment_ids, input_mask, mask_qkv, masked_ids, masked_pos, masked_weights, -1, self.task_idx)
        return (input_ids, segment_ids, input_mask, mask_qkv, masked_ids, masked_pos, masked_weights, -1, self.task_idx, label)


