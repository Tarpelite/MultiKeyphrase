import json
import sys
import copy
import nltk


from seq2seq_loader import Seq2SeqDataset
from nltk import sent_tokenize
import pickle
# from transformers import BertTokenizer

from tqdm import *
import os

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
    def __init__(self,input_file, experiment):
        self.input_file = input_file
        self.experiment = experiment
    
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

        elif self.experiment == "single":
            clu_cnt = 0
            glo_cnt = 0
            map_dict = {}
            with open(self.input_file) as fin:
                for line in tqdm(fin.readlines()):
                    line = line.strip().split("\t")
                    docs = [load_full(json.loads(doc)) for doc in line]
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





