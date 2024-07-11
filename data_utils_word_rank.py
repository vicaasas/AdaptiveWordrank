from torch.utils.data import Dataset
import os
import json
import torch
from transformers import BartTokenizerFast, PegasusTokenizer
import numpy as np
import string
from nltk.corpus import stopwords
from nltk import word_tokenize,sent_tokenize
import nltk
from datasets import load_from_disk,load_dataset

def to_cuda(batch, gpuid):
    for n in batch:
        if n != "data":
            batch[n] = batch[n].to(gpuid)


class BrioDataset(Dataset):
    def __init__(self, data_type, model_type,dataset, max_len=-1, is_test=False, total_len=1024, is_sorted=True, max_num=-1, is_untok=True, is_pegasus=False, num=-1):
        """ data format: article, abstract, [(candidiate_i, score_i)] """
        if dataset=="cnndm":
            # self.data = load_from_disk("/mnt/nas4/m11115088/WordRank/Dataset/cnndm_origin_norm_dataset_format")[data_type]
            self.data = load_from_disk("/mnt/nas4/m11115088/WordRank/Dataset/cnndm_gpt_all")[data_type]
            # self.data = load_dataset('cnn_dailymail',"3.0.0")[data_type]
        else:
            if data_type=="test":
                
                self.data = load_from_disk("/mnt/nas4/m11115088/WordRank/xsum/xsum_origin_norm_dataset_format")[data_type]
                # self.data = load_from_disk("/mnt/nas4/m11115088/WordRank/Dataset/xsum/xsum_test_gpt4_turbo")[data_type]
                # self.data = load_dataset('EdinburghNLP/xsum')[data_type]
            else:
                self.data = load_from_disk("/mnt/nas4/m11115088/WordRank/xsum/xsum_origin_norm_dataset_format")[data_type]
                # self.data = load_from_disk("/mnt/nas4/m11115088/WordRank/Dataset/xsum/xsum_100k_gpt4_turbo")[data_type]
                # self.data = load_from_disk("/mnt/nas4/m11115088/WordRank/Dataset/xsum/xsum_20k_gpt4_turbo")[data_type]
                # self.data = load_dataset('EdinburghNLP/xsum')[data_type]

        if "document" in self.data.features.keys():
            self.resource = "document"
            self.summary = "summary"
        else:
            self.resource = "article"
            self.summary = "highlights"
        
        self.num = len(self.data)
        if is_pegasus:
            self.tok = PegasusTokenizer.from_pretrained(model_type, verbose=False)
        else:
            self.tok = BartTokenizerFast.from_pretrained(model_type, verbose=False)
        self.maxlen = max_len
        self.is_test = is_test
        self.total_len = total_len
        self.sorted = is_sorted
        self.maxnum = max_num
        self.is_untok = is_untok
        self.is_pegasus = is_pegasus

    def __len__(self):
        return self.num
    def __getitem__(self, idx):
        # if self.isdir:
        #     with open(os.path.join(self.fdir, "%d.json"%idx), "r") as f:
        #         data = json.load(f)
        # else:
        #     with open(self.files[idx]) as f:
        #         data = json.load(f)
        # if self.is_untok:
        #     article = data["article_untok"]
        # else:
        #     article = data["article"]
        data = self.data[idx]
        article = sent_tokenize(data[self.resource])
        src_txt = " ".join(article).strip("\n")
        src_input_ids=[]
        src_word_position=[]
        seg_count = -1
        if self.is_pegasus:
            word_count = -1
            for i,src_stn in enumerate(article):
                sg_id=[]
                ton_stn = self.tok.tokenize(src_stn)
                for j,src_word in enumerate(ton_stn):
                    token=self.tok.convert_tokens_to_ids(src_word)
                    sg_id.append(token)
                    if "▁" in src_word:
                        word_count+=1
                    src_word_position.append(word_count)
                if len(sg_id)==1:
                    src_input_ids.extend(sg_id)
                else:
                    seg_count+=1
                    src_input_ids.extend(sg_id)
            # pass
        else:
            word_count = 0
            for i,src_stn in enumerate(article):
                sg_id=[]
                if i==0:
                    ton_stn = self.tok.tokenize(src_stn)
                else:
                    ton_stn = self.tok.tokenize(" "+src_stn)
                for j,src_word in enumerate(ton_stn):
                    token=self.tok.convert_tokens_to_ids(src_word)
                    sg_id.append(token)
                    if "Ġ" in src_word or (src_word in [".",","] and word_count!=0):
                        word_count+=1
                    src_word_position.append(word_count)
                if len(sg_id)==1:
                    if i==0:
                        seg_count+=1
                    src_input_ids.extend(sg_id)
                else:
                    seg_count+=1
                    src_input_ids.extend(sg_id)

        if self.is_pegasus:
            # src_input_ids = self.tok(src_txt, max_length=self.total_len, truncation=True, padding=False).input_ids
            src_input_ids = src_input_ids[:self.total_len-1]
            src_word_position = src_word_position[:self.total_len-1]
            src_input_ids = src_input_ids+[self.tok.eos_token_id]
            src_word_position = torch.tensor(src_word_position+[-2])
            # src_word_position = torch.tensor([-2])
        else:
            src_input_ids = src_input_ids[:self.total_len-2]
            src_word_position = src_word_position[:self.total_len-2]
            src_input_ids = [self.tok.bos_token_id]+src_input_ids+[self.tok.eos_token_id]
            src_word_position = torch.tensor([-1]+src_word_position+[-2])

        # segment_id = torch.tensor([998]+segment_id[:self.total_len-2]+[999])

        # src = self.tok.batch_encode_plus([src_txt], max_length=self.total_len, return_tensors="pt", padding=False, truncation=True)
        # x = src["input_ids"].squeeze(0)
        # src_input_ids = src_input_ids.squeeze(0)
        # candidates=[]
        # if self.is_untok:
        #     abstract = data["abstract_untok"]
        # else:
        abstract = data[self.summary]
        keyword=[]
        src_txt_list = word_tokenize(src_txt)
        src_txt_list_lower = word_tokenize(src_txt.lower())
        # candidate_ids=[]
        for abs_stn in sent_tokenize(abstract):
            nomal_abstract=word_tokenize(abs_stn)
            for i,(word,tag) in enumerate(nltk.pos_tag(nomal_abstract)):
                # if tag not in ["CC","CD","DT","MD","IN","VBZ","TO","WRB","WP$","WDT","WP",":"]:
                # if "NN" in tag or "VB" in tag or "JJ" in tag or "RB" in tag or "CD" in tag or "RP" in tag:
                if "NN" in tag or "VB" in tag or "JJ" in tag or "RB" in tag or "CD" in tag:
                # if "NN" in tag or "VB" in tag or "JJ" in tag or "RB" in tag:
                # if tag not in ["CC","CD","DT","MD","IN","VBZ","TO","WRB","WP$","WDT","WP",":"]:
                # if "NN" in tag or "VB" in tag or "JJ" in tag:
                # if "NN" in tag or "VB" in tag:
                    word=word.strip(string.punctuation)
                    if word !="" and  word not in keyword and (word.lower() not in stopwords.words('english')+list(string.punctuation)):
                        if word in src_txt_list:
                            keyword.append(word)
                        if word.lower() in src_txt_list:
                            keyword.append(word.lower())
                        if word.lower() in src_txt_list_lower:
                            w_idx = src_txt_list_lower.index(word.lower())
                            keyword.append(src_txt_list[w_idx])
        keyword = list(set(keyword))

        
        cand = self.tok(abstract, max_length=self.maxlen, return_tensors="pt", truncation=True, padding=False)
        
        candidate_ids = cand["input_ids"].squeeze(0)
        def find_subsequences(x, y):
            result = []
            y_len = len(y)
            for i in range(len(x)):
                if x[i] == y[0]:
                    if x[i:i+y_len] == y:
                        result.append([j for j in range(i, i+y_len)])
            return result
        keyword_position=[]
        keyword_ids=[]
        flag_tmp = []
        for _,word in enumerate(keyword):

            keyword_id=self.tok.encode(word, add_special_tokens=False)
            flag=0
            for interval in find_subsequences(src_input_ids, keyword_id):
                if interval not in flag_tmp and interval!=[]:
                    flag_tmp.append(interval)
                    keyword_position.extend(interval)
                    flag+=1
                if flag==1:
                    keyword_ids.extend(keyword_id)
            keyword_id=self.tok.encode(" "+word, add_special_tokens=False)
            flag=0
            for interval in find_subsequences(src_input_ids, keyword_id):
                if interval not in flag_tmp and interval!=[]:
                    flag_tmp.append(interval)
                    keyword_position.extend(interval)
                    flag+=1
                if flag==1:
                    keyword_ids.extend(keyword_id)
        # x = torch.tensor(src_input_ids)
        # y = torch.tensor(keyword_position)
        # z = x[y]
        # w = torch.isin(torch.tensor(keyword_ids),z)
        if self.is_pegasus:
            # add start token
            _candidate_ids = candidate_ids.new_zeros(candidate_ids.size(0) + 1)
            _candidate_ids[1:] = candidate_ids.clone()
            _candidate_ids[0] = self.tok.pad_token_id
            candidate_ids = _candidate_ids[:self.maxlen]
        result = {
            "src_input_ids": torch.tensor(src_input_ids), 
            "candidate_ids": candidate_ids,

            "src_word_position": src_word_position, 
            # "segment_ids":segment_id,
            "keyword_ids" : torch.tensor(list(set(keyword_ids))),
        }
        if self.is_test:
            result["data"] = data
        else:
            result["keyword_position"]=torch.tensor(keyword_position)
        return result


def collate_mp_brio(batch, pad_token_id, is_test=False):
    def pad(X,flag=False, max_len=-1):
        if max_len < 0:
            max_len = max(x.size(0) for x in X)
        if flag==True:
            result = torch.ones(len(X), max_len, dtype=X[0].dtype) * -100
        else:
            result = torch.ones(len(X), max_len, dtype=X[0].dtype) * pad_token_id
        for (i, x) in enumerate(X):
            result[i, :x.size(0)] = x
        return result

    src_input_ids = pad([x["src_input_ids"] for x in batch])
    src_word_position = pad([x["src_word_position"] for x in batch],flag=True)
    # segment_ids=pad([x["segment_ids"] for x in batch])
    # max_len = max([len(x) for x in src_input_ids])
    # src_input_ids = [pad(x, max_len) for x in src_input_ids]

    candidate_ids = pad([x["candidate_ids"] for x in batch])

    keyword_ids = pad([x["keyword_ids"] for x in batch],flag=True)
    
    if is_test:
        data = [x["data"] for x in batch]
    result = {
        "src_input_ids": src_input_ids,
        "candidate_ids": candidate_ids,
        "src_word_position":src_word_position,
        # "segment_ids":segment_ids,
        "keyword_ids":keyword_ids,
        # "labels":labels
    }
    if is_test:
        result["data"] = data
    else:
        result["keyword_position"] = pad([x["keyword_position"] for x in batch],flag=True)
        # result["keyword_position"] = [x["keyword_position"] for x in batch]
    return result