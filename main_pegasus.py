import torch
import torch.nn as nn
import torch.optim as optim
import argparse
import numpy as np
import os
import random
from compare_mt.rouge.rouge_scorer import RougeScorer
from transformers import (
    PegasusTokenizer,
    Adafactor,
    HammingDiversityLogitsProcessor
)
from utils import Recorder
from data_utils_word_rank import to_cuda, collate_mp_brio, AdaptiveWrodRankDataset
from torch.utils.data import DataLoader
import torch.distributed as dist
import torch.multiprocessing as mp
from functools import partial
from model import RankingLoss, BRIO
# from model_pegasus import RankingLoss, BRIO
import logging
from label_smoothing_loss import label_smoothing_loss
from nltk import sent_tokenize, word_tokenize
from config import cnndm_setting, xsum_setting
from tqdm import tqdm
from transformers.models.pegasus.configuration_pegasus import PegasusConfig
# from transformers import PhrasalConstraint,ConstraintListState,DisjunctiveConstraint
import nltk
import string
from transformers import get_linear_schedule_with_warmup
from nltk.corpus import stopwords
import statistics
# from modeling_bart import BartEncoder
# from torch.cuda.amp import autocast, GradScaler

logging.getLogger("transformers.tokenization_utils").setLevel(logging.ERROR)
logging.getLogger("transformers.tokenization_utils_base").setLevel(logging.ERROR)
logging.getLogger("transformers.tokenization_utils_fast").setLevel(logging.ERROR)
avg_word_energy=[]


def base_setting(args):
    args.batch_size = getattr(args, 'batch_size', 1) # batch size on one gpu, one step
    args.epoch = getattr(args, 'epoch', 1) 
    args.report_freq = getattr(args, "report_freq", 100) # report frequency
    args.accumulate_step = getattr(args, "accumulate_step", 16) # accumulate gradients steps
    args.margin = getattr(args, "margin", 0.001) # margin for ranking loss on candidate summaries
    args.gold_margin = getattr(args, "gold_margin", 0) # margin for ranking loss on gold summaries
    args.gold_weight = getattr(args, "gold_weight", 0) # weight for ranking loss on gold summaries
    args.mle_weight = getattr(args, "mle_weight", 1) # weight for mle loss on gold summaries
    args.rank_weight = getattr(args, "rank_weight", 1) # weight for ranking loss on candidate summaries
    args.model_type = getattr(args, "model_type", "facebook/bart-large-cnn") # model type
    args.warmup_steps = getattr(args, "warmup_steps", 10000) # warmup steps
    args.normalize = getattr(args, "normalize", True) # normalize predicited likelihood
    args.grad_norm = getattr(args, "grad_norm", 0) # gradient norm
    args.seed = getattr(args, "seed", 970903) # random seed
    args.no_gold = getattr(args, "no_gold", False) # whether to use gold summaries
    args.pretrained = getattr(args, "pretrained", None) # pretrained model path
    args.max_lr = getattr(args, "max_lr", 2e-3) # max learning rate (* 1e-2)
    args.scale = getattr(args, "scale", 1) # scale of ranking loss
    args.score_mode = getattr(args, "score_mode", "log") # use log-likelihood for ranking loss
    args.datatype = getattr(args, "datatype", "diverse") # data type
    args.dataset = getattr(args, "dataset", "cnndm") # dataset cnndm/diverse/test
    args.max_len = getattr(args, "max_len", 120) # max length of summary
    args.max_num = getattr(args, "max_num", 16) # max number of candidate summaries
    args.smooth = getattr(args, "smooth", 0.1) # label smoothing
    args.total_len = getattr(args, "total_len", 512) # total length of source article
    args.length_penalty = getattr(args, "length_penalty", 2.0) # length penalty
    args.do_sample = getattr(args, "do_sample", True) # whether to generaet summaries during evaluation
    args.gen_max_len = getattr(args, "gen_max_len", 140) # max length of generated summaries
    args.gen_min_len = getattr(args, "gen_min_len", 55) # min length of generated summaries
    args.is_pegasus = getattr(args, "is_pegasus", False) # whether to use Pegasus as the baseline model
    args.adding = getattr(args, "adding", 0) # used for numerical stability
    args.eval_interval = getattr(args, "eval_interval", 1000) # evaluation intervals
    args.num_beams = getattr(args, "num_beams", 4) # number of beams for beam search

def evaluation(args):
    # load data
    if args.config == "cnndm":
        cnndm_setting(args)
        args.model_type = "/mnt/nas4/m11115088/WordRank/CNN_Model/GPT_final_chpt_pegasus"
        args.max_len=1024
        args.total_len=1024
        args.gen_max_len=128
        args.gen_min_len=32
        args.num_beams=8
        args.length_penalty=0.8
    elif args.config == "xsum":
        xsum_setting(args)
        args.model_type = "/mnt/nas4/m11115088/WordRank/XSUM_Models/GPT_pegasus"
        # args.model_type = "google/pegasus-large"
        args.max_len=512
        args.total_len=512
        args.gen_max_len=62 + 2
        args.gen_min_len=10 + 1
        args.num_beams=8
        args.length_penalty=0.6
    else:
        base_setting(args)
    # if args.is_pegasus:
    #     tok = PegasusTokenizer.from_pretrained(args.model_type)
    # else:

    tok = PegasusTokenizer.from_pretrained(args.model_type)
    collate_fn = partial(collate_mp_brio, pad_token_id=tok.pad_token_id, is_test=True)
    # test_set = AdaptiveWrodRankDataset(f"/work/u5516210/BRIO/cnndm/diverse/test", args.model_type, is_test=True, max_len=1024,
    test_set = AdaptiveWrodRankDataset(f"test", args.model_type,dataset=args.config, is_test=True, max_len=512,
     is_sorted=False, max_num=args.max_num, is_untok=True, total_len=args.total_len, is_pegasus=args.is_pegasus)
    # test_set = AdaptiveWrodRankDataset(f"/work/u5516210/BRIO/xsum/diverse/test", args.model_type, is_test=True, max_len=512,
    # test_set = AdaptiveWrodRankDataset(f"/work/u5516210/BRIO/cnndm/diverse/test", args.model_type, is_test=True, max_len=512,
    # test_set = AdaptiveWrodRankDataset(f"/work/u5516210/ctrl-sum/datasets/cnndm/test.source",
    #                         "/work/u5516210/ctrl-sum/datasets/cnndm/test.target",
    #                         "facebook/bart-large-cnn", max_len=120, is_test=True,total_len=1024)
    batch_size = 12
    dataloader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=12, collate_fn=collate_fn)
    # build models
    
    model_path = args.pretrained if args.pretrained is not None else args.model_type
    config = PegasusConfig.from_pretrained(model_path)
    config.auto_calculate_SCAN_threshold = args.auto_calculate_SCAN_threshold
    config.keyword_threshold = args.keyword_threshold
    model = BRIO(model_path, tok.pad_token_id,config, args.is_pegasus)
    device = f'cuda:{args.gpuid[0]}'
    model = model.to(device)
    
    model.load_state_dict(torch.load(os.path.join("./", args.model_pt), map_location=f'cuda:{args.gpuid[0]}'))

    model.eval()

    # model_name = "pegasus_llm"
    model_name = args.model_pt.replace("/", "_").replace(".bin", "")

    print(model_name)
    def mkdir(path):
        if not os.path.exists(path):
            os.mkdir(path)

    # print(model_name)
    root_dir = "./result/%s"%model_name
    mkdir(root_dir)
    rouge_scorer = RougeScorer(['rouge1', 'rouge2', 'rougeLsum'], use_stemmer=True)
    def process(x):
        return sent_tokenize(" ".join(word_tokenize(x.strip())))
    if args.do_reranking:
        # evaluate the model as a scorer
        mkdir("./result/%s/reference_ranking"%model_name)
        mkdir("./result/%s/candidate_ranking"%model_name)
        rouge1, rouge2, rougeLsum = 0, 0, 0
        cnt = 0
        # model.scoring_mode()
        tokenizer = tok
        count = 1
        # bsz = 8
        
        # model.generation_mode()
        # word_size=10
        with torch.no_grad():
            with tqdm(dataloader, ncols=100) as t:
                for batch in dataloader:
                    if args.cuda:
                        to_cuda(batch, args.gpuid[0])
                    samples = batch["data"]
                    # slines = [x["document"] for x in samples]
                    # dct = tok.batch_encode_plus(slines, max_length=args.total_len, return_tensors="pt", padding=True, truncation=True)
                    
                    input_mask = batch["src_input_ids"] != tok.pad_token_id

                    summaries = model.generate(
                        input_ids=batch["src_input_ids"].to(device),
                        attention_mask=input_mask.to(device),
                        # segment_ids=batch["segment_ids"].to(device),
                        max_length=args.gen_max_len,  # +2 from original because we start at step=1 and stop before max_length
                        min_length=args.gen_min_len,  # +1 from original because we start at step=1
                        no_repeat_ngram_size=3,
                        num_beams=args.num_beams,
                        # num_beam_groups = 8,
                        # num_return_sequences = 8,
                        # diversity_penalty = 1.0,
                        # repetition_penalty = 1.0,
                        length_penalty=args.length_penalty,
                        early_stopping=True,
                        src_word_position=batch["src_word_position"]
                    )
                    dec = [tok.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=True) for g in summaries]
                    # samples = [tok.decode(c, skip_special_tokens=True, clean_up_tokenization_spaces=False) for c in batch["candidate_ids"].squeeze(1)]
                    for (hypothesis, ref) in zip(dec, samples):
                        hypothesis = hypothesis.replace("\n", " ")
                        ref = ref["summary"]
                        x = process(ref)
                        y = process(hypothesis)
                        score = rouge_scorer.score("\n".join(x), "\n".join(y))
                        rouge1 += score["rouge1"].fmeasure
                        rouge2 += score["rouge2"].fmeasure
                        rougeLsum += score["rougeLsum"].fmeasure

                        with open("./result/%s/candidate_ranking/%d.dec"%(model_name, cnt), "w") as f:
                            for s in y:
                                print(s, file=f)
                        with open("./result/%s/reference_ranking/%d.ref"%(model_name, cnt), "w") as f:
                            for s in x:
                                print(s, file=f)
                        cnt+=1
                    t.set_postfix(rouge1 = f"{rouge1 / cnt:.4f}",rouge2 = f"{rouge2 / cnt:.4f}",rougeLsum = f"{rougeLsum / cnt:.4f}")
                    t.update(1)
        rouge1 = rouge1 / cnt
        rouge2 = rouge2 / cnt
        rougeLsum = rougeLsum / cnt
        print("ranking rouge1: %.6f, rouge2: %.6f, rougeL: %.6f"%(rouge1, rouge2, rougeLsum))

    if args.do_generation:
        # evaluate the model as a generator
        rouge1, rouge2, rougeLsum = 0, 0, 0
        tokenizer = tok
        count = 1
        bsz = 8
        model.generation_mode()
        total_num = len(os.listdir(f"/mnt/nas2/m11115088/SimCLS/preprocess_data/{args.dataset}/{args.datatype}/test"))
        with open(f'./{args.dataset}/{args.datatype}/test.source') as source, open(os.path.join(root_dir, "test.out"), 'w') as fout:
            sline = source.readline().strip()
            slines = [sline]
            for sline in tqdm(source, total=total_num):
                if count % bsz == 0:
                    with torch.no_grad():
                        dct = tokenizer.batch_encode_plus(slines, max_length=args.total_len, return_tensors="pt", pad_to_max_length=True, truncation=True)
                        summaries = model.generate(
                            input_ids=dct["input_ids"].to(device),
                            attention_mask=dct["attention_mask"].to(device),
                            max_length=args.gen_max_len + 2,  # +2 from original because we start at step=1 and stop before max_length
                            min_length=args.gen_min_len + 1,  # +1 from original because we start at step=1
                            no_repeat_ngram_size=3,
                            num_beams=args.num_beams,
                            length_penalty=args.length_penalty,
                            early_stopping=True,
                        )
                        dec = [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=False) for g in summaries]
                    for hypothesis in dec:
                        hypothesis = hypothesis.replace("\n", " ")
                        fout.write(hypothesis + '\n')
                        fout.flush()
                    slines = []
                sline = sline.strip()
                if len(sline) == 0:
                    sline = " "
                slines.append(sline)
                count += 1
            if slines != []:
                with torch.no_grad():
                    dct = tokenizer.batch_encode_plus(slines, max_length=args.total_len, return_tensors="pt", pad_to_max_length=True, truncation=True)
                    summaries = model.generate(
                        input_ids=dct["input_ids"].to(device),
                        attention_mask=dct["attention_mask"].to(device),
                        max_length=args.gen_max_len + 2,  # +2 from original because we start at step=1 and stop before max_length
                        min_length=args.gen_min_len + 1,  # +1 from original because we start at step=1
                        no_repeat_ngram_size=3,
                        num_beams=args.num_beams,
                        length_penalty=args.length_penalty,
                        early_stopping=True,
                    )
                    dec = [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=False) for g in summaries]
                    for hypothesis in dec:
                        hypothesis = hypothesis.replace("\n", " ")
                        fout.write(hypothesis + '\n')
                        fout.flush()
        # calculate rouge score
        def process(x):
            return sent_tokenize(" ".join(word_tokenize(x.strip())))
        
        with open(os.path.join(root_dir, "test.out")) as fout, open(f'./{args.dataset}/test.target') as target:
            for (hyp, ref) in zip(fout, target):
                hyp = hyp.strip()
                ref = ref.strip()
                hyp = process(hyp)
                ref = process(ref)
                score = rouge_scorer.score("\n".join(ref), "\n".join(hyp))
                rouge1 += score["rouge1"].fmeasure
                rouge2 += score["rouge2"].fmeasure
                rougeLsum += score["rougeLsum"].fmeasure
            rouge1 = rouge1 / total_num
            rouge2 = rouge2 / total_num
            rougeLsum = rougeLsum / total_num
            print("evaluation rouge1: %.6f, rouge2: %.6f, rougeL: %.6f"%(rouge1, rouge2, rougeLsum))


def test(dataloader, gen_dataloader, model, args, tok, gpuid, do_sample=False):
    model.eval()
    if args.cuda:
        device = f"cuda:{gpuid}"
    else:
        device = "cpu"
    if len(args.gpuid) > 1:
        _model = model.module
    else:
        _model = model
    cnt = 0
    rouge_scorer = RougeScorer(['rouge1', 'rouge2', 'rougeLsum'], use_stemmer=True)
    # rouge1, rouge2, rougeLsum = 0, 0, 0
    mle_loss = 0
    word_rank_loss = 0
    if args.smooth > 0:
        mle_fn = label_smoothing_loss(ignore_index=tok.pad_token_id, epsilon=args.smooth)
    else:
        mle_fn = nn.CrossEntropyLoss(ignore_index=tok.pad_token_id)
    
    cnt = 0
    sample_rouge1, sample_rouge2, sample_rougeLsum = 0, 0, 0
    if do_sample:
        # generation
        
        def process(x):
            return sent_tokenize(" ".join(word_tokenize(x.strip())))
        with torch.no_grad():
            with tqdm(gen_dataloader) as t:
                for (i, batch) in enumerate(gen_dataloader):
                    if args.cuda:
                        to_cuda(batch, device)
                    samples = batch["data"]
                    # slines = [x["document"] for x in samples]
                    # dct = tok.batch_encode_plus(slines, max_length=args.total_len, return_tensors="pt", pad_to_max_length=True, truncation=True)
                    input_mask = batch["src_input_ids"] != tok.pad_token_id
                    summaries = _model.generate(
                        input_ids=batch["src_input_ids"].to(device),
                        attention_mask=input_mask.to(device),
                        max_length=args.gen_max_len + 2,  # +2 from original because we start at step=1 and stop before max_length
                        min_length=args.gen_min_len + 1,  # +1 from original because we start at step=1
                        no_repeat_ngram_size=3,
                        num_beams=args.num_beams,
                        length_penalty=args.length_penalty,
                        early_stopping=True,
                        src_word_position=batch["src_word_position"]
                    )
                    dec = [tok.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=False) for g in summaries]
                    for (hypothesis, x) in zip(dec, samples):
                        hypothesis = hypothesis.replace("\n", " ")
                        ref = x["summary"]
                        x = process(ref)
                        y = process(hypothesis)
                        score = rouge_scorer.score("\n".join(x), "\n".join(y))
                        sample_rouge1 += score["rouge1"].fmeasure
                        sample_rouge2 += score["rouge2"].fmeasure
                        sample_rougeLsum += score["rougeLsum"].fmeasure
                        cnt += 1
                        # res.append({"predict_summary": hypothesis,"human_summary":ref})
                        
                    t.set_postfix(rouge1 = f"{sample_rouge1 / cnt:.4f}",rouge2 = f"{sample_rouge2 / cnt:.4f}",rougeLsum = f"{sample_rougeLsum / cnt:.4f}")
                    t.update(1)
                    # break
        sample_rouge1 = sample_rouge1 / cnt
        sample_rouge2 = sample_rouge2 / cnt
        sample_rougeLsum = sample_rougeLsum / cnt
        if len(args.gpuid) > 1:
            sample_rouge1 = torch.FloatTensor([sample_rouge1]).to(device)
            dist.all_reduce(sample_rouge1, op=dist.reduce_op.SUM)
            sample_rouge1 = sample_rouge1.item() / len(args.gpuid)
            sample_rouge2 = torch.FloatTensor([sample_rouge2]).to(device)
            dist.all_reduce(sample_rouge2, op=dist.reduce_op.SUM)
            sample_rouge2 = sample_rouge2.item() / len(args.gpuid)
            sample_rougeLsum = torch.FloatTensor([sample_rougeLsum]).to(device)
            dist.all_reduce(sample_rougeLsum, op=dist.reduce_op.SUM)
            sample_rougeLsum = sample_rougeLsum.item() / len(args.gpuid)
    model.train()
    return {
        # "rouge1": rouge1,
        # "rouge2": rouge2,
        # "rougeLsum": rougeLsum,
        "sample_rouge1": sample_rouge1,
        "sample_rouge2": sample_rouge2,
        "sample_rougeLsum": sample_rougeLsum,
        } 

def WordRankingLoss(text_id,keyword_position,word_energy,tok,d_model=1024,length_penalty=1):
    global avg_word_energy
    input_mask = (text_id != tok.pad_token_id) & (text_id != tok.eos_token_id) & (text_id != tok.bos_token_id)
    word_energy.masked_fill_(~input_mask, torch.finfo(torch.float32).min)
    # word_energy = word_energy*input_mask
    word_rank_loss = torch.tensor([0.0], dtype=torch.float32,requires_grad=True,device=word_energy.device)
    keyword_position = keyword_position.type(torch.int64)
    for i,_ in enumerate(text_id):
        flag_idx=torch.where(keyword_position[i]==-100,False,True)
        idx=keyword_position[i][flag_idx]
        if idx.size(0)!=0:
            txt_len=sum(input_mask[i])
            word_energy[i][idx]=word_energy[i][idx]-1e-6
            topk=torch.topk(word_energy[i],idx.size(0),0)
            num = int(idx.size(0)*1.5)
            if num>txt_len:
                num = idx.size(0)
            avg_word_energy.extend(torch.topk(word_energy[i],num,0).values.tolist())
            diff = ~torch.isin(topk.indices, idx)
            diff_eng = ~torch.isin(idx,topk.indices)

            diff_num = sum(diff)
            if diff_num==0:
                continue

            word_energy[i][idx]=word_energy[i][idx]+1e-6
            top_max_key_values=(topk.values[diff]).sum(0)
            positive_energy = word_energy[i][idx[diff_eng]].sum(0)
            ones = torch.ones_like(top_max_key_values)
            word_rank_loss_func = torch.nn.MarginRankingLoss(
                # 0.0
                # diff_num*(txt_len/(idx.size(0)*(d_model**0.5)))
                # diff_num*(txt_len/(idx.size(0)*d_model**0.99))
                diff_num*(txt_len/(idx.size(0)*d_model))
            )
            word_rank_loss = word_rank_loss+ word_rank_loss_func(
                positive_energy/(diff_num**length_penalty), 
                top_max_key_values/(diff_num**length_penalty),
                ones
            )

        
    return word_rank_loss/text_id.size(0)
# python -m torch.distributed.launch --nproc_per_node=2 main_pegasus.py --cuda --gpuid 0 1 --config cnndm -l

# python main_pegasus.py --cuda --gpuid 0 --config cnndm -l
def run(rank, args):
    global avg_word_energy
    if args.config == "cnndm":
        cnndm_setting(args)
        model_path = "google/pegasus-cnn_dailymail"
        args.model_type = "google/pegasus-cnn_dailymail"
        args.batch_size=1
        args.accumulate_step=16
        args.max_len=1024
        args.total_len=1024
        args.gen_max_len=128 + 2
        args.gen_min_len=32 + 1
        args.num_beams=8
        args.length_penalty=0.8
    elif args.config == "xsum":
        xsum_setting(args)
        # model_path = "/mnt/nas4/m11115088/WordRank/XSUM_Models/pegasus"
        # args.model_type = "/mnt/nas4/m11115088/WordRank/XSUM_Models/pegasus"
        model_path = "google/pegasus-xsum"
        args.model_type = "google/pegasus-xsum"
        args.max_len=512
        args.total_len=512
        args.batch_size=4
        args.accumulate_step=4
    else:
        base_setting(args)
    # task initialization
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    gpuid = args.gpuid[rank]
    is_master = rank == 0
    is_mp = len(args.gpuid) > 1
    world_size = len(args.gpuid)
    if is_master:
        id = len(os.listdir("./cache"))
        recorder = Recorder(args.log,args.name)
    recorder.print(f"load chpt from: {args.model_pt if args.model_pt != '' else args.model_type}")

    # build dataloader
    # if args.is_pegasus:
    #     tok = PegasusTokenizer.from_pretrained(args.model_type)
    # else:


    tok = PegasusTokenizer.from_pretrained(args.model_type)
    config = PegasusConfig.from_pretrained(model_path)
    config.auto_calculate_SCAN_threshold = True
    config.keyword_threshold = args.keyword_threshold
    collate_fn = partial(collate_mp_brio, pad_token_id=tok.pad_token_id, is_test=False)
    collate_fn_val = partial(collate_mp_brio, pad_token_id=tok.pad_token_id, is_test=True)
    # train_set = AdaptiveWrodRankDataset(f"/work/u5516210/BRIO/cnndm/diverse/train", args.model_type,is_sorted=False, max_len=args.max_len, max_num=args.max_num, total_len=args.total_len, is_pegasus=args.is_pegasus)
    # train_set = AdaptiveWrodRankDataset(f"/work/u5516210/BRIO/cnndm_cased/train", args.model_type,is_sorted=False, max_len=args.max_len, max_num=args.max_num, total_len=args.total_len, is_pegasus=args.is_pegasus)
    train_set = AdaptiveWrodRankDataset(f"train", args.model_type,dataset=args.config,
                            is_sorted=False, max_len=args.max_len, max_num=args.max_num, total_len=args.total_len, is_pegasus=args.is_pegasus)
    # train_set = AdaptiveWrodRankDataset(f"/mnt/nas2/m11115088/SimCLS/preprocess_data/{args.dataset}/{args.datatype}/train", args.model_type,is_sorted=False, max_len=args.max_len, max_num=args.max_num, total_len=args.total_len, is_pegasus=args.is_pegasus)
    val_set = AdaptiveWrodRankDataset(f"test", args.model_type,dataset=args.config, is_test=True,
                            is_sorted=False, max_len=args.max_len, max_num=args.max_num, total_len=args.total_len, is_pegasus=args.is_pegasus)
    # val_set = AdaptiveWrodRankDataset(f"/work/u5516210/BRIO/cnndm_cased/val", args.model_type, is_test=True, max_len=512, is_sorted=False, max_num=args.max_num, total_len=args.total_len, is_pegasus=args.is_pegasus)

    dataloader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=8, collate_fn=collate_fn)
        # val_dataloader = DataLoader(val_set, batch_size=1, shuffle=False, num_workers=4, collate_fn=collate_fn_val)
    val_dataloader = DataLoader(val_set, batch_size=4, shuffle=False, num_workers=4, collate_fn=collate_fn_val)
    val_gen_dataloader = DataLoader(val_set, batch_size=8, shuffle=False, num_workers=8, collate_fn=collate_fn_val)
    # build models
    model = BRIO(model_path, tok.pad_token_id,config, is_pegasus=args.is_pegasus)
    
    if len(args.model_pt) > 0:
        model.load_state_dict(torch.load(os.path.join("./", args.model_pt), map_location=f'cuda:{gpuid}'),strict=True)
        # model.load_state_dict(torch.load(os.path.join("./cache", args.model_pt), map_location='cpu'),strict=False)

    model = model.to(f"cuda:{gpuid}")
    model.train()
    # set the model to scoring mode
    # if is_mp:
    #     model.module.scoring_mode()
    # else:
    #     model.scoring_mode()
    if args.smooth > 0:
        mle_fn = label_smoothing_loss(ignore_index=tok.pad_token_id, epsilon=args.smooth)
    else:
        mle_fn = nn.CrossEntropyLoss(ignore_index=tok.pad_token_id)
    # bart_encoder = BartEncoder(config)
    # energy_model = WordEnergyModel(model)
        
    def linear_lr_decay(step, total_steps, warmup_ratio):
        if step < total_steps * warmup_ratio:
            return step / (total_steps * warmup_ratio)
        else:
            return 1 - (step - total_steps * warmup_ratio) / (total_steps * (1 - warmup_ratio))

    d_optimizer = optim.Adam(model.parameters(),lr=args.max_lr)
    # d_optimizer = optim.AdamW(model.parameters())
    # d_optimizer = Adafactor(model.parameters(),lr=args.max_lr,relative_step=False)

    # scaler = GradScaler()
    all_step_cnt = 0
    
    if is_mp:
        if is_master:
            id = torch.FloatTensor([id]).to(gpuid)
        else:
            id = torch.zeros(1).to(gpuid)
        dist.all_reduce(id, op=dist.reduce_op.SUM)
        id = int(id.item())
    # define evaluation function

    # start training
    cn=0
    
    end = len(dataloader)
    def linear_lr_decay(step, total_steps, warmup_ratio):
        if step < total_steps * warmup_ratio:
            return step / (total_steps * warmup_ratio)
        else:
            return 1 - (step - total_steps * warmup_ratio) / (total_steps * (1 - warmup_ratio))
    if args.dataset == "xsum":
        def eval_fn(rouge1, rouge2, rougeLsum):
            return 1 - 2 * rouge1 * rouge2 / (rouge1 + rouge2)
    else:
        def eval_fn(rouge1, rouge2, rougeLsum):
            return 1 - (rouge1 * rouge2 + rougeLsum) / 3
    warmup_ratio = 0.4
    minimum_mle_loss = 1e5


    for epoch in range(args.epoch):
        d_optimizer.zero_grad()
        avg_mle_loss = 0
        avg_word_ranking_loss = 0
        step_cnt = 0
        epoch_step = 0
        avg_loss = 0
        avg_word_energy=[]

        # scheduler.step(avg_word_ranking_loss)
        scheduler = optim.lr_scheduler.LambdaLR(d_optimizer, lr_lambda=lambda step: linear_lr_decay(step, int(end/args.accumulate_step)+1, warmup_ratio))
        with tqdm(dataloader, ncols=100) as t:
            for (i, batch) in enumerate(dataloader):
                # batch = fifth_batch
                # if i<80312:
                #     t.update(1)
                #     continue
                if args.cuda:
                    to_cuda(batch, gpuid)
                # if batch["src_input_ids"].size(1)==2 or batch["keyword_position"].size(1)==0:
                #     continue
                if batch["src_input_ids"].size(1)==2:
                    continue
                step_cnt += 1
                # with autocast():
                output = model(
                    text_id=batch["src_input_ids"],
                    candidate_id=batch["candidate_ids"],
                    keyword_ids=batch["keyword_ids"],
                    keyword_position=batch["keyword_position"],
                    src_word_position=batch["src_word_position"],  
                )
                # output = model(batch["src_input_ids"], batch["candidate_ids"],batch["keyword_ids"],batch["src_word_position"], args.normalize, args.score_mode, args.length_penalty, adding=args.adding)
                #word_rank_loss = WordRankingLoss(batch["src_input_ids"],batch["keyword_ids"],output["word_energy"],config.d_model)
                word_rank_loss = WordRankingLoss(batch["src_input_ids"],batch["keyword_position"],output["word_energy"],tok,config.d_model)

                probs = output["probs"][:, :-1]  # truncate last token
                gold = batch["candidate_ids"][:, 1:]  # shift right
                # gold = batch["candidate_ids"][:, 0, 1:]  # shift right
                mle_loss = mle_fn(probs.transpose(1, 2), gold)
                # loss =  args.mle_weight * mle_loss
                all_loss =  args.mle_weight * mle_loss + args.word_rank_weight * word_rank_loss

                loss = all_loss / args.accumulate_step
                avg_loss += loss.item()
                avg_mle_loss += mle_loss.item() / args.accumulate_step
                avg_word_ranking_loss += word_rank_loss.item() / args.accumulate_step
                
                # scaler.scale(loss).backward()
                loss.backward()
                if step_cnt == args.accumulate_step:
                    # updating
                    if args.grad_norm > 0:
                        nn.utils.clip_grad_norm_(model.parameters(), args.grad_norm)
                    step_cnt = 0
                    epoch_step += 1
                    all_step_cnt += 1
                    d_optimizer.step()
                    scheduler.step()
                    d_optimizer.zero_grad()

                if epoch_step % args.report_freq == 0 and step_cnt == 0 and is_master:
                    # report stats
                    # scheduler.step(avg_word_ranking_loss)
                    
                    print("id: %d"%id)
                    # print(f"similarity: {similarity[:, :10]}")
                    # if not args.no_gold:
                    #     print(f"gold similarity: {gold_similarity}")
                    recorder.print("epoch: %d, batch: %d, avg loss: %.6f, avg word ranking loss: %.6f, avg mle loss: %.6f, avg word energy: %.6f"
                    %(epoch+1, epoch_step, avg_loss / args.report_freq, avg_word_ranking_loss / args.report_freq, avg_mle_loss / args.report_freq, statistics.mean(avg_word_energy)))
                    # recorder.print(f"learning rate: {lr:.6f}")
                    recorder.print(f"encoder learning rate: {d_optimizer.param_groups[0]['lr']:.10f}")
                    recorder.plot("loss", {"loss": avg_loss / args.report_freq}, all_step_cnt)
                    recorder.plot("mle_loss", {"loss": avg_mle_loss / args.report_freq}, all_step_cnt)
                    recorder.plot("avg_word_ranking_loss", {"loss": avg_word_ranking_loss / args.report_freq}, all_step_cnt)

                    # recorder.print()
                    # avg_mle_loss, avg_loss = 0, 0
                    avg_mle_loss, avg_loss,avg_word_ranking_loss = 0, 0 , 0

                # if i == end-1 or i==int(end*1/3)or i==int(end*2/3):

                # if 1:
                if all_step_cnt % args.eval_interval == 0 and all_step_cnt != 0 and step_cnt == 0:
                # if i == end-1:
                    # save current model
                    cn+=1
                    if is_master:
                        if is_mp:
                            recorder.save(model.module, "model_cur"+str(cn)+".bin")
                        else:
                            recorder.save(model, "model_cur"+str(cn)+".bin")
                    recorder.print(statistics.mean(avg_word_energy))
                    avg_word_energy=[]
                    result = test(val_dataloader, val_gen_dataloader, model, args, tok, gpuid, args.do_sample)
                    if args.do_sample:
                        mle_loss = eval_fn(result["sample_rouge1"], result["sample_rouge2"], result["sample_rougeLsum"])
                    else:
                        mle_loss = result["mle_loss"]
                    if mle_loss < minimum_mle_loss and is_master:
                        minimum_mle_loss = mle_loss
                        if is_mp:
                            recorder.save(model.module, "model_generation.bin")
                        else:
                            recorder.save(model, "model_generation.bin")
                        recorder.print("best generation loss - epoch: %d, batch: %d"%(epoch, i / args.accumulate_step))
                    if is_master:
                        recorder.print("val generation loss: %.6f"%(mle_loss))
                        if args.do_sample:
                            recorder.print("val generation rouge1: %.6f, rouge2: %.6f, rougeLsum: %.6f"
                            %(result["sample_rouge1"], result["sample_rouge2"], result["sample_rougeLsum"]))
                
                try:
                    t.set_postfix(all_loss=f"{all_loss.item():.4f}",WR_loss = f"{word_rank_loss.item():.4f}",mle_loss = f"{mle_loss.item():.4f}")
                except:
                    pass
                t.update(1)


if __name__ ==  "__main__":
    parser = argparse.ArgumentParser(description='Parameters')
    parser.add_argument("--cuda", action="store_true", help="use cuda")
    parser.add_argument("--gpuid", nargs='+', type=int, default=0, help="gpu ids")
    parser.add_argument("-e", "--evaluate", action="store_true", help="evaluate model")
    parser.add_argument("-r", "--do_reranking", action="store_true", help="do reranking evaluation")
    parser.add_argument("-g", "--do_generation", action="store_true", help="do generation evaluation")
    parser.add_argument("--name", default="", type=str, help="project name")
    parser.add_argument("-l", "--log", action="store_true", help="logging")
    parser.add_argument("-p", "--port", type=int, default=12355, help="port")
    parser.add_argument("--model_pt", default="", type=str, help="model path")
    parser.add_argument("--config", default="", type=str, help="config path")
    parser.add_argument("--auto_calculate_SCAN_threshold", default=False, type=bool, help="auto calculate SCAN threshold")
    parser.add_argument("-is_pegasus", "--is_pegasus", action="store_true", help="use pegasus")
    args = parser.parse_args()
    if args.evaluate:
        evaluation(args)
    else:
        run(0, args)

