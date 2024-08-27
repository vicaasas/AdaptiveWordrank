from transformers import PegasusTokenizer, BartTokenizer, PegasusTokenizer
from modeling_pegasus_origin import PegasusForConditionalGeneration
from modeling_bart_SSAN import BartForConditionalGeneration
# from modeling_bart_origin import BartForConditionalGeneration
import torch
import argparse
from datasets import load_dataset, load_from_disk
import evaluate
from nltk import sent_tokenize
import json
from tqdm import tqdm
from data_utils_word_rank import to_cuda, collate_mp_brio, BrioDataset
from functools import partial
from torch.utils.data import DataLoader
from statistics import mean

def generate_summaries_cnndm(args):
    rouge = evaluate.load("rouge",experiment_id="cnndm")
    device = f"cuda:{args.gpuid}"
    mname = "CNN_Model/GPT_final_chpt_bart"
    model = BartForConditionalGeneration.from_pretrained(mname)
    # x = torch.load("CNN_Model/cased_bart_WR/model_cur3.bin", map_location='cpu')
    # x = torch.load("cache/07-11-16-08-1720685307_BART_SCAN_10_new_c_again/model_generation.bin", map_location='cpu')
    x = torch.load("cache/08-01-20-02-1722513737_BART_cnndm_origin_only_SCAN/model_generation.bin", map_location='cpu')
    new_x = {}
    for key, value in x.items():
        if "final_logits_bias" in key or "lm_head.weight" in key:
            new_x[key.replace('model.', '')] =  value 
        else:
            new_x[key.replace('model.model.', 'model.')] =  value 
    model.load_state_dict(new_x)
    model.eval()
    model.to(device)
    tokenizer = BartTokenizer.from_pretrained(mname)
    max_length = 140
    min_length = 55
    count = 0
    # dataset = load_from_disk("/mnt/nas4/m11115088/WordRank/Dataset/cnndm_origin_norm_dataset_format")['train']
    # dataset = load_from_disk("/mnt/nas4/m11115088/WordRank/Dataset/cnndm_gpt_all")['train']
    # length = len(dataset)
    collate_fn = partial(collate_mp_brio, pad_token_id=tokenizer.pad_token_id, is_test=True)
    dataset = BrioDataset(f"train", mname,dataset=args.dataset, is_test=True,
                         max_len=1024, total_len=1024, is_pegasus=False)
    gen_dataloader = DataLoader(dataset, batch_size=8, shuffle=False, num_workers=12, collate_fn=collate_fn)
    # dataset = dataset.select(range(0,40000))
    # dataset = dataset.select(range(82232,110000))
    # dataset = dataset.select(range(80000,140000))
    # dataset = dataset.select(range(142552,170000))
    # dataset = dataset.select(range(200000,length))
    # dataset = dataset.select(range(0,int(length/2)))
    # dataset = dataset.select(range(int(length/2),length))
    # full_index = 0
    full_index = 270000
    print(full_index)
    # with open(args.tgt_dir, 'w') as fout:
    # with open(args.src_dir) as source, open(args.tgt_dir, 'w') as fout:
        # sline = source.readline().strip().lower()

    avg_score=[]
    with torch.no_grad():
        with tqdm(gen_dataloader, ncols=100) as t:
            for (i, batch) in enumerate(gen_dataloader):
                to_cuda(batch, args.gpuid)
                samples = batch["data"]
                # dct = tokenizer.batch_encode_plus(train['document'], max_length=1024, return_tensors="pt", pad_to_max_length=True, truncation=True)
                input_mask = batch["src_input_ids"] != tokenizer.pad_token_id
                summaries = model.generate(
                    input_ids=batch["src_input_ids"],
                    attention_mask=input_mask.to(device),
                    num_return_sequences=16, num_beam_groups=16, diversity_penalty=1.0, num_beams=16,
                    max_length=max_length + 2,  # +2 from original because we start at step=1 and stop before max_length
                    min_length=min_length + 1,  # +1 from original because we start at step=1
                    no_repeat_ngram_size=3,
                    length_penalty=2.0,
                    early_stopping=True,
                    src_word_position=batch["src_word_position"]
                )
                dec = [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=False) for g in summaries]
            # for hypothesis,gpt_gen in zip(dec,gpts):
                idx=0
                for i in range(0,len(dec),16):
                    output={}
                    hypothesis = dec[i:i+16]
                    gpt_gen = [samples[idx]["summary"]]*16
                    rouge_result = rouge.compute(predictions=hypothesis, references=gpt_gen, use_stemmer=True,use_aggregator=False)
                    candidates=[]
                    for hpy,r1,r2,rl in zip(hypothesis,rouge_result["rouge1"],rouge_result["rouge2"],rouge_result["rougeLsum"]):
                        score  = (r1 + r2 + rl) / 3
                        avg_score.append(score)
                        candidates.append([sent_tokenize(hpy),score])
                    output = {
                        "article": sent_tokenize(samples[idx]['document']), 
                        "abstract": sent_tokenize(samples[idx]['summary']),
                        "candidates": candidates,
                    }
                    with open("gen_cand_AWR_cnndm_origin/train/"+str(full_index)+".json", "w") as f:
                        json.dump(output, f,indent=3)
                    idx+=1
                    full_index+=1
                t.set_postfix(avg_rouge = f"{mean(avg_score)}")
                t.update(1)
            # hypothesis = hypothesis.replace("\n", " ")
            # fout.write(hypothesis + '\n')
            # fout.flush()
        # slines = []
        # gpts = []

        # slines.append(train['document'])
        # gpts.append(train['summary'])
        # count += 1
    # if slines != []:
    #     with torch.no_grad():
    #         dct = tokenizer.batch_encode_plus(slines, max_length=1024, return_tensors="pt", pad_to_max_length=True, truncation=True)
    #         summaries = model.generate(
    #             input_ids=dct["input_ids"].to(device),
    #             attention_mask=dct["attention_mask"].to(device),
    #             num_return_sequences=16, num_beam_groups=16, diversity_penalty=1.0, num_beams=16,
    #             max_length=max_length + 2,  # +2 from original because we start at step=1 and stop before max_length
    #             min_length=min_length + 1,  # +1 from original because we start at step=1
    #             no_repeat_ngram_size=3,
    #             length_penalty=2.0,
    #             early_stopping=True,
    #         )
    #         dec = [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=False) for g in summaries]
    #     idx=0
    #     for i in range(0,len(dec),16):
    #         output={}
    #         hypothesis = dec[i:i+16]
    #         gpt_gen = [gpts[idx]]*16
    #         rouge_result = rouge.compute(predictions=hypothesis, references=gpt_gen, use_stemmer=True,use_aggregator=False)
    #         candidates=[]
    #         for hpy,r1,r2,rl in zip(hypothesis,rouge_result["rouge1"],rouge_result["rouge2"],rouge_result["rougeLsum"]):
    #             score  = (r1 + r2 + rl) / 3
    #             candidates.append([sent_tokenize(hpy),score])
    #         output = {
    #             "article": sent_tokenize(slines[idx]), 
    #             "abstract": sent_tokenize(gpts[idx]),
    #             "candidates": candidates,
    #         }
    #         with open("gen_cand_AWR_cnndm_GPT/train/"+str(full_index)+".json", "w") as f:
    #             json.dump(output, f)
    #         idx+=1
    #         full_index+=1

def generate_summaries_xsum(args):
    rouge = evaluate.load("rouge",experiment_id="xsum")
    device = f"cuda:{args.gpuid}"
    mname = "google/pegasus-xsum"
    model = PegasusForConditionalGeneration.from_pretrained(mname,cache_dir="./local_cache")
    # x = torch.load("cache/06-28-21-03-1719579831_pegasus_origin_norm_xsum_WR/model_cur9.bin", map_location='cpu')
    # x = torch.load("cache/08-01-21-32-1722519128_pegasus_gpt_xsum_only_SCAN/model_generation.bin", map_location='cpu')
    x = torch.load("cache/08-01-20-02-1722513762_pegasus_xsum_origin_only_SCAN/model_generation.bin", map_location='cpu')
    new_x = {}
    for key, value in x.items():
        if "final_logits_bias" in key or "lm_head.weight" in key:
            new_x[key.replace('model.', '')] =  value 
        else:
            new_x[key.replace('model.model.', 'model.')] =  value 
    model.load_state_dict(new_x)
    model.eval()
    model.to(device)
    tokenizer = PegasusTokenizer.from_pretrained(mname)
    max_length = 62
    min_length = 11
    length_penalty=0.6
    count = 0
    # dataset = load_from_disk("/mnt/nas4/m11115088/WordRank/xsum/xsum_origin_norm_dataset_format")
    collate_fn = partial(collate_mp_brio, pad_token_id=tokenizer.pad_token_id, is_test=True)
    dataset = BrioDataset(f"train", mname,dataset=args.dataset, is_test=True,
                         max_len=512, total_len=512, is_pegasus=True)
    gen_dataloader = DataLoader(dataset, batch_size=8, shuffle=False, num_workers=12, collate_fn=collate_fn)
    # with open(args.tgt_dir, 'w') as fout:
    # with open(args.src_dir) as source, open(args.tgt_dir, 'w') as fout:
        # sline = source.readline().strip().lower()
    full_index = 0
    avg_score=[]
    with torch.no_grad():
        with tqdm(gen_dataloader, ncols=100) as t:
            for (i, batch) in enumerate(gen_dataloader):
                to_cuda(batch, args.gpuid)
                samples = batch["data"]
                # dct = tokenizer.batch_encode_plus(train['document'], max_length=1024, return_tensors="pt", pad_to_max_length=True, truncation=True)
                input_mask = batch["src_input_ids"] != tokenizer.pad_token_id
                summaries = model.generate(
                    input_ids=batch["src_input_ids"],
                    attention_mask=input_mask.to(device),
                    num_return_sequences=16, num_beam_groups=16, diversity_penalty=1.0, num_beams=16,
                    max_length=max_length + 2,  # +2 from original because we start at step=1 and stop before max_length
                    min_length=min_length + 1,  # +1 from original because we start at step=1
                    no_repeat_ngram_size=3,
                    length_penalty=length_penalty,
                    early_stopping=True,
                    src_word_position=batch["src_word_position"]
                )
                dec = [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=False) for g in summaries]
            # for hypothesis,gpt_gen in zip(dec,gpts):
                idx=0
                for i in range(0,len(dec),16):
                    output={}
                    hypothesis = dec[i:i+16]
                    gpt_gen = [samples[idx]["summary"]]*16
                    rouge_result = rouge.compute(predictions=hypothesis, references=gpt_gen, use_stemmer=True,use_aggregator=False)
                    candidates=[]
                    for hpy,r1,r2,rl in zip(hypothesis,rouge_result["rouge1"],rouge_result["rouge2"],rouge_result["rougeLsum"]):
                        score  = (r1 + r2 + rl) / 3
                        avg_score.append(score)
                        candidates.append([sent_tokenize(hpy),score])
                    output = {
                        "article": sent_tokenize(samples[idx]['document']), 
                        "abstract": sent_tokenize(samples[idx]['summary']),
                        "candidates": candidates,
                    }
                    with open("gen_cand_AWR_xsum_origin/train/"+str(full_index)+".json", "w") as f:
                        json.dump(output, f,indent=3)
                    idx+=1
                    full_index+=1
                t.set_postfix(avg_rouge = f"{mean(avg_score)}")
                t.update(1)
    # for train in tqdm(dataset['train']):
    # # for sline in source:
    #     # if count % 100 == 0:
    #     #     print(count, flush=True)
    #     if count % bsz == 0 and count!=0:
    #         with torch.no_grad():
    #             dct = tokenizer.batch_encode_plus(slines, max_length=512, return_tensors="pt", pad_to_max_length=True, truncation=True)
    #             summaries = model.generate(
    #                 input_ids=dct["input_ids"].to(device),
    #                 attention_mask=dct["attention_mask"].to(device),
    #                 num_return_sequences=16, num_beam_groups=16, diversity_penalty=1.0, num_beams=16,
    #                 max_length=max_length + 2,  # +2 from original because we start at step=1 and stop before max_length
    #                 min_length=min_length + 1,  # +1 from original because we start at step=1
    #                 no_repeat_ngram_size=3,
    #                 length_penalty=length_penalty,
    #                 early_stopping=True,
    #             )
    #             dec = [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=False) for g in summaries]
    #         idx=0
    #         for i in range(0,len(dec),16):
    #             output={}
    #             hypothesis = dec[i:i+16]
    #             gpt_gen = [gpts[idx]]*16
    #             rouge_result = rouge.compute(predictions=hypothesis, references=gpt_gen, use_stemmer=True,use_aggregator=False)
    #             candidates=[]
    #             for hpy,r1,r2,rl in zip(hypothesis,rouge_result["rouge1"],rouge_result["rouge2"],rouge_result["rougeLsum"]):
    #                 score  = (r1 + r2 + rl) / 3
    #                 candidates.append([sent_tokenize(hpy),score])
    #             output = {
    #                 "article": sent_tokenize(slines[idx]), 
    #                 "abstract": sent_tokenize(gpts[idx]),
    #                 "candidates": candidates,
    #             }
    #             with open("gen_cand_WR_xsum_origin/train/"+str(full_index)+".json", "w") as f:
    #                 json.dump(output, f,indent=3)
    #             idx+=1
    #             full_index+=1

    #         slines = []
    #         gpts = []

    #     slines.append(train['document'])
    #     gpts.append(train['summary'])
    #     count += 1
    # if slines != []:
    #     with torch.no_grad():
    #         dct = tokenizer.batch_encode_plus(slines, max_length=512, return_tensors="pt", pad_to_max_length=True, truncation=True)
    #         summaries = model.generate(
    #             input_ids=dct["input_ids"].to(device),
    #             attention_mask=dct["attention_mask"].to(device),
    #             num_return_sequences=16, num_beam_groups=16, diversity_penalty=1.0, num_beams=16,
    #             max_length=max_length + 2,  # +2 from original because we start at step=1 and stop before max_length
    #             min_length=min_length + 1,  # +1 from original because we start at step=1
    #             no_repeat_ngram_size=3,
    #             length_penalty=length_penalty,
    #             early_stopping=True,
    #         )
    #         dec = [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=False) for g in summaries]
    #     idx=0
    #     for i in range(0,len(dec),16):
    #         output={}
    #         hypothesis = dec[i:i+16]
    #         gpt_gen = [gpts[idx]]*16
    #         rouge_result = rouge.compute(predictions=hypothesis, references=gpt_gen, use_stemmer=True,use_aggregator=False)
    #         candidates=[]
    #         for hpy,r1,r2,rl in zip(hypothesis,rouge_result["rouge1"],rouge_result["rouge2"],rouge_result["rougeLsum"]):
    #             score  = (r1 + r2 + rl) / 3
    #             candidates.append([sent_tokenize(hpy),score])
    #         output = {
    #             "article": sent_tokenize(slines[idx]), 
    #             "abstract": sent_tokenize(gpts[idx]),
    #             "candidates": candidates,
    #         }
    #         with open("gen_cand_WR_xsum_origin/train/"+str(full_index)+".json", "w") as f:
    #             json.dump(output, f)
    #         idx+=1
    #         full_index+=1


# python gen_candidate.py --gpuid 0 --dataset xsum
# python gen_candidate.py --gpuid 3 --dataset cnndm
if __name__ ==  "__main__":
    parser = argparse.ArgumentParser(description='Parameters')
    parser.add_argument("--gpuid", type=int, default=0, help="gpu id")
    parser.add_argument("--src_dir", type=str, help="source file")
    parser.add_argument("--tgt_dir", type=str, help="target file")
    parser.add_argument("--dataset", type=str, default="cnndm", help="dataset")
    args = parser.parse_args()
    if args.dataset == "cnndm":
        generate_summaries_cnndm(args)
    elif args.dataset == "xsum":
        generate_summaries_xsum(args)