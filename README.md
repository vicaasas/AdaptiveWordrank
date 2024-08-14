# Adaptive-WordRank: An Adaptive Word Ranking Approach for Enhanced Document Summaries

## 1. Environments

```
- python (3.9)
```

## 2. Dependencies

```
- compare_mt==0.2.10
- datasets==2.20.0
- nltk==3.6.5
- numpy==2.0.0
- torch==2.3.1
- tqdm==4.66.4
- transformers==4.20.1
```

## 3. Dataset

Please download each dataset from the provided

https://drive.google.com/drive/folders/1AXrsdX7OJTDFrpYleMtIEM2E8O3iJot0?usp=drive_link

and create a folder named 'Dataset' to store them.


## 4. Checkpoint

|          | Checkpoints  |
|----------|---------|
| CNNDM_GPT    | [vickt/LLM_Teached_BART_CNNDM](https://huggingface.co/vickt/LLM_Teached_BART_CNNDM) <br> [vickt/LLM_Teached_PEGASUS_CNNDM_2](https://huggingface.co/vickt/LLM_Teached_PEGASUS_CNNDM_2)
| XSUM_GPT    | [GlycerinLOL/LLM_Teached_Bart_100k](https://huggingface.co/GlycerinLOL/LLM_Teached_Bart_100k)<br> [GlycerinLOL/LLM_Teached_Pegasus_100k](https://huggingface.co/GlycerinLOL/LLM_Teached_Pegasus_100k)
| CNNDM_Human    | [facebook/bart-large-cnn](https://huggingface.co/facebook/bart-large-cnn) <br> [google/pegasus-cnn_dailymail](google/pegasus-cnn_dailymail)
| XSUM_Human    | [https://huggingface.co/facebook/bart-large-xsum](facebook/bart-large-xsum)<br> [google/pegasus-xsum](https://huggingface.co/google/pegasus-xsum)


## 5. Training

```bash
>> python main.py --cuda --gpuid [single gpu id] --config [name of the config (cnndm/xsum)]  -l --name [name of current run] 
>> python main_pegasus.py --cuda --gpuid [single gpu id] --config [name of the config (cnndm/xsum)]  -l --name [name of current run] 
```
## 6. Train from checkpoint
```bash
>> python main.py --cuda --gpuid [single gpu id] --config [name of the config (cnndm/xsum)] --model_pt [model path] -l --name [name of current run] 
>> python main_pegasus.py --cuda --gpuid [single gpu id] --config [name of the config (cnndm/xsum)] --model_pt [model path] -l --name [name of current run] 
```

## 7. Evaluate
```bash
>> python main.py --cuda --gpuid [single gpu id] --config [name of the config (cnndm/xsum)] -e --model_pt [model path]  -r 
```

### New Feature update 9/1
During inference, you can use the --auto_calculate_SCAN_threshold parameter to automate the calculation of the keyword threshold
```bash
>> python main.py --cuda --gpuid [single gpu id] --config [name of the config (cnndm/xsum)] -e --model_pt [model path]  -r --auto_calculate_SCAN_threshold
```