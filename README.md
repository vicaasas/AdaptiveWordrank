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

## 3. Dataset and 4. Checkpoint

Please download each dataset and checkpoint from the provided 

https://drive.google.com/drive/folders/1AXrsdX7OJTDFrpYleMtIEM2E8O3iJot0?usp=drive_link

Create a folder named 'Dataset' to store the datasets, while the checkpoints can be placed in any desired location.
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