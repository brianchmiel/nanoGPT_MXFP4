
# nanoGPT with MXFP4 quantization 

The code is based on https://github.com/karpathy/nanoGPT/tree/master 

## install

```
pip install torch numpy transformers datasets tiktoken wandb tqdm
cd mxfp4_kernel
pip install .
```

Dependencies:

- [pytorch](https://pytorch.org) <3
- [numpy](https://numpy.org/install/) <3
-  `transformers` for huggingface transformers <3 (to load GPT-2 checkpoints)
-  `datasets` for huggingface datasets <3 (if you want to download + preprocess OpenWebText)
-  `tiktoken` for OpenAI's fast BPE code <3
-  `wandb` for optional logging <3
-  `tqdm` for progress bars <3



## data preparation

```sh
python data/openwebtext/prepare.py
```
This downloads and tokenizes the [OpenWebText](https://huggingface.co/datasets/openwebtext) dataset. It will create a `train.bin` and `val.bin`

## quick start

In order to run the baseline run:
```sh
torchrun --standalone --nproc_per_node=8 train.py config/train_gpt2_124m.py
```


In order to run the MXFP4 quantization model run:
```sh
torchrun --standalone --nproc_per_node=8 train.py config/train_gpt2_124m_mxfp4.py
```

| model | params | train loss | 
| ------| ------ | ---------- | 
| gpt2 | 124M         | 3.11  | 
| gpt2-mxfp4 | 124M  | 2.85  | 


In order to run a bigger model, simple add to the config file the following lines:

### GPT 350m
n_layer = 24
n_head = 16
n_embd = 1024


### GPT 774m
n_layer = 36
n_head = 20
n_embd = 1280


### GPT-1.5b

n_layer = 48
n_head = 25
n_embd = 1600

