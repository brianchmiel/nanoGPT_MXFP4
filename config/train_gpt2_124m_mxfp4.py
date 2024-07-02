
wandb_log = True
wandb_project = 'owt'
wandb_run_name='gpt2-124M-MXFP4'

mxfp4_quant= True
data_dir = '/software/users/bchmiel/nanoGPT/data/openwebtext'
# these make the total batch size be ~0.5M
batch_size = 24
block_size = 1024
gradient_accumulation_steps = 20

# this makes total number of tokens be 30B
max_iters = 60000
lr_decay_iters = 60000

# eval stuff
eval_interval = 1000
eval_iters = 200
log_interval = 10

# weight decay
weight_decay = 1e-1
dtype = 'bfloat16' 