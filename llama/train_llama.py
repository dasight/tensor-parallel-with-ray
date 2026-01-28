from datetime import timedelta
import os, sys
sys.path += ['/home/cdsw/.venv/lib/python3.12/site-packages', '/home/cdsw/llama']

import safetensors
import ray
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as mp
import torch.distributed as dist
from transformers import AutoTokenizer
from llama import LlamaConfig, LlamaModel

tp_size = 2
model_path = "/home/cdsw/models/Llama-3.2-1B-Instruct"
data_path = "/home/cdsw/llama/tiny-shakespeare.txt"
ray_dir = '/home/cdsw/ray'

with open(f'{ray_dir}/ray_current_cluster') as f:
    ray_head_addr = f.read().split(':')[0]
ray.init(f'ray://{ray_head_addr}:10001', runtime_env={"working_dir": "/home/cdsw/llama"})


class DataLoaderLite:
    def __init__(self, model_path, filename, BATCH_SIZE, SEQ_LEN):
        self.BATCH_SIZE = BATCH_SIZE
        self.SEQ_LEN = SEQ_LEN

        with open(filename) as f:
            text = f.read()
        tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
        tokens = tokenizer(text, return_tensors='pt')
        self.tokens = tokens.input_ids.squeeze()
        self.position = 0
        print(f'Loaded {len(self.tokens)} tokens.')
        print(f'1 epoch = {len(self.tokens) // (BATCH_SIZE * SEQ_LEN)} batches.')

    def next_batch(self):
        BATCH_SIZE, SEQ_LEN = self.BATCH_SIZE, self.SEQ_LEN
        buf = self.tokens[self.position : self.position+BATCH_SIZE*SEQ_LEN+1]
        x = (buf[:-1]).view(BATCH_SIZE, SEQ_LEN) # inputs
        y = (buf[1:]).view(BATCH_SIZE, SEQ_LEN) # targets
        self.position += BATCH_SIZE * SEQ_LEN
        
        # if loading the next batch would be out of bounds
        if self.position + (BATCH_SIZE * SEQ_LEN + 1) > len(self.tokens):
            self.position = 0
        return x, y


def load_embed(model, safetensors_file):
    with safetensors.safe_open(safetensors_file, framework="pt") as f:
        embed_weight = f.get_tensor(f'model.embed_tokens.weight').to(torch.float)
        embed_weight.requires_grad = False
        model.embed_tokens.weight = nn.Parameter(embed_weight)
        model.embed_tokens.requires_grad = False
        model.lm_head.weight = nn.Parameter(embed_weight)
        model.lm_head.requires_grad = False


def init_dist_group(tp_rank, tp_size):
    if tp_rank == 0:
        print(f"Start to Initialize TP Rank {tp_rank} ...")
        ip = ray.util.get_node_ip_address()
        with open(f'{ray_dir}/pytorch_master', 'w') as f:
            f.write(ip)

        os.environ['MASTER_ADDR'] = ip
        os.environ['MASTER_PORT'] = "23333"
        dist.init_process_group(backend='gloo', init_method='env://', world_size=tp_size, rank=tp_rank)
        print(f"TP Rank {tp_rank} initialized.")
    else:
        print(f"Start to Initialize TP Rank {tp_rank} ...")
        while True:
            with open(f'{ray_dir}/pytorch_master',) as f:
                ip=f.read().strip()
            if len(ip) > 0:
                print(f"TP Rank {tp_rank} initializing with {ip}.")
            else:
                continue
            
            os.environ['MASTER_ADDR'] = ip
            os.environ['MASTER_PORT'] = "23333"
            timeout = timedelta(seconds=10)
            dist.init_process_group(backend='gloo', init_method='env://', world_size=tp_size, rank=tp_rank, timeout=timeout)
            if dist.is_initialized():
                print(f"TP Rank {tp_rank} initialized.")
                break
            else:
                print(f"TP Rank {tp_rank} timed out during initialization. Retry ...")


@ray.remote(num_gpus=1)
def train(tp_rank, tp_size, device='cpu'):
    init_dist_group(tp_rank, tp_size)
    config = LlamaConfig(num_hidden_layers=2)
    model = LlamaModel(config, device=device, tp_rank=tp_rank, tp_size=tp_size)
    load_embed(model, f"{model_path}/model.safetensors")
    model.to(device)

    B, T = 4, 512
    data_loader = DataLoaderLite(model_path, data_path, B, T)
    optim = torch.optim.AdamW(model.parameters(), lr=3e-4, betas=(0.9, 0.95), eps=1e-8, fused=True)

    for step in range(20):
        optim.zero_grad()

        x, y = data_loader.next_batch()
        x, y = x.cuda(), y.cuda()
        logits = model(x)
        loss = F.cross_entropy(logits.view(B*T, -1), y.view(-1))        
        loss.backward()
        optim.step()
        print(f"tp_rank {tp_rank} step {step} loss {loss.item()}")


if __name__ == "__main__":
    tp_size = 2
    workers = [train.remote(tp_rank, tp_size, 'cuda') for tp_rank in range(tp_size)]
    ray.get(workers)
