from datetime import timedelta
import time
import os, sys
# sys.path += ['/home/cdsw/.venv/lib/python3.12/site-packages', '/home/cdsw/llama']

import safetensors
import ray
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as mp
import torch.distributed as dist
from transformers import AutoTokenizer
from llama import LlamaConfig, LlamaModel

tp_size = 1
B, T = 1, 512
# model_path = "/home/cdsw/models/Llama-3.2-1B-Instruct"
model_path = "/home/cdsw/models/Llama-3.1-8B-Instruct"
data_path = "/home/cdsw/llama/tiny-shakespeare.txt"
ray_dir = '/home/cdsw/ray'

# with open(f'{ray_dir}/ray_current_cluster') as f:
#     ray_head_addr = f.read().split(':')[0]
# ray.init(f'ray://{ray_head_addr}:10001', runtime_env={"working_dir": "/home/cdsw/llama"})
ray.init()

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
        # model.embed_tokens.requires_grad = False
        model.lm_head.weight = nn.Parameter(embed_weight)
        # model.lm_head.requires_grad = False


@ray.remote(num_cpus=8, num_gpus=1)
class RayTrainer:
    def __init__(self, tp_rank, tp_size):
        self.tp_rank = tp_rank
        self.tp_size = tp_size

    def master_addr(self):
        return ray.util.get_node_ip_address()
    
    def init_dist_group(self, master_addr):
        os.environ['MASTER_ADDR'] = master_addr
        os.environ['MASTER_PORT'] = "23333"
        dist.init_process_group(
            backend='nccl', init_method='env://', 
            world_size=self.tp_size, rank=self.tp_rank
        )
        print(f"TP Rank {self.tp_rank} initialized.")

    def train(self, device):
        config = LlamaConfig(num_hidden_layers=2)
        model = LlamaModel(config, device=device, tp_rank=self.tp_rank, tp_size=self.tp_size)
        load_embed(model, f"{model_path}/model-00001-of-00004.safetensors")
        model.to(device)
    
        data_loader = DataLoaderLite(model_path, data_path, B, T)
        optim = torch.optim.AdamW(model.parameters(), lr=3e-4, betas=(0.9, 0.95), eps=1e-8, fused=True)
    
        for step in range(20):
            t0 = time.time()
            optim.zero_grad()
            
            x, y = data_loader.next_batch()
            x, y = x.to(device), y.to(device)
    
            with torch.autocast(device_type=device, dtype=torch.bfloat16):
                logits = model(x)
                loss = F.cross_entropy(logits.view(B*T, -1), y.view(-1))
            
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            optim.step()
            torch.cuda.synchronize()
    
            if self.tp_rank == 0:
                dt = (time.time() - t0) * 1000
                print(f"tp_rank {self.tp_rank} step {step}: loss {loss.item():.4f}, dt {dt:.2f}ms")


if __name__ == "__main__":
    workers = [RayTrainer.remote(tp_rank=tp_rank, tp_size=tp_size) for tp_rank in range(tp_size)]
    # master_addr = ray.get(workers[0].master_addr.remote())
    master_addr = '127.0.0.1'
    ray.get([w.init_dist_group.remote(master_addr) for w in workers])
    ray.get([w.train.remote('cuda') for w in workers])


# Llama 3-1b
# tp_size = 2, (B, T) = (4, 512), dt = 252.91ms
# tp_size = 4, (B, T) = (4, 1024), dt = 401.06ms

# Llama 3-8b
# tp_size = 1, (B, T) = (2, 512), dt = ms
# tp_size = 2, (B, T) = (4, 512), dt = 550.49ms
# tp_size = 4, (B, T) = (4, 1024), dt = 725.07ms
