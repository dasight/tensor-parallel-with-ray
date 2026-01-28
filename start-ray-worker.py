import os, sys
sys.path += ['/home/cdsw/.venv/lib/python3.12/site-packages', '/home/cdsw/llama']

import asyncio
import uvicorn
from fastapi import FastAPI

ip = os.environ['CDSW_IP_ADDRESS']
port = os.environ['CDSW_APP_PORT']
ray_dir = '/home/cdsw/ray'
app = FastAPI()

@app.get("/")
def read_root():
    return {"Hello": "World"}

with open(f'{ray_dir}/ray_current_cluster') as f:
    ray_head_addr = f.read()

cmd = [
    'source .venv/bin/activate',
    f'ray start --address={ray_head_addr}',
    # f'fastapi run --host 127.0.0.1 --port {port} main.py',
]
cmd = f"bash -c '{' && '.join(cmd)}'"
print('Running', cmd)
os.system(cmd)

config = uvicorn.Config(app, host="127.0.0.1", port=int(port), reload=False)
server = uvicorn.Server(config)
await server.serve()
