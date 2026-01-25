import os
import time

ip = os.environ['CDSW_IP_ADDRESS']
port = os.environ['CDSW_APP_PORT']
ray_dir = '/home/cdsw/ray'

with open(f'{ray_dir}/ray_current_cluster') as f:
    ray_head_addr = f.read()

cmd = [
    'source .venv/bin/activate',
    f'ray start --address={ray_head_addr}',
    f'fastapi run --host 127.0.0.1 --port {port} main.py',
]
cmd = f"bash -c '{' && '.join(cmd)}'"
print('Running', cmd)
os.system(cmd)

while True:
    time.sleep(60)