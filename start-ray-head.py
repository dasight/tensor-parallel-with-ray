import os
import time

ip = os.environ['CDSW_IP_ADDRESS']
port = os.environ['CDSW_APP_PORT']
cmd = [
    'rm -f /home/cdsw/ray/ray_current_cluster',
    'source .venv/bin/activate',
    f'ray start --head --port=6379 --dashboard-host=127.0.0.1 --dashboard-port={port}',
]
cmd = f"bash -c '{' && '.join(cmd)}'"
print('Running', cmd)
os.system(cmd)
os.system('cp -f /tmp/ray/ray_current_cluster /home/cdsw/ray')

print(f'Ray Head: {ip}:6379')
print(f'Ray Dashboard: 127.0.01:{port}')

while True:
    time.sleep(60)
    