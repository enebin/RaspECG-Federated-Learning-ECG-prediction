import torch
import syft
from syft.workers.websocket_server import WebsocketServerWorker
from syft.workers.websocket_client import WebsocketClientWorker

hook = syft.TorchHook(torch)
local_worker = WebsocketServerWorker(
                            host='localhost',
                            hook=hook,
                            id=0,
                            port=8181,
                            log_msgs=True,
                            verbose=True)


hook = syft.TorchHook(torch, local_worker=local_worker)
remote_client = WebsocketClientWorker(
                            host='localhost',
                            hook=hook,
                            id=2,
                            port=8182)


hook.local_worker.add_worker(remote_client)

x = syft.FixedPrecisionTensor([1,3,5,7,9]).share(remote_client)
x2 = syft.FixedPrecisionTensor([2,4,6,8,10]).share(remote_client)

y = x + x2 + x
y.get()