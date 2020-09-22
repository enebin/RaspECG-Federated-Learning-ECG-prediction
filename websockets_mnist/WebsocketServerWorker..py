import torch
import os
import sys
import logging
import syft as sy


print(sys.path)

hook = sy.TorchHook(torch)

from syft.workers.websocket_server import WebsocketServerWorker

local_worker = WebsocketServerWorker(
                            host="localhost",
                            hook=hook,
                            id=0,
                            port=8182)

local_worker.start()  # Might need to interrupt with `CTRL-C` or some other means

local_worker.list_objects()
local_worker.objects_count()
local_worker.host
local_worker.port