# If not using local runtimes, uncomment out the following snippets where necessary.

# ! rm -rf /content/PySyft
# ! git clone https://github.com/OpenMined/PySyft.git
# http://pytorch.org/
# from os import path
# from wheel.pep425tags import get_abbr_impl, get_impl_ver, get_abi_tag

# platform = '{}{}-{}'.format(get_abbr_impl(), get_impl_ver(), get_abi_tag())
# cuda_output = !ldconfig -p|grep cudart.so|sed -e 's/.*\.\([0-9]*\)\.\([0-9]*\)$/cu\1\2/'
# accelerator = cuda_output[0] if path.exists('/opt/bin/nvidia-smi') else 'cpu'

# !pip3 install https://download.pytorch.org/whl/cu100/torch-1.1.0-cp36-cp36m-linux_x86_64.whl
# !pip3 install https://download.pytorch.org/whl/cu100/torchvision-0.3.0-cp36-cp36m-linux_x86_64.whl

import torch

# !cd PySyft; pip3 install -r requirements.txt; pip3 install -r requirements_dev.txt; python3 setup.py install

import os
import sys

# module_path = '/content/PySyft'  # You want './PySyft' to be on your sys.path
# if module_path not in sys.path:
#    sys.path.append(module_path)

import syft
from syft.workers.websocket_server import WebsocketServerWorker

hook = syft.TorchHook(torch)

local_worker = WebsocketServerWorker(
                            host='localhost',
                            hook=hook,
                            id=0,
                            port=8182,
                            log_msgs=True,
                            verbose=True)

hook = syft.TorchHook(torch, local_worker=local_worker)

from syft.workers.websocket_client import WebsocketClientWorker

remote_client = WebsocketClientWorker(
                            host = 'localhost',
                            hook=hook,
                            id=2,
                            port=8181)

hook.local_worker.add_worker(remote_client)

x = syft.FixedPrecisionTensor([1,3,5,7,9]).share(remote_client)
x2 = syft.FixedPrecisionTensor([2,4,6,8,10]).share(remote_client)

y = x + x2 + x
y.get()