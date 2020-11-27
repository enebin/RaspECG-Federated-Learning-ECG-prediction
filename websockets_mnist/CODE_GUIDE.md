# CODE GUIDE

마지막 수정 일자: README
상태: 📝게시판
작성일시: 2020년 11월 2일 오전 11:53

## 시작에 앞서

 이 프로젝트는 기본적으로 2개의 python 파일 (run_websocket_server(0.2.3).py, run_websocket_client(0.2.3).py**)**로 구성되어있습니다. 두 파일은 각각 라즈베리파이와 컴퓨터에 다운로드되어있어야 하며 서버를 구성 및 구동하는 역할을 합니다. 역할 및 사용법에 대한 자세한 설명은 '[이 곳](https://www.notion.so/README-76afc5f599944e26929750dfd104106b)'을 참조해 주십시오. 본 문서에서는 향후 유지보수를 위해 파이썬 파일의 코드만을 간략하게 설명합니다. 

# run_websocket_server(0.2.3).py

### 역할

✔️ 라즈베리파이에서 실행하며  중앙 서버로부터 커맨드를 수신하는 서버를 구동합니다.

✔️ 전처리된 데이터셋을 토대로 수신한 커맨드에 따라 정해진 동작을 수행합니다.

### 코드

1. **def : start_proc**

```python
def start_proc(participant, kwargs):  # pragma: no cover
    """ helper function for spinning up a websocket participant """

    def target():
        server = participant(**kwargs)
        server.start()

    p = Process(target=target)
    p.start()
    return p
```

- 매개변수 participant 에는 syft 라이브러리의 WebsocketServerWorker 함수가 주로 들어갑니다.
- 매개변수 kwargs 에는 파일을 실행할 때 넣은 인자값이 들어갑니다.
- 두 개의 매개변수를 토대로 서버를 구동합니다. multiprocess 라이브러리의 Processs 함수를 이용한 쓰레딩을 이용합니다.

**2. part : parser** 

```python
parser = argparse.ArgumentParser(description="Run websocket server worker.")
parser.add_argument(
    "--port", "-p", type=int, help="port number of the websocket server worker, e.g. --port 8777"
)
parser.add_argument("--host", type=str, default="localhost", help="host for the connection")
parser.add_argument(
    "--id", type=str, help="name (id) of the websocket server worker, e.g. --id alice"
)
parser.add_argument(
    "--verbose",
    "-v",
    action="store_true",
    help="if set, websocket server worker will be started in verbose mode",
)

args = parser.parse_args()
```

- 파일을 실행할 때 가능한 옵션(argument)에 대해 정의합니다.
- port, host(아이피 주소), id, verbose 4개의 인자가 존재합니다.

**3. part : main**

```python
kwargs = {
    "id": args.id,
    "host": args.host,
    "port": args.port,
    "hook": hook,
    "verbose": args.verbose,
}
server = start_proc(WebsocketServerWorker, kwargs)
```

- 실행시 받은 인자를 토대로 [start_proc]()을 실행합니다.

# FL_ECG.py

### 역할

✔️ 중앙 장치(데스크톱, 노트북)에 존재하며 실행 시 라즈베리파이가 구동중인 서버에 연결합니다.

✔️ 각 라즈베리파이에 FL을 위한 커맨드를 송신합니다. 

✔️ 라즈베리파이로부터 수신한 모델을 합산, 처리한 후 업데이트된 모델을 회신합니다. 

 

### 코드

1. **class : 네트워크**

```python
import torch.nn as nn
import torch.nn.functional as f

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4 * 4 * 50, 500)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
        x = f.relu(self.conv1(x))
        x = f.max_pool2d(x, 2, 2)
        x = f.relu(self.conv2(x))
        x = f.max_pool2d(x, 2, 2)
        x = x.view(-1, 4 * 4 * 50)
        x = f.relu(self.fc1(x))
        x = self.fc2(x)
        return f.log_softmax(x, dim=1)
```

- 머신러닝 모델 네트워크를 설정합니다.
- nn 라이브러리에서 추상화된 Module 클래스를 상속합니다.
- nn 라이브러리의 functional 에서 relu, pool과 같은 레이어 프리셋을 사용할 수 있습니다.

**2. def : train_on_batches**

[매개변수](https://www.notion.so/eb6a24eeb8f74be79a5990af8ea2960d)

```python
import torch.optim as optim

def train_on_batches(worker, batches, model_in, device, lr):
    """Train the model on the worker on the provided batches
    Args:
        worker(syft.workers.BaseWorker): worker on which the
        training will be executed
        batches: batches of data of this worker
        model_in: machine learning model, training will be done on a copy
        device (torch.device): where to run the training
        lr: learning rate of the training steps
    Returns:
        model, loss: obtained model and loss after training
    """
    model = model_in.copy()
    optimizer = optim.SGD(model.parameters(), lr=lr)  # TODO momentum is not supported at the moment

    model.train()
    model.send(worker)
    loss_local = False
```

- optimizer에 사용하고자 하는 옵티마이저를 선택하고 모델을 연결 한 후 학습률을 정합니다.
- train() 함수를 이용하여 모델을 학습모드로 전환합니다.

```python
 for batch_idx, (data, target) in enumerate(batches):
        loss_local = False
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = f.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % LOG_INTERVAL == 0:
            loss = loss.get()  # <-- NEW: get the loss back
            loss_local = True
            logger.debug(
                "Train Worker {}: [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                    worker.id,
                    batch_idx,
                    len(batches),
                    100.0 * batch_idx / len(batches),
                    loss.item(),
                )
            )

    if not loss_local:
        loss = loss.get()  # <-- NEW: get the loss back
    model.get()  # <-- NEW: get the model back
    return model, loss
```

- 받아온 배치 데이터셋을 디바이스로 보냅니다. 데이터셋은 데이터와 타겟으로 이루어져 있습니다.
- 

**3. def : get_next_batches**

[매개변수](https://www.notion.so/c813c3e10fe64a1cb751913eaf791ece)

```python
def get_next_batches(fdataloader: sy.FederatedDataLoader, nr_batches: int):
    """retrieve next nr_batches of the federated data loader and group
    the batches by worker
    Args:
        fdataloader (sy.FederatedDataLoader): federated data loader
        over which the function will iterate
        nr_batches (int): number of batches (per worker) to retrieve
    Returns:
        Dict[syft.workers.BaseWorker, List[batches]]
    """
    batches = {}
    for worker_id in fdataloader.workers:
        worker = fdataloader.federated_dataset.datasets[worker_id].location
        batches[worker] = []
    try:
        for i in range(nr_batches):
            next_batches = next(fdataloader)
            for worker in next_batches:
                batches[worker].append(next_batches[worker])
    except StopIteration:
        pass
    return batches
```

- 

**4. def : train** 

[매개변수](https://www.notion.so/53f0bf10a6304d9097270f2dd6931e30)

```python
def train(
    model, device, federated_train_loader, lr, federate_after_n_batches, abort_after_one=False
):
    model.train()

    nr_batches = federate_after_n_batches

    models = {}
    loss_values = {}

    iter(federated_train_loader)  # initialize iterators
    batches = get_next_batches(federated_train_loader, nr_batches)
    counter = 0

    while True:
        logger.debug(
            "Starting training round, batches [{}, {}]".format(counter, counter + nr_batches)
        )
        data_for_all_workers = True
        for worker in batches:
            curr_batches = batches[worker]
            if curr_batches:
                models[worker], loss_values[worker] = train_on_batches(
                    worker, curr_batches, model, device, lr
                )
            else:
                data_for_all_workers = False
        counter += nr_batches
        if not data_for_all_workers:
            logger.debug("At least one worker ran out of data, stopping.")
            break

        model = utils.federated_avg(models)
        batches = get_next_batches(federated_train_loader, nr_batches)
        if abort_after_one:
            break
    return model
```

- 모델을 트레이닝하는 부분입니다. model.train() 에 사용되는 트레이닝 메서드와는 별개입니다.
- batches 변수에는 get_next_batches 함수를 이용하여 미리 정한 배치 수 만큼의 데이터셋을 받아옵니다.
- 

**5. def : test**

[매개변수](https://www.notion.so/f73e2c70b084411584664abb93c0cedf)

```python
def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += f.nll_loss(output, target, reduction="sum").item()  # sum up batch loss
            pred = output.argmax(1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    logger.debug("\n")
    accuracy = 100.0 * correct / len(test_loader.dataset)
    logger.info(
        "Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n".format(
            test_loss, correct, len(test_loader.dataset), accuracy
        )
    )
```

**6. def : define_and_get_arguments**

[매개변수](https://www.notion.so/ee9ca83e4fc54910ab46cafdade2ce86)

```python
def define_and_get_arguments(args=sys.argv[1:]):
    parser = argparse.ArgumentParser(
        description="Run federated learning using websocket client workers."
    )
    parser.add_argument("--batch_size", type=int, default=64, help="batch size of the training")
    parser.add_argument(
        "--test_batch_size", type=int, default=1000, help="batch size used for the test data"
    )
    parser.add_argument("--epochs", type=int, default=2, help="number of epochs to train")
    parser.add_argument(
        "--federate_after_n_batches",
        type=int,
        default=50,        help="number of training steps performed on each remote worker " "before averaging",
    )
    parser.add_argument("--lr", type=float, default=0.01, help="learning rate")
    parser.add_argument("--cuda", action="store_true", help="use cuda")
    parser.add_argument("--seed", type=int, default=1, help="seed used for randomization")
    parser.add_argument("--save_model", action="store_true", help="if set, model will be saved")
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="if set, websocket client workers will " "be started in verbose mode",
    )
    parser.add_argument(
        "--use_virtual", action="store_true", help="if set, virtual workers will be used"
    )

    args = parser.parse_args(args=args)
    return args
```

**7. def : main**

*매개변수 없음*

```python
def main():
    args = define_and_get_arguments()

    hook = sy.TorchHook(torch)

    # 가상작업자(시뮬레이션) 사용시 이곳으로 분기
    if args.use_virtual:
        alice = VirtualWorker(id="alice", hook=hook, verbose=args.verbose)
        bob = VirtualWorker(id="bob", hook=hook, verbose=args.verbose)
        charlie = VirtualWorker(id="charlie", hook=hook, verbose=args.verbose)
    # 웹소켓작업자 사용시 이곳으로 분기
    else:
        a_kwargs_websocket = {"host": "192.168.0.52", "hook": hook}
        b_kwargs_websocket = {"host": "192.168.0.53", "hook": hook}
        c_kwargs_websocket = {"host": "192.168.0.54", "hook": hook}

        baseport = 10002
        alice = WebsocketClientWorker(id="alice", port=baseport, **a_kwargs_websocket)
        bob = WebsocketClientWorker(id="bob", port=baseport, **b_kwargs_websocket)
        charlie = WebsocketClientWorker(id="charlie", port=baseport, **c_kwargs_websocket)

		# 객체를 리스트로 묶음
    workers = [alice, bob, charlie]

		# 쿠다 사용 여부
    use_cuda = args.cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    kwargs = {"num_workers": 1, "pin_memory": True} if use_cuda else {}

		# 랜덤 시드 설정
    torch.manual_seed(args.seed)
```

- [define_and_get_arguments]()() 를 이용하여 실행 옵션을 받아옵니다.
- use_virtual 옵션을 실행했을 경우 웹소켓을 이용하지 않고 가상 워커로 시뮬레이션 합니다. 실제로 라즈베리 파이에 연결하여 실행하기 전 **가상 워커 시뮬레이션을 이용해 테스트 시간을 단축할 수 있습니다**.
- use_virtual을 따로 설정하지 않았을 경우 웹소켓으로 동작합니다. 이 경우 kwargs_websocket에는  라즈베리파이의 IP, hook이 주어집니다. 그 후, WebsocketClientWorker를 이용하여 각 ID에 워커 객체를 할당합니다.
- 딱히 수정할 일이 없는 파트입니다.

```python
  federated_train_loader = sy.FederatedDataLoader(
        datasets.MNIST(
            "../data",
            train=True,
            download=True,
            transform=transforms.Compose(
                [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
            ),
        ).federate(tuple(workers)),
        batch_size=args.batch_size,
        shuffle=True,
        iter_per_worker=True,
        **kwargs,
    )

    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST(
            "../data",
            train=False,
            transform=transforms.Compose(
                [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
            ),
        ),
        batch_size=args.test_batch_size,
        shuffle=True,
        **kwargs,
    )

    model = Net().to(device)

    for epoch in range(1, args.epochs + 1):
				# output : 2020-11-05 15:07:04,953 INFO run_websocket_client(0.2.3).py(l:268) - Starting epoch 1/2
        logger.info("Starting epoch %s/%s", epoch, args.epochs)
        model = train(model, device, federated_train_loader, args.lr, args.federate_after_n_batches)
        test(model, device, test_loader)

    if args.save_model:
        torch.save(model.state_dict(), "mnist_cnn.pt")
```

- federated_train_loader는 불러온 datasets을 Federated Learning이 가능한 객체로 만듭니다.
- [FederateDataloader]()는 Federate Learning을 실행하기 위한 명령어들이 모여있는 객체입니다. 반복자로 변환하여 사용합니다.
- 그 전에 각 워커에게 데이터를 분배해야 합니다. 이는 datasets 라이브러리의  federated(tuple(workers))를 이용합니다.
- 이제 args.epochs에 명시된 수만큼 학습을 반복합니다. 이 epoch는 중앙 서버에서 모델을 집계하는 epoch입니다. 기본값은 2입니다.

**8. part : main**

```python
if __name__ == "__main__":
    FORMAT = "%(asctime)s %(levelname)s %(filename)s(l:%(lineno)d) - %(message)s"
    LOG_LEVEL = logging.DEBUG
    logging.basicConfig(format=FORMAT, level=LOG_LEVEL)

    websockets_logger = logging.getLogger("websockets")
    websockets_logger.setLevel(logging.DEBUG)
    websockets_logger.addHandler(logging.StreamHandler())

    main()
```

- 별건 없고 로깅 메시지 설정과 메인 함수 진입하는 두가지 파트로 나뉩니다.
- getLogger를 이용해 websockets라는 로거를 생성합니다. setLevel을 이용해 DEBUG 레벨 위의 레벨은 모두 프린트합니다. (로거 레벨은 DEBUG, INFO,  WARNING, ERROR, CRITICAL 5개가 존재합니다.)
- addHandler를 이용해 콘솔창에 로그가 출력되게끔 설정합니다. 파일,DB,소켓,큐 등을 통해 출력하도록 설정할 수도 있습니다.
- 로깅에 대해 참고할만한 블로그 글 ⬇️

[파이썬 로깅의 모든것](https://hamait.tistory.com/880)

- 이후 [메인함수에]() 진입합니다.

### 참고할 함수

- **FederatedDataLoader**

    ```python
    class FederatedDataLoader(object):
        """
        Data loader. Combines a dataset and a sampler, and provides
        single or several iterators over the dataset.
        Arguments:
            federated_dataset (FederatedDataset): dataset from which to load the data.
            batch_size (int, optional): how many samples per batch to load
                (default: ``1``).
            shuffle (bool, optional): set to ``True`` to have the data reshuffled
                at every epoch (default: ``False``).
            collate_fn (callable, optional): merges a list of samples to form a mini-batch.
            drop_last (bool, optional): set to ``True`` to drop the last incomplete batch,
                if the dataset size is not divisible by the batch size. If ``False`` and
                the size of dataset is not divisible by the batch size, then the last batch
                will be smaller. (default: ``False``)
            num_iterators (int): number of workers from which to retrieve data in parallel.
                num_iterators <= len(federated_dataset.workers) - 1
                the effect is to retrieve num_iterators epochs of data but at each step data
                from num_iterators distinct workers is returned.
            iter_per_worker (bool): if set to true, __next__() will return a dictionary
                containing one batch per worker
        """

        __initialized = False

        def __init__(
            self,
            federated_dataset,
            batch_size=8,
            shuffle=False,
            num_iterators=1,
            drop_last=False,
            collate_fn=default_collate,
            iter_per_worker=False,
            **kwargs,
        ):
            if len(kwargs) > 0:
                options = ", ".join([f"{k}: {v}" for k, v in kwargs.items()])
                logging.warning(f"The following options are not supported: {options}")

            try:
                self.workers = federated_dataset.workers
            except AttributeError:
                raise Exception(
                    "Your dataset is not a FederatedDataset, please use "
                    "torch.utils.data.DataLoader instead."
                )

            self.federated_dataset = federated_dataset
            self.batch_size = batch_size
            self.drop_last = drop_last
            self.collate_fn = collate_fn
            self.iter_class = _DataLoaderOneWorkerIter if iter_per_worker else _DataLoaderIter

            # Build a batch sampler per worker
            self.batch_samplers = {}
            for worker in self.workers:
                data_range = range(len(federated_dataset[worker]))
                if shuffle:
                    sampler = RandomSampler(data_range)
                else:
                    sampler = SequentialSampler(data_range)
                batch_sampler = BatchSampler(sampler, batch_size, drop_last)
                self.batch_samplers[worker] = batch_sampler

            if iter_per_worker:
                self.num_iterators = len(self.workers)
            else:
                # You can't have more iterators than n - 1 workers, because you always
                # need a worker idle in the worker switch process made by iterators
                if len(self.workers) == 1:
                    self.num_iterators = 1
                else:
                    self.num_iterators = min(num_iterators, len(self.workers) - 1)

        def __iter__(self):
            self.iterators = []
            for idx in range(self.num_iterators):
                self.iterators.append(self.iter_class(self, worker_idx=idx))
            return self

        def __next__(self):
            if self.num_iterators > 1:
                batches = {}
                for iterator in self.iterators:
                    data, target = next(iterator)
                    batches[data.location] = (data, target)
                return batches
            else:
                iterator = self.iterators[0]
                data, target = next(iterator)
                return data, target

        def __len__(self):
            length = len(self.federated_dataset) / self.batch_size
            if self.drop_last:
                return int(length)
            else:
                return math.ceil(length)
    ```

    - .federate() 함수를 이용해 federated된 데이터셋을 입력으로 받습니다.
- **torch.device**

    CUDA Tensors : `.to` 메소드를 사용하여 Tensor를 어떠한 장치로도 옮길 수 있습니다.

    ```python
    # 이 코드는 CUDA가 사용 가능한 환경에서만 실행합니다.
    # ``torch.device`` 를 사용하여 tensor를 GPU 안팎으로 이동해보겠습니다.
    if torch.cuda.is_available(): 
    	device = torch.device("cuda")         # CUDA 장치 객체(device object)로 
    	y = torch.ones_like(x, device=device) # GPU 상에 직접적으로 tensor를 생성하거나
    	x = x.to(device)                      # ``.to("cuda")`` 를 사용하면 됩니다. 
    	z = x + y 
    	print(z) 
    	print(z.to("cpu", torch.double))      # ``.to`` 는 dtype도 함께 변경합니다!
    ```

### 해결할 일

- [x]  중앙장치가 아닌 워커가 소유한 데이터를 이용한 학습
- [ ]
