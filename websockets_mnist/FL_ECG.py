import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import sklearn.utils as skl
import scipy.signal as sci
import torch
import random
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as D

import syft as sy
from syft.workers.websocket_client import WebsocketClientWorker
from syft.workers.virtual import VirtualWorker
from syft.frameworks.torch.fl import utils

import logging
import argparse
import sys

logger = logging.getLogger(__name__)
LOG_INTERVAL = 25

random_seed = 1024
np.random.seed(random_seed)


# define the 1st architecture (from the paper)
class Net(nn.Module):
    def __init__(self, input_features, output_dim):
        super(Net, self).__init__()
        # 1-dimensional convolutional layer
        self.conv0 = nn.Conv1d(input_features, 128, output_dim, stride=1, padding=0)
        self.conv1 = nn.Conv1d(128, 128, output_dim, stride=1, padding=2)

        # max pooling layer
        self.pool1 = nn.MaxPool1d(5, 2)

        # fully-connected layer
        self.fc1 = nn.Linear(256, 32)
        self.fc2 = nn.Linear(32, output_dim)

        # softmax output
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        # add sequence of convolutional and max pooling layers

        inp = x.view(32, -1, 187)
        C = self.conv0(inp)

        # first conv layer
        C11 = self.conv0(inp)
        A11 = F.relu(C11)
        C12 = self.conv1(A11)
        S11 = torch.add(C12, C)
        M11 = self.pool1(S11)

        # second conv layer
        C21 = self.conv1(M11)
        A21 = F.relu(C21)
        C22 = self.conv1(A21)
        S21 = torch.add(C22, M11)
        M21 = self.pool1(S21)

        # third conv layer
        C31 = self.conv1(M21)
        A31 = F.relu(C31)
        C32 = self.conv1(A31)
        S31 = torch.add(C32, M21)
        M31 = self.pool1(S31)

        # fourth conv layer
        C41 = self.conv1(M31)
        A41 = F.relu(C41)
        C42 = self.conv1(A41)
        S41 = torch.add(C42, M31)
        M41 = self.pool1(S41)

        # last layer
        C51 = self.conv1(M41)
        A51 = F.relu(C51)
        C52 = self.conv1(A51)
        S51 = torch.add(C52, M41)
        M51 = self.pool1(S51)

        # flatten the output of the last layer
        F1 = M51.view(32, -1)

        D1 = self.fc1(F1)
        A6 = F.relu(D1)
        D2 = self.fc2(A6)

        return self.softmax(D2)


class Net2(nn.Module):
    def __init__(self):
        super(Net2, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4 * 4 * 50, 500)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4 * 4 * 50)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


def process_data():
    # second version of adding random noise (Amplify and Stretch)
    def stretch(x):
        l = int(187 * (1 + (random.random() - 0.5) / 3))
        y = sci.resample(x, l)
        if l < 187:
            y_ = np.zeros(shape=(187,))
            y_[:l] = y
        else:
            y_ = y[:187]
        return y_

    def amplify(x):
        alpha = (random.random() - 0.5)
        factor = -alpha * x + (1 + alpha)
        return x * factor

    def add_amplify_and_stretch_noise(x):
        result = np.zeros(shape=187)
        if random.random() < 0.33:
            new_y = stretch(x)
        elif random.random() < 0.66:
            new_y = amplify(x)
        else:
            new_y = stretch(x)
            new_y = amplify(new_y)
        return new_y

    # Data Exploration (MIT-BIH)
    mitbih_train_loc = "C:/Users/Lee/Downloads/archive/mitbih_train.csv"
    mitbih_test_loc = "C:/Users/Lee/Downloads/archive/mitbih_test.csv"
    mitbih_train_df = pd.read_csv(mitbih_train_loc, header=None)
    mitbih_test_df = pd.read_csv(mitbih_test_loc, header=None)

    dataset = pd.concat([mitbih_train_df, mitbih_test_df], axis=0, sort=True).reset_index(drop=True)

    labels = dataset.iloc[:, -1].astype('category').map({
        0: 'N - Normal Beat',
        1: 'S - Supraventricular premature or ectopic beat',
        2: 'V - Premature ventricular contraction',
        3: 'F - Fusion of ventricular and normal beat',
        4: 'Q - Unclassified beat'})

    # since the last column is the category
    obs = np.array(dataset.iloc[:, :187])

    # get the indexes of all labels
    n_indexes = labels.index[labels == 'N - Normal Beat']
    q_indexes = labels.index[labels == 'Q - Unclassified beat']
    v_indexes = labels.index[labels == 'V - Premature ventricular contraction']
    s_indexes = labels.index[labels == 'S - Supraventricular premature or ectopic beat']
    f_indexes = labels.index[labels == 'F - Fusion of ventricular and normal beat']

    # resample indexes of each class
    n_indexes_resampled = skl.resample(n_indexes, replace=True, n_samples=10000, random_state=random_seed)
    q_indexes_resampled = skl.resample(q_indexes, replace=True, n_samples=10000, random_state=random_seed)
    v_indexes_resampled = skl.resample(v_indexes, replace=True, n_samples=10000, random_state=random_seed)
    s_indexes_resampled = skl.resample(s_indexes, replace=True, n_samples=10000, random_state=random_seed)
    f_indexes_resampled = skl.resample(f_indexes, replace=True, n_samples=10000, random_state=random_seed)

    # initialize the labels_resampled to empty pandas series
    labels_resampled = pd.Series([])
    obs_resampled = None

    # add all indexes_resampled for all classes to iterate
    label_indexes_list = [n_indexes_resampled,
                          q_indexes_resampled,
                          v_indexes_resampled,
                          s_indexes_resampled,
                          f_indexes_resampled]

    for label_indexes in label_indexes_list:
        # append labels for all resampled classes
        labels_resampled = labels_resampled.append(labels[label_indexes], ignore_index=True)

        # append observations for all resampled classes
        if obs_resampled is None:
            obs_resampled = obs[label_indexes]
        else:
            obs_resampled = np.concatenate((obs_resampled, obs[label_indexes]))

    # convert labels_resampled to its integer encoding of the following listing:
    #     0: 'N - Normal Beat'
    #     1: 'S - Supraventricular premature or ectopic beat'
    #     2: 'V - Premature ventricular contraction'
    #     3: 'F - Fusion of ventricular and normal beat'
    #     4: 'Q - Unclassified beat
    # ---------------------------------------------------------------------------- #
    labs = pd.factorize(labels_resampled.astype('category'))[0]
    obs = np.array([add_amplify_and_stretch_noise(obs) for obs in obs_resampled])

    return labs, obs


def define_and_get_arguments(args=sys.argv[1:]):
    parser = argparse.ArgumentParser(
        description="Run federated learning using websocket client workers."
    )
    parser.add_argument("--batch_size", type=int, default=32, help="batch size of the training")
    parser.add_argument(
        "--test_batch_size", type=int, default=1000, help="batch size used for the test data"
    )
    parser.add_argument("--epochs", type=int, default=2, help="number of epochs to train")
    parser.add_argument(
        "--federate_after_n_batches",
        type=int,
        default=50,
        help="number of training steps performed on each remote worker " "before averaging",
    )
    parser.add_argument("--lr", type=float, default=0.001, help="learning rate")
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


def train_on_batches(worker, batches, model_in, device, criterion, lr=0.001):
    model = model_in.copy()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    model = model.float()
    model.train()
    model.send(worker)
    loss_local = False
    for batch_idx, (data, target) in enumerate(batches):
        loss_local = False
        data, target = data.to(device), target.to(device)

        if batch_idx == 9999:
            print(data.get(), target.get())

        # clear the gradients of all optimized variables
        optimizer.zero_grad()
        # forward pass: compute predicted outputs by passing inputs to the model
        output = model(data.float())
        # calculate the batch loss
        loss = criterion(output, target)
        # backward pass: compute gradient of the loss with respect to model parameters
        loss.backward()
        # perform a single optimization step (parameter update)
        optimizer.step()

        # Calculating loss
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


def get_next_batches(fdataloader: sy.FederatedDataLoader, nr_batches: int):
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


def train(
        model, device, federated_train_loader, lr, federate_after_n_batches, abort_after_one=False
):
    model.train()

    nr_batches = federate_after_n_batches

    models = {}
    loss_values = {}
    criterion = nn.NLLLoss()

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
            if len(curr_batches) < 50:
                data_for_all_workers = False
            elif curr_batches:
                models[worker], loss_values[worker] = train_on_batches(
                    worker, curr_batches, model, device, criterion, lr
                )
        counter += nr_batches

        if not data_for_all_workers:
            logger.debug("At least one worker ran out of data, stopping.")
            break

        model = utils.federated_avg(models)
        batches = get_next_batches(federated_train_loader, nr_batches)
        if abort_after_one:
            break
    return model


def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction="sum").item()  # sum up batch loss
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


def convert_to_dataset(x, y):
    data = []
    for i in range(len(x)):
        data.append([x[i], y[i]])

    return data


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

    # 워커 객체를 리스트로 묶음
    workers = [alice, bob, charlie]

    # 쿠다 사용 여부
    use_cuda = args.cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    kwargs = {"num_workers": 1, "pin_memory": True} if use_cuda else {}

    # 랜덤 시드 설정
    torch.manual_seed(args.seed)

    labels_resampled_factorized, obs_resampled_with_noise_2 = process_data()

    # percentage of test/valid set to use for testing and validation from the test_valid_idx (to be called test_size)
    test_size = 0.1

    # obtain training indices that will be used for validation
    num_train = len(obs_resampled_with_noise_2)
    indices = list(range(num_train))
    np.random.shuffle(indices)
    split = int(np.floor(test_size * num_train))
    train_idx, test_idx = indices[split:], indices[:split]

    federated_train_dataset = D.TensorDataset(torch.tensor(obs_resampled_with_noise_2[train_idx]),
                                              torch.tensor(labels_resampled_factorized[train_idx]))

    federated_train_loader = sy.FederatedDataLoader(
                                                    federated_train_dataset.federate(tuple(workers)),
                                                    batch_size=args.batch_size,
                                                    shuffle=True,
                                                    iter_per_worker=True,
                                                    **kwargs,
                                                    )

    test_dataset = D.TensorDataset(torch.tensor(obs_resampled_with_noise_2[test_idx]),
                                   torch.tensor(labels_resampled_factorized[test_idx]))

    test_loader = D.DataLoader(test_dataset, shuffle=True, batch_size=args.batch_size,
                             num_workers=0, drop_last=True)

    model = Net(input_features=1, output_dim=5).to(device)
    # model = Net2().to(device)


    for epoch in range(1, args.epochs + 1):
        logger.info("Starting epoch %s/%s", epoch, args.epochs)
        model = train(model, device, federated_train_loader, args.lr, args.federate_after_n_batches)
        test(model, device, test_loader)

    if args.save_model:
        torch.save(model.state_dict(), "mnist_cnn.pt")


if __name__ == "__main__":
    FORMAT = "%(asctime)s %(levelname)s %(filename)s(l:%(lineno)d) - %(message)s"
    LOG_LEVEL = logging.DEBUG
    logging.basicConfig(format=FORMAT, level=LOG_LEVEL)

    websockets_logger = logging.getLogger("websockets")
    websockets_logger.setLevel(logging.DEBUG)
    websockets_logger.addHandler(logging.StreamHandler())

    main()
