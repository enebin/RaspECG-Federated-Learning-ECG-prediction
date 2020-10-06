from __future__ import unicode_literals, print_function, division
from torch.utils.data import Dataset

import torch
from io import open
import glob
import os
import numpy as np
import unicodedata
import string
import random
import torch.nn as nn
import time
import math
import pandas as pd
import random
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import urllib.request
from zipfile import ZipFile

#hide TF-related warnings in PySyft
import warnings
warnings.filterwarnings("ignore")
import syft as sy
from syft.frameworks.torch.fl import utils
from syft.workers.websocket_client import WebsocketClientWorker


# create a function for checking if the dataset does indeed exist
def dataset_exists():
    return (os.path.isfile('./data/eng-fra.txt') and
            # check if all 18 files are indeed in the ./data/names/ directory
            os.path.isdir('./data/names/') and
            os.path.isfile('./data/names/Arabic.txt') and
            os.path.isfile('./data/names/Chinese.txt') and
            os.path.isfile('./data/names/Czech.txt') and
            os.path.isfile('./data/names/Dutch.txt') and
            os.path.isfile('./data/names/English.txt') and
            os.path.isfile('./data/names/French.txt') and
            os.path.isfile('./data/names/German.txt') and
            os.path.isfile('./data/names/Greek.txt') and
            os.path.isfile('./data/names/Irish.txt') and
            os.path.isfile('./data/names/Italian.txt') and
            os.path.isfile('./data/names/Japanese.txt') and
            os.path.isfile('./data/names/Korean.txt') and
            os.path.isfile('./data/names/Polish.txt') and
            os.path.isfile('./data/names/Portuguese.txt') and
            os.path.isfile('./data/names/Russian.txt') and
            os.path.isfile('./data/names/Scottish.txt') and
            os.path.isfile('./data/names/Spanish.txt') and
            os.path.isfile('./data/names/Vietnamese.txt'))


# If the dataset does not exist, then proceed to download the dataset anew
if not dataset_exists():
    # If the dataset does not already exist, let's download the dataset directly from the URL where it is hosted
    print('Downloading the dataset with urllib2 to the current directory...')
    url = 'https://download.pytorch.org/tutorial/data.zip'
    urllib.request.urlretrieve(url, './data.zip')
    print("The dataset was successfully downloaded")
    print("Unzipping the dataset...")
    with ZipFile('./data.zip', 'r') as zipObj:
        # Extract all the contents of the zip file in current directory
        zipObj.extractall()
    print("Dataset successfully unzipped")
else:
    print("Not downloading the dataset because it was already downloaded")


# Load all the files in a certain path
def findFiles(path):
    return glob.glob(path)


# Read a file and split into lines
def readLines(filename):
    lines = open(filename, encoding='utf-8').read().strip().split('\n')
    return [unicodeToAscii(line) for line in lines]


# convert a string 's' in unicode format to ASCII format
def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
        and c in all_letters
    )


all_letters = string.ascii_letters + " .,;'"
n_letters = len(all_letters)

# dictionary containing the nation as key and the names as values
# Example: category_lines["italian"] = ["Abandonato","Abatangelo","Abatantuono",...]
category_lines = {}
# List containing the different categories in the data
all_categories = []

for filename in findFiles('data/names/*.txt'):
    print(filename)
    category = os.path.splitext(os.path.basename(filename))[0]
    all_categories.append(category)
    lines = readLines(filename)
    category_lines[category] = lines

n_categories = len(all_categories)

print("Amount of categories:" + str(n_categories))


class LanguageDataset(Dataset):
    # Constructor is mandatory
    def __init__(self, text, labels, transform=None):
        self.data = text
        self.targets = labels  # categories
        # self.to_torchtensor()
        self.transform = transform

    def to_torchtensor(self):
        self.data = torch.from_numpy(self.text, requires_grad=True)
        self.labels = torch.from_numpy(self.targets, requires_grad=True)

    def __len__(self):
        # Mandatory
        '''Returns:
                Length [int]: Length of Dataset/batches
        '''
        return len(self.data)

    def __getitem__(self, idx):
        # Mandatory

        '''Returns:
                 Data [Torch Tensor]:
                 Target [ Torch Tensor]:
        '''
        sample = self.data[idx]
        target = self.targets[idx]

        if self.transform:
            sample = self.transform(sample)

        return sample, target


# The list of arguments for our program. We will be needing most of them soon.
class Arguments():
    def __init__(self):
        self.batch_size = 1
        self.learning_rate = 0.005
        self.epochs = 10000
        self.federate_after_n_batches = 15000
        self.seed = 1
        self.print_every = 200
        self.plot_every = 100
        self.use_cuda = False


args = Arguments()

# Set of names(X)
names_list = []
# Set of labels (Y)
category_list = []

# Convert into a list with corresponding label.

for nation, names in category_lines.items():
    # iterate over every single name
    for name in names:
        names_list.append(name)  # input data point
        category_list.append(nation)  # label

# let's see if it was successfully loaded. Each data sample(X) should have its own corresponding category(Y)
print(names_list[1:20])
print(category_list[1:20])

print("\n \n Amount of data points loaded: " + str(len(names_list)))

#Assign an integer to every category
categories_numerical = pd.factorize(category_list)[0]
#Let's wrap our categories with a tensor, so that it can be loaded by LanguageDataset
category_tensor = torch.tensor(np.array(categories_numerical), dtype=torch.long)
#Ready to be processed by torch.from_numpy in LanguageDataset
categories_numpy = np.array(category_tensor)

#Let's see a few resulting categories
print(names_list[1200:1210])
print(categories_numpy[1200:1210])


def letterToIndex(letter):
    return all_letters.find(letter)


# Just for demonstration, turn a letter into a <1 x n_letters> Tensor
def letterToTensor(letter):
    tensor = torch.zeros(1, n_letters)
    tensor[0][letterToIndex(letter)] = 1
    return tensor


# Turn a line into a <line_length x 1 x n_letters>,
# or an array of one-hot letter vectors
def lineToTensor(line):
    tensor = torch.zeros(len(line), 1, n_letters)  # Daniele: len(max_line_size) was len(line)
    for li, letter in enumerate(line):
        tensor[li][0][letterToIndex(letter)] = 1
    # Daniele: add blank elements over here
    return tensor


def list_strings_to_list_tensors(names_list):
    lines_tensors = []
    for index, line in enumerate(names_list):
        lineTensor = lineToTensor(line)
        lineNumpy = lineTensor.numpy()
        lines_tensors.append(lineNumpy)

    return (lines_tensors)


lines_tensors = list_strings_to_list_tensors(names_list)

print(names_list[0])
print(lines_tensors[0])
print(lines_tensors[0].shape)

max_line_size = max(len(x) for x in lines_tensors)


def lineToTensorFillEmpty(line, max_line_size):
    tensor = torch.zeros(max_line_size, 1, n_letters)  # notice the difference between this method and the previous one
    for li, letter in enumerate(line):
        tensor[li][0][letterToIndex(letter)] = 1

        # Vectors with (0,0,.... ,0) are placed where there are no characters
    return tensor


def list_strings_to_list_tensors_fill_empty(names_list):
    lines_tensors = []
    for index, line in enumerate(names_list):
        lineTensor = lineToTensorFillEmpty(line, max_line_size)
        lines_tensors.append(lineTensor)
    return (lines_tensors)


lines_tensors = list_strings_to_list_tensors_fill_empty(names_list)

# Let's take a look at what a word now looks like
print(names_list[0])
print(lines_tensors[0])
print(lines_tensors[0].shape)

#And finally, from a list, we can create a numpy array with all our word embeddings having the same shape:
array_lines_tensors = np.stack(lines_tensors)
#However, such operation introduces one extra dimension (look at the dimension with index=2 having size '1')
print(array_lines_tensors.shape)
#Because that dimension just has size 1, we can get rid of it with the following function call
array_lines_proper_dimension = np.squeeze(array_lines_tensors, axis=2)
print(array_lines_proper_dimension.shape)


def find_start_index_per_category(category_list):
    categories_start_index = {}

    # Initialize every category with an empty list
    for category in all_categories:
        categories_start_index[category] = []

    # Insert the start index of each category into the dictionary categories_start_index
    # Example: "Italian" --> 203
    #         "Spanish" --> 19776
    last_category = None
    i = 0
    for name in names_list:
        cur_category = category_list[i]
        if (cur_category != last_category):
            categories_start_index[cur_category] = i
            last_category = cur_category

        i = i + 1

    return (categories_start_index)


categories_start_index = find_start_index_per_category(category_list)

print(categories_start_index)

def randomChoice(l):
    rand_value = random.randint(0, len(l) - 1)
    return l[rand_value], rand_value


def randomTrainingIndex():
    category, rand_cat_index = randomChoice(all_categories) #cat = category, it's not a random animal
    #rand_line_index is a relative index for a data point within the random category rand_cat_index
    line, rand_line_index = randomChoice(category_lines[category])
    category_start_index = categories_start_index[category]
    absolute_index = category_start_index + rand_line_index
    return(absolute_index)


# Two hidden layers, based on simple linear layers

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
        self.i2o = nn.Linear(input_size + hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        combined = torch.cat((input, hidden), 1)
        hidden = self.i2h(combined)
        output = self.i2o(combined)
        output = self.softmax(output)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, self.hidden_size)


# Let's instantiate the neural network already:
n_hidden = 128
# Instantiate RNN

device = torch.device("cuda" if args.use_cuda else "cpu")
model = RNN(n_letters, n_hidden, n_categories).to(device)
# The final softmax layer will produce a probability for each one of our 18 categories
print(model)

#Now let's define our workers. You can either use remote workers or virtual workers
hook = sy.TorchHook(torch)  # <-- NEW: hook PyTorch ie add extra functionalities to support Federated Learning
#alice = sy.VirtualWorker(hook, id="alice")
#bob = sy.VirtualWorker(hook, id="bob")
#charlie = sy.VirtualWorker(hook, id="charlie")

#workers_virtual = [alice, bob]

ip_alice = '192.168.0.52'
ip_bob = '192.168.0.53'

#If you have your workers operating remotely, like on Raspberry PIs
kwargs_websocket_alice = {"host": ip_alice, "hook": hook}
alice = WebsocketClientWorker(id="alice", port=10002, **kwargs_websocket_alice)

kwargs_websocket_bob = {"host": ip_bob, "hook": hook}
bob = WebsocketClientWorker(id="bob", port=10003, **kwargs_websocket_bob)

workers_virtual = [alice, bob]

#array_lines_proper_dimension = our data points(X)
#categories_numpy = our labels (Y)
langDataset =  LanguageDataset(array_lines_proper_dimension, categories_numpy)

#assign the data points and the corresponding categories to workers.
federated_train_loader = sy.FederatedDataLoader(
            langDataset
            .federate(workers_virtual),
            batch_size=args.batch_size)


def categoryFromOutput(output):
    top_n, top_i = output.topk(1)
    category_i = top_i[0].item()
    return all_categories[category_i], category_i


def timeSince(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def fed_avg_every_n_iters(model_pointers, iter, federate_after_n_batches):
    models_local = {}

    if (iter % args.federate_after_n_batches == 0):
        for worker_name, model_pointer in model_pointers.items():
            #                #need to assign the model to the worker it belongs to.
            models_local[worker_name] = model_pointer.copy().get()
        model_avg = utils.federated_avg(models_local)

        for worker in workers_virtual:
            model_copied_avg = model_avg.copy()
            model_ptr = model_copied_avg.send(worker)
            model_pointers[worker.id] = model_ptr

    return (model_pointers)


def fw_bw_pass_model(model_pointers, line_single, category_single):
    # get the right initialized model
    model_ptr = model_pointers[line_single.location.id]
    line_reshaped = line_single.reshape(max_line_size, 1, len(all_letters))
    line_reshaped, category_single = line_reshaped.to(device), category_single.to(device)
    # Firstly, initialize hidden layer
    hidden_init = model_ptr.initHidden()
    # And now zero grad the model
    model_ptr.zero_grad()
    hidden_ptr = hidden_init.send(line_single.location)
    amount_lines_non_zero = len(torch.nonzero(line_reshaped.copy().get()))
    # now need to perform forward passes
    for i in range(amount_lines_non_zero):
        output, hidden_ptr = model_ptr(line_reshaped[i], hidden_ptr)
    criterion = nn.NLLLoss()
    loss = criterion(output, category_single)
    loss.backward()

    model_got = model_ptr.get()

    # Perform model weights' updates
    for param in model_got.parameters():
        param.data.add_(-args.learning_rate, param.grad.data)

    model_sent = model_got.send(line_single.location.id)
    model_pointers[line_single.location.id] = model_sent

    return (model_pointers, loss, output)


def train_RNN(n_iters, print_every, plot_every, federate_after_n_batches, list_federated_train_loader):
    current_loss = 0
    all_losses = []

    model_pointers = {}

    # Send the initialized model to every single worker just before the training procedure starts
    for worker in workers_virtual:
        model_copied = model.copy()
        model_ptr = model_copied.send(worker)
        model_pointers[worker.id] = model_ptr

    # extract a random element from the list and perform training on it
    for iter in range(1, n_iters + 1):
        random_index = randomTrainingIndex()
        line_single, category_single = list_federated_train_loader[random_index]
        # print(category_single.copy().get())
        line_name = names_list[random_index]
        model_pointers, loss, output = fw_bw_pass_model(model_pointers, line_single, category_single)
        # model_pointers = fed_avg_every_n_iters(model_pointers, iter, args.federate_after_n_batches)
        # Update the current loss a
        loss_got = loss.get().item()
        current_loss += loss_got

        if iter % plot_every == 0:
            all_losses.append(current_loss / plot_every)
            current_loss = 0

        if (iter % print_every == 0):
            output_got = output.get()  # Without copy()
            guess, guess_i = categoryFromOutput(output_got)
            category = all_categories[category_single.copy().get().item()]
            correct = '✓' if guess == category else '✗ (%s)' % category
            print('%d %d%% (%s) %.4f %s / %s %s' % (
            iter, iter / n_iters * 100, timeSince(start), loss_got, line_name, guess, correct))
    return (all_losses, model_pointers)

#This may take a few seconds to complete.
print("Generating list of batches for the workers...")
list_federated_train_loader = list(federated_train_loader)

start = time.time()
all_losses, model_pointers = train_RNN(args.epochs, args.print_every, args.plot_every, args.federate_after_n_batches, list_federated_train_loader)

#Let's plot the loss we got during the training procedure
plt.figure()
plt.ylabel("Loss")
plt.xlabel('Epochs (100s)')
plt.plot(all_losses)


def predict(model, input_line, worker, n_predictions=3):
    #     model = model.copy().get()
    print('\n> %s' % input_line)
    model_remote = model.send(worker)
    line_tensor = lineToTensor(input_line)
    line_remote = line_tensor.copy().send(worker)
    # line_tensor = lineToTensor(input_line)
    # output = evaluate(model, line_remote)
    # Get top N categories
    hidden = model_remote.initHidden()
    hidden_remote = hidden.copy().send(worker)

    with torch.no_grad():
        for i in range(line_remote.shape[0]):
            output, hidden_remote = model_remote(line_remote[i], hidden_remote)

    topv, topi = output.copy().get().topk(n_predictions, 1, True)
    predictions = []

    for i in range(n_predictions):
        value = topv[0][i].item()
        category_index = topi[0][i].item()
        print('(%.2f) %s' % (value, all_categories[category_index]))
        predictions.append([value, all_categories[category_index]])

model_alice = model_pointers["alice"].get()
model_bob = model_pointers["bob"].get()

predict(model_alice.copy(), "Qing", alice)
predict(model_alice.copy(), "Daniele", alice)

predict(model_bob.copy(), "Qing", alice)
predict(model_bob.copy(), "Daniele", alice)