from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
import multiprocessing
from utils.config_utils import read_args, load_config, Dict2Object
import warnings
warnings.filterwarnings("ignore")
class Arg:
    config_file: str
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output


def train(args, model, device, train_loader, optimizer, epoch):
    """
    tain the model and return the training accuracy
    :param args: input arguments
    :param model: neural network model
    :param device: the device where model stored
    :param train_loader: data loader
    :param optimizer: optimizer
    :param epoch: current epoch
    :return:
    """
    model.train()
    train_acc = train_loss = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        '''Fill your code'''
        this_loss = float(loss.item())
        train_loss += this_loss 
        this_acc = (torch.argmax(output, 1) == target).sum().item()
        train_acc += float(this_acc)
    training_acc, training_loss = train_acc / len(train_loader.dataset), train_loss / len(train_loader.dataset)  # replace this line
    print("Loss: ", training_loss, ", Accuracy: ", 100.0 * training_acc, "%")
    try:
        with open("train_" + str(args.seed) + ".txt", "a") as f:
            f.write("Loss: " + str(training_loss) + "\n")
    except:
        with open("train_" + str(args.seed) + ".txt", "w") as f:
            f.write("Loss: " + str(training_loss) + "\n")
    return training_acc, training_loss


def test(config, model, device, test_loader):
    """
    test the model and return the tesing accuracy
    :param model: neural network model
    :param device: the device where model stored
    :param test_loader: data loader
    :return:
    """
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            '''Fill your code'''
            output = model(data)
            test_loss += F.nll_loss(output, target, size_average=False).item()
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).sum()
    testing_acc, testing_loss = float(correct / len(test_loader.dataset)), test_loss / len(test_loader.dataset)  # replace this line
    print("Test Loss: ", testing_loss, ", Test Accuracy: ", testing_acc * 100.0, "%")
    try:
        with open("test_" + str(config.seed) + ".txt", "a") as f:
            f.write("Test Loss: " + str(testing_loss) + ", Test Accuracy: " + str(testing_acc * 100.0) + "%\n")
    except:
        with open("test_" + str(config.seed) + ".txt", "w") as f:
            f.write("Test Loss: " + str(testing_loss) + ", Test Accuracy: " + str(testing_acc * 100.0) + "%\n")
    return testing_acc, testing_loss


def plot(epoches, performance, title):
    """
    plot the model peformance
    :param epoches: recorded epoches
    :param performance: recorded performance
    :return:
    """
    """Fill your code"""
    import matplotlib.pyplot as plt
    plt.plot(epoches, performance)
    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel("Performance: " + title)
    plt.show()


def run(config, pipe):
    use_cuda = not config.no_cuda and torch.cuda.is_available()
    use_mps = not config.no_mps and torch.backends.mps.is_available()

    torch.manual_seed(config.seed)

    if use_cuda:
        device = torch.device("cuda")
    elif use_mps:
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    train_kwargs = {'batch_size': config.batch_size}
    test_kwargs = {'batch_size': config.test_batch_size}
    if use_cuda:
        cuda_kwargs = {'num_workers': 1,
                       'pin_memory': True,
                       'shuffle': True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

    # download data
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    dataset1 = datasets.MNIST('./data', train=True, download=True, transform=transform)
    dataset2 = datasets.MNIST('./data', train=False, transform=transform)

    """add random seed to the DataLoader, pls modify this function"""
    train_loader = torch.utils.data.DataLoader(dataset1, **train_kwargs)
    test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)

    model = Net().to(device)
    optimizer = optim.Adadelta(model.parameters(), lr=config.lr)

    """record the performance"""
    epoches = []
    training_accuracies = []
    training_loss = []
    testing_accuracies = []
    testing_loss = []

    scheduler = StepLR(optimizer, step_size=1, gamma=config.gamma)
    for epoch in range(1, config.epochs + 1):
        print("Epoch: " + str(epoch) + ", Seed: " + str(config.seed))
        train_acc, train_loss = train(config, model, device, train_loader, optimizer, epoch)
        """record training info, Fill your code"""
        training_accuracies.append(train_acc)
        training_loss.append(train_loss)
        test_acc, test_loss = test(config, model, device, test_loader)
        """record testing info, Fill your code"""
        testing_accuracies.append(test_acc)
        testing_loss.append(test_loss)
        scheduler.step()
        """update the records, Fill your code"""
        epoches.append(epoch)

    """plotting training performance with the records"""
    plot(epoches, training_loss, "Training Loss, Seed = " + str(config.seed))

    """plotting testing performance with the records"""
    plot(epoches, testing_accuracies, "Testing Accuracy, Seed = " + str(config.seed))
    plot(epoches, testing_loss, "Testing Loss, Seed = " + str(config.seed))

    if config.save_model:
        torch.save(model.state_dict(), "mnist_cnn_" + str(config.seed) + ".pt")
    pipe.send(training_loss)
    pipe.send(testing_accuracies)
    pipe.send(testing_loss)



def plot_mean(result):
    """
    Read the recorded results.
    Plot the mean results after three runs.
    :return:
    """
    """fill your code"""
    mean_training_loss = [sum(i) / len(i) for i in zip(*result[0])]
    mean_testing_accuracies = [sum(i) / len(i) for i in zip(*result[1])]
    mean_testing_loss = [sum(i) / len(i) for i in zip(*result[2])]
    epoches = list(range(1, len(mean_training_loss) + 1)) # Should depend on config
    """plotting training performance with the records"""
    plot(epoches, mean_training_loss, "Mean Training Loss")

    """plotting testing performance with the records"""
    plot(epoches, mean_testing_accuracies, "Mean Testing Accuracy")
    plot(epoches, mean_testing_loss, "Mean Testing Loss")
    


if __name__ == '__main__':
    processes = []
    """remove existing files"""
    import os
    result = [[], [], []]
    pipes = []
    files = os.listdir(".")
    for file in files:
        if ".txt" in file and file != "requirement.txt":
            os.remove(file)
    """toad training settings"""
    try:
        arg = read_args()
        config = load_config(arg)
        run(config)
    except:
        print("Trying to read configs from \"config\" folder.")
        files = os.listdir("config")
        arg = Arg()
        for file in files:
            if file != "minist.yaml":
                arg.config_file = "config/" + file
                config = load_config(arg)
                pipe_recv, pipe_send = multiprocessing.Pipe(duplex=False)
                pipes.append(pipe_recv)
                processes.append(multiprocessing.Process(target=run, args=(config, pipe_send)))
                processes[-1].start()

    # """train model and record results"""
    # run(config)
    for pipe in pipes:
        result[0].append(pipe.recv())
        result[1].append(pipe.recv())
        result[2].append(pipe.recv())
    for process in processes:
        process.join()
    """plot the mean results"""
    plot_mean(result)
