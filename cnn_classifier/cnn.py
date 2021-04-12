import torch
from torch import nn
#from torchvision import transforms
from torch.utils.data import TensorDataset, DataLoader, random_split
#from torch.utils.data.sampler import SubsetRandomSampler
import numpy as np
import argparse
from pathlib import Path
from sklearn.metrics import roc_auc_score

class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()

        # Convolution 1
        self.cnn1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=5, stride=1, padding=2)
        self.relu1 = nn.ReLU()

        # Max pool 1
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)

        # Convolution 2
        self.cnn2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, stride=1, padding=2)
        self.relu2 = nn.ReLU()
        # Max pool 2
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)

        # Fully connected 1 (readout)
        self.fc1 = nn.Linear(32 * 232 * 2, 2)
        # self.fc1 = nn.Linear(32 * 232, 1)

    def forward(self, x):
        # Convolution 1
        out = self.cnn1(x)
        out = self.relu1(out)

        # Max pool 1
        out = self.maxpool1(out)

        # Convolution 2 
        out = self.cnn2(out)
        out = self.relu2(out)

        # Max pool 2 
        out = self.maxpool2(out)

        # Resize
        # Original size: (100, 32, 7, 7)
        # out.size(0): 100
        # New out size: (100, 32*7*7)
        out = out.view(out.size(0), -1)

        # Linear function (readout)
        out = self.fc1(out)

        return out

parser = argparse.ArgumentParser()
parser.add_argument("-data_dir", help = "dataset folder", type = str)
args = parser.parse_args()
if not args.data_dir:
    print("Dataset directory does not exist")
    exit()
else:
    dataset = Path(args.data_dir)

#pos_x = np.load('pos-train.c.1.1.npy')
#neg_x = np.load('neg-train.c.1.1.npy')
for file in sorted(dataset.glob('*.npy')):
    if 'pos-train' in file.name:
        pos_x = np.load(file)
    if 'neg-train' in file.name:
        neg_x = np.load(file)
    if 'pos-test' in file.name:
        pos_test_x = np.load(file)
    if 'neg-test' in file.name:
        neg_test_x = np.load(file)

pos_y = np.ones((pos_x.shape[0]))
neg_y = np.zeros((neg_x.shape[0]))

pos_test_y = np.ones((pos_test_x.shape[0]))
neg_test_y = np.zeros((neg_test_x.shape[0]))

X = np.vstack((pos_x, neg_x))
print("x shape", X.shape)
y = np.concatenate((pos_y, neg_y), axis = 0)
print("y shape", y.shape)

X_test = np.vstack((pos_test_x, neg_test_x))
print("x test shape", X_test.shape)
y_test = np.concatenate((pos_test_y, neg_test_y), axis = 0)
print("y test shape", y_test.shape)

#shuffle_idx = np.random.permutation(X.shape[0])
#X = torch.from_numpy(X[shuffle_idx])
#y = torch.from_numpy(y[shuffle_idx])
#X = torch.from_numpy(X).float()
#y = torch.from_numpy(y).float()

train_set = TensorDataset(torch.from_numpy(X), torch.from_numpy(y))
test_set = TensorDataset(torch.from_numpy(X_test), torch.from_numpy(y_test))

#train_size = np.asarray(X.shape) * 0.75
#test_size = np.asarray(X.shape) * 0.25
#train_size = int(float(X.shape[0]) * 0.75)
#test_size = int(float(X.shape[0]) * 0.25)
#train_set, test_set = random_split(dataset, [train_size, test_size], generator=torch.Generator().manual_seed(42))

batch_size = 32
#n_iters = 10000
num_epochs = 10#int(n_iters / (len(train_set) / batch_size))
train_loader = DataLoader(dataset = train_set, batch_size = batch_size, shuffle = True)
test_loader = DataLoader(dataset = test_set, batch_size = batch_size, shuffle = True)

model = CNNModel()
criterion = nn.CrossEntropyLoss()
learning_rate = 0.01

optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

print(model.parameters())

print(len(list(model.parameters())))

# Convolution 1: 16 Kernels
print(list(model.parameters())[0].size())

# Convolution 1 Bias: 16 Kernels
print(list(model.parameters())[1].size())

# Convolution 2: 32 Kernels with depth = 16
print(list(model.parameters())[2].size())

# Convolution 2 Bias: 32 Kernels with depth = 16
print(list(model.parameters())[3].size())

# Fully Connected Layer 1
print(list(model.parameters())[4].size())

# Fully Connected Layer Bias
print(list(model.parameters())[5].size())

iter = 0
alpha = 10
for epoch in range(num_epochs):
    y_pred = np.empty(y_test.shape)
    for i, (x_train, labels) in enumerate(train_loader):
        # Load images
        #images = images.requires_grad_()
        num_samples_train = x_train.size()[0]
        x_train = x_train.unsqueeze(1).expand(num_samples_train, 1, 931, alpha).requires_grad_()

        # Clear gradients w.r.t. parameters
        optimizer.zero_grad()

        # Forward pass to get output/logits
        outputs = model(x_train)

        # Calculate Loss: softmax --> cross entropy loss
        #print(labels.size())
        loss = criterion(outputs, labels.long())

        # Getting gradients w.r.t. parameters
        loss.backward()

        # Updating parameters
        optimizer.step()

        # iter += 1

        # if iter % 500 == 0:
        # Calculate Accuracy
    correct = 0
    total = 0
    # Iterate through test dataset
    for i, (x_test, labels) in enumerate(test_loader):
        # Load test set
        num_samples_test = x_test.size()[0]
        x_test = x_test.unsqueeze(1).expand(num_samples_test, 1, 931, alpha).requires_grad_()

        # Forward pass only to get logits/output
        outputs = model(x_test)

        # Get predictions from the maximum value
        _, predicted = torch.max(outputs.data, 1)
        # Total number of labels
        total += labels.size(0)

        # Total correct predictions
        correct += (predicted == labels).sum()

        pred_size = [*predicted.size()][0]
        y_pred[i * pred_size : (i + 1) * (pred_size)] = predicted.numpy()

    accuracy = 100 * correct / total

    # Print Loss
    print('Epoch: {}, Loss: {}, ROC AUC: {}, Accuracy: {}'.format(epoch, loss.item(), roc_auc_score(y_test, y_pred), accuracy))