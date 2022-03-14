from locale import normalize
from turtle import forward
import numpy as np
import matplotlib.pyplot as plt

import torch
from torch.nn import MSELoss, Sequential, ReLU, Sigmoid, Tanh, Linear
from torch.optim import SGD
from torch.utils.data import DataLoader
from dataset import DummyDataset

# Which columun of the dataset to use for input and output
input_columns = ["x", "y"]
output_columns = ["x+y"]

# Training procedure hyper-parameters
nb_epoch = 150
batch_size = 16
learning_rate = 1e-3
save_plot_path = "training_plot.png"

# Keep False until you've found an architecture you're confident of/happy with.
evaluate_on_test_set = False

# Load data
dataloader_train = DataLoader(dataset=DummyDataset("train_data.csv", input_columns, output_columns), batch_size=batch_size)
dataloader_eval = DataLoader(dataset=DummyDataset("evaluation_data.csv", input_columns, output_columns), batch_size=batch_size)
dataloader_dev = DataLoader(dataset=DummyDataset("validation_data.csv", input_columns, output_columns), batch_size=batch_size)

# Build model
model = Sequential(
    Linear(in_features=2, out_features=3),
    Tanh(),
    Linear(3, 3),
    Tanh(),
    Linear(3, 1),
    Sigmoid()
)
print(model)

# Define loss
class MSLELoss(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = MSELoss()
        
    def forward(self, pred, actual):
        return self.mse(torch.log(pred + 1), torch.log(actual + 1))
mse = MSELoss()
# mse = MSLELoss()

# Define optimizer
optimizer = SGD(params=model.parameters(), lr=learning_rate)

# Prepare normalization
class TorchStandardScaler:
    def fit(self, x):
        self.mean = x.mean(0, keepdim=True)
        self.std = x.std(0, unbiased=False, keepdim=True)
  
    def transform(self, x):
        x -= self.mean
        x /= (self.std + 1e-7)
        return x

    def inverse_transform(self, x):
        x *= (self.std + 1e-7)
        x += self.mean
        return x
train_set=DummyDataset("train_data.csv", input_columns, output_columns)        
loader = DataLoader(train_set, batch_size=len(train_set), num_workers=1)
data = next(iter(loader))
standard_scaler = TorchStandardScaler()
standard_scaler.fit(data[1])
del train_set, loader, data


# Training loop
mse_train_overall, mse_dev_overall = [], []
for epoch in range(1,nb_epoch+1): # Perform training and validation for an epoch (ie. Every sample of the dataset is seen exactly once.)
    
    # Perform training using mini-batches
    mse_train_epoch = []
    for i, batch_data in enumerate(dataloader_train):
        # Load a batch and perform feed-forward
        x, y_true =  batch_data
        # y_true = standard_scaler.transform(y_true)
        y_pred = model(x)

        # Compute loss between prediction and ground-truth
        loss = mse(y_true, y_pred)
        mse_train_epoch.append(loss.item())

        # Perform back-propagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    mse_train_overall.append(np.mean(mse_train_epoch))

    # At the end of an epoch, evaluate the model on the development set. It could be used to stop training when we diagnose over-fitting.
    mse_dev_epoch = []
    for i, batch_data in enumerate(dataloader_dev):
        # Load a batch and perform feed-forward
        x, y_true =  batch_data
        # y_true = standard_scaler.transform(y_true)
        y_pred = model(x)

        # Compute loss between prediction and ground-truth
        mse_dev_epoch.append(mse(y_true, y_pred).item())
    mse_dev_overall.append(np.mean(mse_dev_epoch))

    # Print training progress
    print(f"Epoch {epoch}/{nb_epoch}, mse_train={mse_train_overall[-1]}, mse_dev={mse_dev_overall[-1]}")

# Once the architecture has been hand-crafted over multiple tries and evaluated with the dev-set, the developper might have 
# manually "over-fitted" the solution on the dev-set. We can then evaluate the final performances of the system on the test-set.
if evaluate_on_test_set:
    mse_test = []
    for i, batch_data in enumerate(dataloader_eval):
        # Load a batch and perform feed-forward
        x, y_true =  batch_data
        # y_true = standard_scaler.transform(y_true)
        y_pred = model(x)

        # Compute loss between prediction and ground-truth
        mse_test.append(mse(y_true, y_pred).item())
    mse_test = np.mean(mse_test)

print(x, y_true, y_pred)


# Plot the training progress and saves it as a file
fig, ax = plt.subplots()
x = [i for i in range(1,nb_epoch+1)]
ax.plot(x, mse_train_overall, label="mse_train")
ax.plot(x, mse_dev_overall, label="mse_dev")
if evaluate_on_test_set:
    ax.plot(x, [mse_test]*len(x), label="mse_test")
ax.legend()
plt.savefig(save_plot_path)