#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 10 16:10:17 2025

@author: lzxvc
"""


import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np


def weighted_mse_loss(pred, target, alpha=1.0, eps=1e-8):
    weights = 1.0 + alpha * target
    
    # Weighted sum of squared errors
    mse = weights * (pred - target)**2
    
    # Normalize by the sum of weights
    wmse = mse.sum() / (weights.sum() + eps)
    return wmse


# Check MPS availability (for Mac Studio w/ Apple Silicon)
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Using MPS device")
else:
    device = torch.device("cpu")
    print("MPS not available, using CPU")
    

# ---------------------------------------------------------------------------------------------------  
#                                          Training Set
# --------------------------------------------------------------------------------------------------- 
X = np.load('/Users/lzxvc/Downloads/OneDrive_1_1-28-2025/ll_1t_X_lhs_100000000.npy')
Y = np.load('/Users/lzxvc/Downloads/OneDrive_1_1-28-2025/ll_1t_Y_lhs_100000000.npy')
X = X.astype(np.float32)
Y_log = np.log(10-Y).astype(np.float32)


# Convert to PyTorch tensors
X_torch = torch.from_numpy(X)
y_torch = torch.from_numpy(Y_log).unsqueeze(1)  # shape (N, 1)



# ---------------------------------------------------------------------------------------------------  
#                                        Validation Set
# --------------------------------------------------------------------------------------------------- 
X_new = np.load('/Users/lzxvc/Downloads/OneDrive_1_1-28-2025/ll_1t_X_lhs_val_1000000.npy')
Y_new = np.load('/Users/lzxvc/Downloads/OneDrive_1_1-28-2025/ll_1t_Y_lhs_val_1000000.npy')
X_new = X_new.astype(np.float32)
Y_new_log = np.log(10-Y_new).astype(np.float32)



# Convert to PyTorch tensors
X_new_torch = torch.from_numpy(X_new)
y_new_torch = torch.from_numpy(Y_new_log).unsqueeze(1)  # shape (N, 1)




train_dataset = TensorDataset(X_torch, y_torch)
val_dataset   = TensorDataset(X_new_torch, y_new_torch)
batch_size = 8192

train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=8,       # Increase if CPU cores are available
    pin_memory=True      # Potentially improve GPU transfer (may help on MPS)
)

val_loader = DataLoader(
    val_dataset,
    batch_size=batch_size,
    shuffle=False,   # usually no need to shuffle validation
    num_workers=8,       # Increase if CPU cores are available
    pin_memory=True      # Potentially improve GPU transfer (may help on MPS)
)



class MLP(nn.Module):
    def __init__(self, input_dim=9, hidden_dim=512, output_dim=1, num_layers=5):
        super(MLP, self).__init__()
        layers = []

        # First layer
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.Tanh())

        # Additional hidden layers
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.Tanh())

        # Final output layer
        layers.append(nn.Linear(hidden_dim, output_dim))

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

model = MLP().to(device)



loss_fn = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-5, weight_decay=1e-7)


# ---------------------------------------------------------------------------------------------------  
#                                          Model Training
# --------------------------------------------------------------------------------------------------- 
num_epochs = 200


for epoch in range(num_epochs):
    #######################
    # Training Mode
    #######################
    model.train()
    total_loss = 0.0
    
    for batch_idx, (features, targets) in enumerate(train_loader):
        # Move data to MPS (or CPU if MPS not available)
        features = features.to(device)
        targets = targets.to(device)

        # Forward pass
        predictions = model(features)
        loss = weighted_mse_loss(predictions, targets, alpha=1)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * features.size(0)

        # Print progress occasionally
        if (batch_idx + 1) % 1000 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], "
                  f"Step [{batch_idx+1}/{len(train_loader)}], "
                  f"Loss: {loss.item():.4f}")

    # Average loss over entire dataset
    avg_epoch_loss = total_loss / len(train_loader.dataset)
    
    
    #######################
    # Validation Mode
    #######################
    model.eval()
    running_val_loss = 0.0
    
    with torch.no_grad():
        for features, targets in val_loader:
            features = features.to(device)
            targets = targets.to(device)
            
            preds = model(features)
            loss = weighted_mse_loss(preds, targets, alpha=1)
            running_val_loss += loss.item() * features.size(0)
    
    epoch_val_loss = running_val_loss / len(val_dataset)

    print(f"Epoch [{epoch+1}/{num_epochs}] | "
          f"Train Loss: {avg_epoch_loss:.4f} | "
          f"Val Loss: {epoch_val_loss:.4f}")





# ---------------------------------------------------------------------------------------------------  
#                                          Model Prediction
# ---------------------------------------------------------------------------------------------------    
model.eval()

with torch.no_grad():
    y_pred = model(X_new_torch.to(device))

print("Predictions:", y_pred.cpu().numpy().flatten())
loss_fn(y_pred.cpu(), y_new_torch)   
    
    





x1 = np.linspace(0.9, 0.999, num=200, dtype=np.float32)
x2 = np.full(shape=(200,), fill_value=0.6, dtype=np.float32)
x3 = np.full(shape=(200,), fill_value=1.0, dtype=np.float32)
x4 = np.full(shape=(200,), fill_value=2.0, dtype=np.float32)
X_new_sub = np.column_stack([x1, x2, x3, x4])
X_new_sub = X_new_sub.astype(np.float32)
X_new_sub_torch = torch.from_numpy(X_new_sub)


X_new_sub = np.load('/Users/lzxvc/Downloads/OneDrive_1_1-28-2025/Y_ll_X_input_Nt_Ns.npy', allow_pickle = True)


X_new_sub_one = X_new_sub[0,:,:]
X_new_sub_one = X_new_sub_one.astype(np.float32)
X_new_sub_one_torch = torch.from_numpy(X_new_sub_one)

with torch.no_grad():
    y_sub_pred = model(X_new_sub_one_torch.to(device))





Y_new_sub = np.load('/Users/lzxvc/Downloads/OneDrive_1_1-28-2025/Y_new_sub.npy')






Y_new_sub = np.load('/Users/lzxvc/Downloads/OneDrive_1_1-28-2025/Y_ll_Y_Nt_Ns.npy')

np.log(10-Y_new_sub).astype(np.float32)

import matplotlib.pyplot as plt

x=np.log(10-Y_new_sub).astype(np.float32)
y= y_sub_pred.cpu().numpy().flatten()



# Create a scatter plot
plt.scatter(x, y, alpha=0.5, label='Data points')

# Draw a 1:1 line
# We'll determine the min and max from both x and y so the line spans the plot range
min_val = min(x.min(), y.min())
max_val = max(x.max(), y.max())
plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='1:1 line')

# Set the axes to have identical ranges
plt.xlim(min_val, 3)
plt.ylim(min_val, 3)

plt.xlabel('True')
plt.ylabel('Emulated')
plt.legend()
plt.title('Scatter Plot with 1:1 Line')
plt.show()









# ----- log scale ------
import matplotlib.pyplot as plt
plt.plot(x1, y_sub_pred.cpu().numpy().flatten(), marker='o', color='b', linewidth=2)
plt.plot(x1, np.log(Y_new_sub), marker='o', color='r', linewidth=2)
plt.title("0.6, 1.0, 2.0")
plt.xlabel("p")
plt.ylabel("quantile")
plt.legend()
plt.grid(True)
plt.show()


# ----- original scale ------
import matplotlib.pyplot as plt
plt.plot(x1, np.exp(y_sub_pred.cpu().numpy().flatten()), marker='o', color='b', linewidth=2)
plt.plot(x1, Y_new_sub, marker='o', color='r', linewidth=2)
plt.title("0.6, 1.0, 2.0")
plt.xlabel("p")
plt.ylabel("quantile")
plt.legend()
plt.grid(True)
plt.show()



y_new_sub_torch = torch.from_numpy(np.log(Y_new_sub)).unsqueeze(1)  # shape (N, 1)
loss_fn(y_sub_pred.cpu(), y_new_sub_torch)   
