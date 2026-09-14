"""
This code is directly imported from chatgpt codex without any editing and its works fine
"""
import torch
import numpy as np
from test_network import dw1, dw2
import pandas as pd

# ============================================================
# 1. RANDOM INITIALIZATION
# ============================================================

np.random.seed(0)

W1 = np.random.randn(64, 784) * 0.1
b1 = np.random.randn(64, 1) * 0.1

W2 = np.random.randn(10, 64) * 0.1
b2 = np.random.randn(10, 1) * 0.1


# ============================================================
# 2. CONVERT NUMPY PARAMETERS TO PYTORCH TENSORS
# ============================================================

W1_t = torch.tensor(
    W1,
    dtype=torch.float64,
    requires_grad=True
)

b1_t = torch.tensor(
    b1,
    dtype=torch.float64,
    requires_grad=True
)

W2_t = torch.tensor(
    W2,
    dtype=torch.float64,
    requires_grad=True
)

b2_t = torch.tensor(
    b2,
    dtype=torch.float64,
    requires_grad=True
)


# ============================================================
# 3. LOAD MNIST DATA
# ============================================================

data = pd.read_csv("mnist_test.csv")
data = np.array(data)

training_data = data[1000:].T

Y = training_data[0]
X = training_data[1:785]

# Normalize pixel values
X = X / 255.0


# ============================================================
# 4. CONVERT INPUTS TO PYTORCH
# ============================================================

X_t = torch.tensor(
    X,
    dtype=torch.float64
)

y_t = torch.tensor(
    Y,
    dtype=torch.long
)


# ============================================================
# 5. FORWARD PASS
# ============================================================

# X_t shape:
# (784, number_of_examples)

# Layer 1
z1 = W1_t @ X_t + b1_t

# ReLU activation
a1 = torch.relu(z1)

# Output layer
z2 = W2_t @ a1 + b2_t

# z2 shape:
# (10, number_of_examples)


# ============================================================
# 6. CROSS ENTROPY LOSS
# ============================================================

# PyTorch expects:
#
# input  -> (number_of_examples, number_of_classes)
# target -> (number_of_examples)
#
# Therefore transpose z2

logits = z2.transpose(0, 1).contiguous()

print("logits shape:", logits.shape)

loss = torch.nn.functional.cross_entropy(
    logits,
    y_t,
    reduction="mean"
)


# ============================================================
# 7. BACKPROPAGATION USING AUTOGRAD
# ============================================================

loss.backward()


# ============================================================
# 8. EXTRACT GRADIENTS
# ============================================================

torch_dW1 = W1_t.grad.detach().cpu().numpy()
torch_db1 = b1_t.grad.detach().cpu().numpy()

torch_dW2 = W2_t.grad.detach().cpu().numpy()
torch_db2 = b2_t.grad.detach().cpu().numpy()




# ============================================================
# 9. CHECK RESULTS
# ============================================================

print("Loss:", loss.item())

print("\nGradient differences")

print("dW1:", torch_dW1 - dw1)
print("db1:", torch_db1.shape)

print("dW2:", torch_dW2 - dw2)
print("db2:", torch_db2.shape)