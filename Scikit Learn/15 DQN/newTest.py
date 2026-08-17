import torch
import torch.nn as nn
import torch.optim as optim


# --------------------------------------------------
# DATA
# --------------------------------------------------

X = torch.tensor([
    [800,  2, 10],
    [1000, 2,  5],
    [1200, 3,  8],
    [1500, 3, 4],
    [1800, 4, 6],
    [2200, 4, 3],
    [2500, 5, 2],
    [3000, 5, 1]
], dtype=torch.float32)


y = torch.tensor([
    [35],
    [48],
    [60],
    [78],
    [95],
    [120],
    [140],
    [170]
], dtype=torch.float32)


# --------------------------------------------------
# NEURAL NETWORK
# --------------------------------------------------

model = nn.Sequential(
    nn.Linear(3, 8),
    nn.ReLU(),
    nn.Linear(8, 8),
    nn.ReLU(),
    nn.Linear(8, 1)
)


# --------------------------------------------------
# LOSS FUNCTION
# --------------------------------------------------

loss_function = nn.MSELoss()


# --------------------------------------------------
# OPTIMIZER
# --------------------------------------------------

optimizer = optim.Adam(
    model.parameters(),
    lr=0.001
)


# --------------------------------------------------
# TRAINING
# --------------------------------------------------

for epoch in range(5000):

    # Forward pass
    prediction = model(X)

    # Calculate loss
    loss = loss_function(prediction, y)

    # Remove old gradients
    optimizer.zero_grad()

    # Backpropagation
    loss.backward()

    # Update weights
    optimizer.step()

    # Display progress
    if (epoch + 1) % 500 == 0:
        print(
            f"Epoch: {epoch + 1:4d} | "
            f"Loss: {loss.item():.4f}"
        )


# --------------------------------------------------
# TESTING
# --------------------------------------------------

test_house = torch.tensor(
    [[1600, 3, 5]],
    dtype=torch.float32
)


predicted_price = model(test_house)


print("\nTraining Complete!")

print(
    f"Predicted Price: "
    f"₹{predicted_price.item():.2f} lakh"
)