import torch
import torch.nn as nn
import torch.optim as optim


# INPUT
x = torch.tensor([2, 3], dtype=torch.float32)


# NEURAL NETWORK
model = nn.Sequential(
    nn.Linear(2, 4),
    nn.ReLU(),
    nn.Linear(4, 1)
)


# CORRECT ANSWER
target = torch.tensor([1.0])


# LOSS FUNCTION
loss_function = nn.MSELoss()


# OPTIMIZER
optimizer = optim.Adam(
    model.parameters(),
    lr=0.01
)


# TRAINING
for epoch in range(1000):

    # FORWARD PASS
    prediction = model(x)

    # CALCULATE LOSS
    loss = loss_function(
        prediction,
        target
    )

    # CLEAR OLD GRADIENTS
    optimizer.zero_grad()

    # BACKPROPAGATION
    loss.backward()

    # UPDATE WEIGHTS
    optimizer.step()

    # PRINT PROGRESS
    if (epoch + 1) % 100 == 0:

        print(
            f"Epoch: {epoch + 1:4d} | "
            f"Prediction: {prediction.item():.4f} | "
            f"Loss: {loss.item():.6f}"
        )


print("\nTraining Complete!")

print(
    "Final Prediction:",
    model(x).item()
)

print(
    "Target:",
    target.item()
)