import torch
import torch.nn as nn
import torch.optim as optim

X = torch.tensor([2.0, 4.0], dtype=torch.float32)
print("Your tensor data :\n", X)

# FIX 2: Changed [4, 1] to [4.0]. Your model's final layer (nn.Linear(4, 1)) 
# outputs 1 number, so the target must also be exactly 1 number.
y = torch.tensor([4.0], dtype=torch.float32)

# Neural Network Model
model = nn.Sequential(
    nn.Linear(2, 4),
    nn.ReLU(),
    nn.Linear(4, 1)
)

# Loss Control
loss_function = nn.MSELoss()

# Optimization
optimization = optim.Adam(model.parameters(), lr=0.01)

# Model Loop Training
# FIX 3: Changed the loop variable from 'x' to 'epoch' so it matches your print statement.
for epoch in range(1000):
    
    # Forward Pass
    prediction = model(X)
    
    # Error Estimation
    error = loss_function(prediction, y)
    
    # Optimization of Weights (Clearing old Gradients)
    # FIX 4: Changed zero.grad() to zero_grad() (underscore, not a dot).
    optimization.zero_grad()
    
    # Backpropagation
    error.backward()
    
    optimization.step()
    
    # PRINT PROGRESS
    if (epoch + 1) % 100 == 0:
        print(
            f"Epoch: {epoch + 1:4d} | "
            f"Prediction: {prediction.item():.4f} | "
            # FIX 5: Changed 'loss' to 'error' to match your variable name above.
            f"Loss: {error.item():.6f}" 
        )
        
# FIX 6: Un-indented all the code below. 
# They must be outside the loop so they only print ONCE at the very end.
print("\nTraining Complete!")

print(
    "Final Prediction:",
    # FIX 7: Changed 'x' to 'X' (Python is case-sensitive!).
    model(X).item()
)

print(
    "Target:",
    # FIX 8: Changed 'target' to 'y' to match your variable name.
    y.item()
)