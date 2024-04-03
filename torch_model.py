from matplotlib import pyplot as plt
import torch.nn as nn
import torch
import pickle
import torch.optim as optim

class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        # Define two hidden layers and an output layer
        self.fc1 = nn.Linear(2, 10)
        self.fc2 = nn.Linear(10, 10)
        self.fc3 = nn.Linear(10, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        x = self.sigmoid(x)
        return x

# Load weights and biases from a pickle file
with open('assignment-one-test-parameters.pkl', 'rb') as f:
    weights_and_biases = pickle.load(f)

# Instantiate the neural network model
model = SimpleNN()

# Assign loaded weights and biases to the model's parameters
model.fc1.weight.data = torch.tensor(weights_and_biases['w1']).float()
model.fc1.bias.data = torch.tensor(weights_and_biases['b1']).float()
model.fc2.weight.data = torch.tensor(weights_and_biases['w2']).float()
model.fc2.bias.data = torch.tensor(weights_and_biases['b2']).float()
model.fc3.weight.data = torch.tensor(weights_and_biases['w3']).float()
model.fc3.bias.data = torch.tensor(weights_and_biases['b3']).float()

# Prepare input data and target labels
input_data = torch.tensor(weights_and_biases['inputs']).float()
targets = torch.ones((200, 1))  # Assuming all targets are 1s

# Define optimizer and loss function
optimizer = optim.SGD(model.parameters(), lr=0.01)
loss_function = nn.BCELoss(reduction="mean")

# Train the model
loss_history = []
n_epochs = 10
for i in range(n_epochs):
    output = model(input_data)
    loss = loss_function(output, targets)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    loss_history.append(loss / 2)  # Append half of the loss (for consistency)

# Print the final loss after training
print(f"Final loss: {loss_history[-1]}")

# Plot the loss history
plt.figure(figsize=(10, 6))
plt.plot([l.detach().numpy() for l in loss_history])
plt.title('Training Loss Over Time')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.grid(True)
plt.show()
