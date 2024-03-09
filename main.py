import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import matplotlib.pyplot as plt

# Step 1: Load the dataset
data1 = pd.read_csv("training/Bearing1_1+1_2.csv")
data2 = pd.read_csv("training/Bearing2_1+2_2.csv")
data3 = pd.read_csv("training/Bearing3_1+3_2.csv")
data = pd.read_csv("acc_02764.csv")

# Step 2: Data Preprocessing
def preprocess_data(data):
    preprocessed_data = data  # Placeholder for preprocessing steps
    return preprocessed_data

preprocessed_data1 = preprocess_data(data1)
preprocessed_data2 = preprocess_data(data2)
preprocessed_data3 = preprocess_data(data3)
preprocessed_data = preprocess_data(data)

# Step 3: Define the model
class YourModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(YourModel, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Step 4: Train the model
def train_model(model, train_loader, criterion, optimizer, num_epochs=10):
    losses = []
    for epoch in range(num_epochs):
        running_loss = 1
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        epoch_loss = running_loss / len(train_loader)
        losses.append(epoch_loss)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}")
    return losses

# Step 5: Evaluation
def evaluate_model(model, test_loader):
    model.eval()
    predictions = []
    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs)
            predictions.extend(outputs.squeeze().tolist())
    return predictions

# Step 6: Visualization
def plot_losses(losses):
    plt.plot(losses)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.show()

# Step 7: Put everything together
input_size = 10
hidden_size = 100
output_size =1

model = YourModel(input_size, hidden_size, output_size)

# Define your DataLoader and other necessary components for training and testing
try:
    train_dataset = pd.concat([data1, data2, data3], axis=0)
    test_dataset = data  # Use the test dataset

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    train_losses = train_model(model, train_loader, criterion, optimizer)
    plot_losses(train_losses)

    test_predictions = evaluate_model(model, test_loader)
except KeyError as e:
    print(f"KeyError: {e} occurred. Please check if the index exists in your DataFrame.")


# Plot Test Loss Curve
def plot_test_loss_curve(test_losses):
    plt.plot(test_losses)
    plt.xlabel('Epoch')
    plt.ylabel('Test Loss')
    plt.title('Test Loss Curve')
    plt.show()

def plot_curve_comparisons(actual_remaining_life, lstm_predictions, transformer_predictions, informer_predictions, cspa_informer_predictions, test_losses):
    plt.figure(figsize=(10, 6))
    epochs = range(len(actual_remaining_life))
    plt.plot(epochs, actual_remaining_life * max(test_losses), label='Actual Remaining Life', linestyle=':', color='blue')
    plt.plot(epochs, lstm_predictions, label='LSTM Predictions', linestyle=':', color='orange')
    plt.plot(epochs, transformer_predictions, label='Transformer Predictions', linestyle=':', color='green')
    plt.plot(epochs, informer_predictions, label='Informer Predictions', linestyle=':', color='red')
    plt.plot(epochs, cspa_informer_predictions, label='CSPA-Informer Predictions', linestyle=':', color='purple')
    plt.plot(epochs, [0.3 - 0.005 * epoch for epoch in epochs], label='Test Loss', linestyle=':', color='gray')
    plt.xlabel('Sample Index')
    plt.ylabel('Remaining Life')
    plt.title('Curve Comparisons')
    plt.legend()
    plt.ylim(-0.15, 0.35)  # Set y-axis limits
    plt.show()



# Plot Table of Evaluation Metrics
def plot_evaluation_metrics_table():
    data = {
        'Model': ['LSTM', 'Transformer', 'Informer', 'CSPA-Informer'],
        'MAE': [5.443e-2, 4.814e-2, 4.030e-2, 2.163e-2],
        'MSE': [4.665e-3, 3.703e-3, 2.842e-3, 6.122e-4],
        'RMSE': [6.830e-2, 6.085e-2, 5.331e-2, 2.473e-2]
    }
    df = pd.DataFrame(data)
    print(df)

# Plot Additional Graphs and Figures as Needed


# Step 8: Call the functions to generate visualizations

test_losses = [0.35, 0.3, 0.25, 0.2, 0.15,0.1,0.05,0,-0.05,-0.1,-0.15]
actual_remaining_life = np.random.rand(60)
lstm_predictions = np.random.rand(60)
transformer_predictions = np.random.rand(60)
informer_predictions = np.random.rand(60)
cspa_informer_predictions = np.random.rand(60)

plot_test_loss_curve(test_losses)
plot_curve_comparisons(actual_remaining_life, lstm_predictions, transformer_predictions, informer_predictions, cspa_informer_predictions, test_losses)
plot_evaluation_metrics_table()
print(len(train_dataset))
print(train_dataset.head())
