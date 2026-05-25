import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
from torch_geometric.utils import dense_to_sparse
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from haversine import haversine
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Step 1: Load Data
data_features = pd.read_csv('/root/09 四审实验/WaterQuality-DCGCN-Mamba/data/data.csv', encoding='gbk')  # Load the data from CSV containing features and labels
stations_data = pd.read_csv('/root/09 四审实验/WaterQuality-DCGCN-Mamba/data/Site location number.csv', encoding='gbk')  # Load the data from CSV containing station locations

# Extract feature and label columns
features = data_features.iloc[:, :-1].values
labels = data_features.iloc[:, -1].values

# Extract station coordinates
stations_coords = stations_data[['纬度', '经度']].values

# Step 2: Preprocess Data
scaler = StandardScaler()
features = scaler.fit_transform(features)

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

# Step 3: Build Graph (Using Haversine Formula)
n_stations = stations_coords.shape[0]
adjacency_matrix = np.zeros((n_stations, n_stations))
thresh_distance = 85  # Threshold distance in km for connecting nodes

# Calculate adjacency matrix based on distance
for i in range(n_stations):
    for j in range(i + 1, n_stations):
        dist = haversine(tuple(stations_coords[i]), tuple(stations_coords[j]))
        if dist <= thresh_distance:
            adjacency_matrix[i, j] = 1
            adjacency_matrix[j, i] = 1

# Convert adjacency matrix to edge_index (PyG format)
edge_index, _ = dense_to_sparse(torch.tensor(adjacency_matrix))

# Step 4: Define the Model Classes
class MambaModule(nn.Module):
    def __init__(self, input_dim):
        super(MambaModule, self).__init__()
        self.linear1 = nn.Linear(input_dim, input_dim)
        self.linear2 = nn.Linear(input_dim, input_dim)
        self.convolution = nn.Conv1d(in_channels=input_dim, out_channels=input_dim, kernel_size=3, padding=1,
                                     groups=input_dim)
        self.ssm = nn.Linear(input_dim, input_dim)  # Placeholder for State Space Model (SSM)
        self.projection = nn.Linear(input_dim, input_dim)
        self.activation = nn.SiLU()

    def forward(self, x):
        # Linear Projection and Convolution Path
        proj1 = self.linear1(x)  # Shape: [batch_size, input_dim]

        # Reshape for Conv1d (requires shape [batch_size, channels, length])
        conv_input = proj1.unsqueeze(2)  # Shape: [batch_size, input_dim, 1]
        conv_out = self.convolution(conv_input).squeeze(2)  # Apply Conv1d and squeeze back to [batch_size, input_dim]

        si_out1 = self.activation(conv_out)
        ssm_out = self.ssm(si_out1)

        # Linear Projection Only Path
        proj2 = self.linear2(x)
        si_out2 = self.activation(proj2)

        # Combine Paths
        combined = ssm_out + si_out2
        out = self.projection(combined)
        return out


class BiConvLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size):
        super(BiConvLSTM, self).__init__()
        self.forward_lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True, bidirectional=False)
        self.backward_lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True, bidirectional=False)
        self.hidden_dim = hidden_dim

    def forward(self, x):
        # Assuming x shape is [batch_size, seq_len, input_dim]
        # Forward LSTM
        forward_out, _ = self.forward_lstm(x)
        # Backward LSTM (Reverse the sequence)
        backward_input = torch.flip(x, dims=[1])
        backward_out, _ = self.backward_lstm(backward_input)
        # Reverse the output of the backward LSTM to original order
        backward_out = torch.flip(backward_out, dims=[1])
        # Concatenate forward and backward outputs
        combined = torch.cat([forward_out, backward_out], dim=-1)
        return combined


class GCN_Mamba_BiConvLSTM_Model(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GCN_Mamba_BiConvLSTM_Model, self).__init__()
        self.embedding_1 = nn.Linear(input_dim, hidden_dim)
        self.embedding_2 = nn.Linear(input_dim, hidden_dim)

        # First GCN, Mamba, and BiConvLSTM branches
        self.gcn1 = GCNConv(hidden_dim, hidden_dim)
        self.mamba1 = MambaModule(hidden_dim)
        self.bi_conv_lstm1 = BiConvLSTM(hidden_dim, hidden_dim, kernel_size=(3, 3))

        self.reduce_dim_1 = nn.Linear(hidden_dim * 3, hidden_dim)  # Reduce concatenated dimension to hidden_dim

        # Second GCN, Mamba, and BiConvLSTM branches
        self.gcn2 = GCNConv(hidden_dim, hidden_dim)
        self.mamba2 = MambaModule(hidden_dim)
        self.bi_conv_lstm2 = BiConvLSTM(hidden_dim, hidden_dim, kernel_size=(3, 3))

        self.reduce_dim_2 = nn.Linear(hidden_dim * 4, hidden_dim)  # Reduce concatenated dimension to hidden_dim

        # Final layers
        self.p1_layer = nn.Linear(hidden_dim, hidden_dim)
        self.p2_layer = nn.Linear(hidden_dim + input_dim, hidden_dim)  # Concatenate P1 output and original input
        self.output_layer = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, edge_index):
        # Ensure data is on the correct device
        x = x.to(device)
        edge_index = edge_index.to(device)

        # Embedding Layers
        e1 = self.embedding_1(x)
        e2 = self.embedding_2(x)

        # GCN Layer 1 + Mamba Layer 1 + BiConvLSTM 1
        gcn1_out = self.gcn1(e2, edge_index)
        mamba1_out = self.mamba1(gcn1_out)
        bi_lstm1_out = self.bi_conv_lstm1(mamba1_out.unsqueeze(1)).squeeze(1)  # BiConvLSTM expects a sequence, add sequence dimension

        # Concatenate e2, mamba1_out, gcn1_out
        concat_1 = torch.cat([e2, mamba1_out, gcn1_out], dim=1)

        # Reduce dimension to match next GCN input
        reduced_1 = self.reduce_dim_1(concat_1)

        # GCN Layer 2 + Mamba Layer 2 + BiConvLSTM 2
        gcn2_out = self.gcn2(reduced_1, edge_index)
        mamba2_out = self.mamba2(gcn2_out)
        bi_lstm2_out = self.bi_conv_lstm2(mamba2_out.unsqueeze(1)).squeeze(1)

        # Concatenate e2, mamba1_out, gcn1_out, and mamba2_out
        concat_2 = torch.cat([e2, mamba1_out, gcn1_out, mamba2_out], dim=1)

        # Reduce dimension to match next layer input
        reduced_2 = self.reduce_dim_2(concat_2)

        # P1 Layer
        p1_out = self.p1_layer(reduced_2)

        # Concatenate P1 output and original input, then reduce dimension
        p1_concat = torch.cat([p1_out, x], dim=1)

        # P2 Layer
        p2_out = self.p2_layer(p1_concat)

        # Output Layer
        output = self.output_layer(p2_out)

        return output


# Step 5: Prepare Data for PyTorch Geometric
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Move model to the appropriate device
model = GCN_Mamba_BiConvLSTM_Model(input_dim=X_train.shape[1], hidden_dim=64, output_dim=1).to(device)

# Prepare training and test data
x_train_tensor = torch.tensor(X_train, dtype=torch.float).to(device)
y_train_tensor = torch.tensor(y_train, dtype=torch.float).to(device).view(-1, 1)
x_test_tensor = torch.tensor(X_test, dtype=torch.float).to(device)
y_test_tensor = torch.tensor(y_test, dtype=torch.float).to(device).view(-1, 1)

train_data = Data(x=x_train_tensor, edge_index=edge_index.to(device), y=y_train_tensor)
test_data = Data(x=x_test_tensor, edge_index=edge_index.to(device), y=y_test_tensor)

# Step 6: Train the Model
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.MSELoss()

def train(model, data):
    model.train()
    optimizer.zero_grad()
    output = model(data.x, data.edge_index)
    loss = criterion(output, data.y)
    loss.backward()
    optimizer.step()
    return loss.item()

# Train the model for 1000 epochs
for epoch in range(1, 10001):
    loss = train(model, train_data)
    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Loss: {loss}")

# Step 7: Evaluate the Model
def evaluate(model, x_test, y_test):
    model.eval()
    with torch.no_grad():
        predictions = model(x_test, edge_index)

        # 将预测值和真实值移到CPU，并转换为NumPy数组
        predictions_cpu = predictions.cpu().numpy()
        y_test_cpu = y_test.cpu().numpy()

        # 计算评估指标
        mse = mean_squared_error(y_test_cpu, predictions_cpu)
        mae = mean_absolute_error(y_test_cpu, predictions_cpu)
        r2 = r2_score(y_test_cpu, predictions_cpu)

        # Create DataFrame to save results
        results_df = pd.DataFrame({
            'True Values': y_test_cpu.flatten(),
            'Predictions': predictions_cpu.flatten()
        })
        results_df.to_csv('real_vs_predicted.csv', index=False)  # Save to CSV

        # Plot results (Scatter plot with updated style)
        plt.figure(figsize=(8, 6))
        plt.scatter(y_test_cpu, predictions_cpu, color='dodgerblue', alpha=0.7, s=40, edgecolors='k', linewidth=0.5)
        plt.plot([min(y_test_cpu), max(y_test_cpu)], [min(y_test_cpu), max(y_test_cpu)], color='darkorange', linestyle='-', linewidth=2)
        plt.xlabel('True Values', fontsize=12)
        plt.ylabel('Predictions', fontsize=12)
        plt.title('True vs Predicted Values (Scatter Plot)', fontsize=14, fontweight='bold')
        plt.grid(color='lightgray', linestyle='--', linewidth=0.5)
        plt.savefig('updated_true_vs_predicted_scatter_plot.png')  # Save updated scatter plot as PNG
        plt.show()

        # Plot results (Line plot with updated style)
        plt.figure(figsize=(10, 6))
        plt.plot(y_test_cpu, color='teal', label='True Values', linewidth=2, alpha=0.8)
        plt.plot(predictions_cpu, color='crimson', linestyle='--', label='Predictions', linewidth=2, alpha=0.8)
        plt.xlabel('Sample Index', fontsize=12)
        plt.ylabel('Values', fontsize=12)
        plt.title('True vs Predicted Values (Line Plot)', fontsize=14, fontweight='bold')
        plt.legend(fontsize=12)
        plt.grid(color='lightgray', linestyle='--', linewidth=0.5)
        plt.savefig('updated_true_vs_predicted_line_plot.png')  # Save updated line plot as PNG
        plt.show()

    return predictions, mse, mae, r2

# Evaluate the model
predictions, mse, mae, r2 = evaluate(model, x_test_tensor, y_test_tensor)

# Output evaluation results
print(f"MSE: {mse}, MAE: {mae}, R2: {r2}")
