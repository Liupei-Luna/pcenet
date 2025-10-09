import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.loader import DataLoader as PyGDataLoader
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_add_pool # Import GCNConv
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# --- 1. 数据加载和预处理 ---
# Ensure your CSV files are in the same directory as the script.
try:
    tra_df = pd.read_csv("train.csv", header=0)
    tra_df = tra_df.iloc[:, 2:]
    tes_df = pd.read_csv("test.csv", header=0)
    tes_df = tes_df.iloc[:, 2:]
except FileNotFoundError:
    print("Error: train.csv or test.csv not found.")
    print("Please ensure your CSV files are in the same directory as the script.")
    exit()

# Extract targets (assuming first column is target)
train_targets = tra_df.iloc[:, 0].values.astype(np.float32)
test_targets = tes_df.iloc[:, 0].values.astype(np.float32)

# Extract features: first 6 columns are receptor, rest are donor
train_receptor_features = tra_df.iloc[:, 1:7].values.astype(np.float32)  # Columns 1 to 6 (inclusive)
train_donor_features = tra_df.iloc[:, 7:].values.astype(np.float32)    # Columns 7 to end

test_receptor_features = tes_df.iloc[:, 1:7].values.astype(np.float32)
test_donor_features = tes_df.iloc[:, 7:].values.astype(np.float32)

print(f"Train Receptor Features shape: {train_receptor_features.shape}")
print(f"Train Donor Features shape: {train_donor_features.shape}")
print(f"Test Receptor Features shape: {test_receptor_features.shape}")
print(f"Test Donor Features shape: {test_donor_features.shape}")

# Standardize Donor features
scaler_donor = StandardScaler()
train_donor_features_scaled = scaler_donor.fit_transform(train_donor_features)
test_donor_features_scaled = scaler_donor.transform(test_donor_features)

# Receptor features (1-hot encoded) usually not scaled. Use as is.
train_receptor_features_processed = train_receptor_features
test_receptor_features_processed = test_receptor_features


# --- 2. 自定义数据集类：将受体-供体对转换为“二节点图” ---
# (此部分与DMPNN版本保持一致，因为它定义了图的拓扑和节点/边特征结构，
# 即使GCNConv不直接使用edge_attr，Data对象仍然可以包含它)

class ReceptorDonorGraphDataset(InMemoryDataset):
    def __init__(self, receptor_features_list, donor_features_list, targets_list, transform=None, pre_transform=None):
        super().__init__('.', transform, pre_transform)
        self.receptor_features_list = receptor_features_list
        self.donor_features_list = donor_features_list
        self.targets_list = targets_list
        # Determine the maximum node feature dimension for consistent Data.x structure
        self.max_node_feature_dim = donor_features_list.shape[1] # Should be 625 (max of 6 and 625)

        self.data, self.slices = self._process()

    def _process(self):
        data_list = []
        
        for i in range(len(self.targets_list)):
            receptor_feats_current = self.receptor_features_list[i]
            donor_feats_current = self.donor_features_list[i]
            y_val_current = self.targets_list[i]

            # Construct node feature tensor `x` with consistent dimensions (625)
            # Receptor node features: Place 6 features into a 625-dim tensor (rest are zeros)
            receptor_x = torch.zeros(self.max_node_feature_dim, dtype=torch.float)
            receptor_x[:receptor_feats_current.shape[0]] = torch.tensor(receptor_feats_current, dtype=torch.float)

            # Donor node features: Use the 625-dim features directly
            donor_x = torch.tensor(donor_feats_current, dtype=torch.float)
            
            # Stack the two node feature vectors to create x for the graph
            # Shape: (num_nodes=2, feature_dim=625)
            x_nodes = torch.stack([receptor_x, donor_x], dim=0) 
            
            # Edge index: Receptor (0) connects to Donor (1), and vice versa (bidirectional)
            edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
            
            # Edge attributes: A simple constant feature (e.g., indicating presence of interaction)
            # Although GCNConv doesn't use it directly, we keep it for Data object consistency
            edge_attr = torch.ones((edge_index.shape[1], 1), dtype=torch.float) 

            # Target value (regression task)
            y_tensor = torch.tensor([[y_val_current]], dtype=torch.float)

            # Create PyG Data object
            data = Data(x=x_nodes, edge_index=edge_index, edge_attr=edge_attr, y=y_tensor)
            data_list.append(data)
        
        if not data_list:
            raise ValueError("No valid data processed. Check input feature shapes or data length.")
            
        return self.collate(data_list)

print("\n--- Creating Training Dataset (receptor-donor graphs) ---")
train_dataset = ReceptorDonorGraphDataset(train_receptor_features_processed, train_donor_features_scaled, train_targets)
print("\n--- Creating Test Dataset (receptor-donor graphs) ---")
test_dataset = ReceptorDonorGraphDataset(test_receptor_features_processed, test_donor_features_scaled, test_targets)

# Create PyG DataLoaders
batch_size = 64 
train_loader = PyGDataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = PyGDataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Get feature dimensions for the GNN from the created dataset
node_feature_dim = train_dataset.max_node_feature_dim # 625
receptor_original_dim = train_receptor_features.shape[1] # 6
donor_original_dim = train_donor_features.shape[1]     # 625

# `edge_feature_dim` is still 1, but will not be explicitly used by GCNConv
edge_feature_dim = train_dataset[0].edge_attr.shape[1] 
output_dim = train_dataset[0].y.shape[1] # 1

print(f"Node feature dimension (unified, for x): {node_feature_dim}")
print(f"Original Receptor feature dimension: {receptor_original_dim}")
print(f"Original Donor feature dimension: {donor_original_dim}")
print(f"Edge feature dimension (will be ignored by GCNConv): {edge_feature_dim}") 
print(f"Output dimension: {output_dim}")


# --- 3. 定义通用的MPNN模型 (使用GCNConv) ---
# GCNConv不使用边特征，因此我们将移除GCNConv层中对edge_dim的依赖
# 且GCNConv本身就集成了消息传递和更新逻辑，不需要手动定义Message/Update
class GeneralMPNN(nn.Module):
    def __init__(self, receptor_input_dim, donor_input_dim, hidden_channels=128, num_layers=3, output_dim=1, dropout_rate=0.3):
        super().__init__()
        self.dropout_rate = dropout_rate
        self.num_layers = num_layers
        self.hidden_channels = hidden_channels

        # Initial MLPs to project different node types to a common hidden space
        self.receptor_lin = nn.Linear(receptor_input_dim, hidden_channels)
        self.donor_lin = nn.Linear(donor_input_dim, hidden_channels)

        # GCNConv layers for message passing
        self.convs = nn.ModuleList()
        # The first GCNConv takes input from initial embeddings
        # All subsequent layers take input from previous layer's output (all hidden_channels)
        for _ in range(num_layers):
            # GCNConv simply needs in_channels and out_channels
            # It internally handles aggregation and linear transformation
            self.convs.append(GCNConv(hidden_channels, hidden_channels))

        # Final readout MLP for prediction
        self.readout_mlp = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.ReLU(),
            nn.Linear(hidden_channels // 2, output_dim) 
        )
        
    def forward(self, data):
        x_raw, edge_index = data.x, data.edge_index # GCNConv does not need edge_attr
        batch = data.batch

        # Initialize node embeddings based on their type
        initial_node_embeddings = torch.zeros(x_raw.size(0), self.hidden_channels).to(x_raw.device)
        
        # Identify Receptor and Donor nodes within the batch (same logic as before)
        receptor_indices = torch.arange(0, x_raw.size(0), 2, device=x_raw.device)
        donor_indices = torch.arange(1, x_raw.size(0), 2, device=x_raw.device)

        # Apply type-specific linear transformations
        initial_node_embeddings[receptor_indices] = self.receptor_lin(x_raw[receptor_indices, :receptor_original_dim])
        initial_node_embeddings[donor_indices] = self.donor_lin(x_raw[donor_indices, :donor_original_dim])

        # This `initial_node_embeddings` becomes the input `h^0` for message passing
        x = initial_node_embeddings 

        # Message Passing Layers (L layers) using GCNConv
        for conv_layer in self.convs:
            # GCNConv's forward takes x (node features) and edge_index
            x = conv_layer(x, edge_index) 
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout_rate, training=self.training)

        # Global Pooling (Readout): Aggregate node embeddings for each graph
        # For this 2-node graph, it sums the final receptor and donor embeddings.
        pooled_graph_embedding = global_add_pool(x, batch) 
        
        # Final Prediction through MLP
        prediction = self.readout_mlp(pooled_graph_embedding)
        return prediction

# --- 4. 实例化模型 ---
model = GeneralMPNN(
    receptor_input_dim=receptor_original_dim, # 6
    donor_input_dim=donor_original_dim,       # 625
    hidden_channels=128, 
    num_layers=4, 
    output_dim=output_dim, 
    dropout_rate=0.3
).to(device)
print("\n--- Model Architecture (General MPNN with GCNConv) ---")
print(model)

# Define loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# --- 5. 模型训练 ---
epochs = 300 # Adjusted epochs
train_losses = []
val_losses = []
best_val_loss = float('inf')
patience = 50 
current_patience = 0

print("\n--- Model Training ---")
for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        
        outputs = model(data) # Pass the whole data object
        loss = criterion(outputs, data.y)

        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * data.num_graphs
    
    epoch_loss = running_loss / len(train_loader.dataset)
    train_losses.append(epoch_loss)

    # Validation phase
    model.eval()
    val_running_loss = 0.0
    with torch.no_grad():
        for data_val in test_loader:
            data_val = data_val.to(device)
            val_outputs = model(data_val) # Pass the whole data object
            val_loss = criterion(val_outputs, data_val.y)
            val_running_loss += val_loss.item() * data_val.num_graphs
    
    epoch_val_loss = val_running_loss / len(test_loader.dataset)
    val_losses.append(epoch_val_loss)

    print(f"Epoch [{epoch+1}/{epochs}], Train Loss: {epoch_loss:.4f}, Val Loss: {epoch_val_loss:.4f}")

    if epoch_val_loss < best_val_loss:
        best_val_loss = epoch_val_loss
        current_patience = 0
        torch.save(model.state_dict(), 'best_general_mpnn_model.pth') # Save with new name
    else:
        current_patience += 1
        if current_patience >= patience:
            print(f"Early stopping at epoch {epoch+1} due to no improvement in validation loss.")
            break

# Load best model weights
model.load_state_dict(torch.load('best_general_mpnn_model.pth'))
print("Loaded best model weights.")

# --- 6. 模型预测和评估 ---
def evaluate_model(model_instance, loader, device, name=""):
    model_instance.eval()
    y_pred_list = []
    y_true_list = []
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            outputs = model_instance(data)
            y_pred_list.append(outputs.cpu().numpy())
            y_true_list.append(data.y.cpu().numpy())

    y_pred = np.vstack(y_pred_list).flatten()
    y_true = np.vstack(y_true_list).flatten()

    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    
    residuals = y_true - y_pred
    se = np.std(residuals) # Standard Error of the Residuals
    
    print(f"\n--- {name} Evaluation ---")
    print(f"R^2 Score: {r2:.4f}")
    print(f"MAE: {mae:.4f}")
    print(f"Standard Deviation of Residuals (SE): {se:.4f}")
    return r2, mae, se, y_true, y_pred

# Evaluate on training and test sets
r2_train, mae_train, se_train, y_true_train, y_pred_train = evaluate_model(model, train_loader, device, "Training Set")
r2_test, mae_test, se_test, y_true_test, y_pred_test = evaluate_model(model, test_loader, device, "Test Set")


# --- 7. 可视化 ---
# Plot training history
plt.figure(figsize=(10, 6))
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss (MSE)')
plt.title('Model Training History (General MPNN with GCNConv)')
plt.legend()
plt.grid(True)
plt.show()

# Plot actual vs. predicted for test set
plt.figure(figsize=(10, 6))
plt.scatter(y_true_test, y_pred_test, alpha=0.7)
plt.plot([y_true_test.min(), y_true_test.max()], [y_true_test.min(), y_true_test.max()], 'r--', lw=2) # y=x line
plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
plt.title(f"Actual vs. Predicted Values (Test Set - General MPNN)\nR^2: {r2_test:.4f}, MAE: {mae_test:.4f}")
plt.grid(True)
plt.tight_layout()
plt.show()

true_values = y_true_test
predicted_values = y_pred_test

# 绘制散点图
plt.figure(figsize=(8, 8))
plt.scatter(true_values, predicted_values, color='blue')
# 绘制y=x线
min_val = min(min(true_values), min(predicted_values))
max_val = max(max(true_values), max(predicted_values))
max_val = 5
plt.plot([min_val, max_val], [min_val, max_val], 'black')

# 设置坐标轴范围
plt.xlim(0, max_val)
plt.ylim(0, max_val)

# 添加图例和标签
plt.xlabel('Experimental PCE/%')
plt.ylabel('Predicted PCE/%')
# plt.title('Scatter Plot of True vs Predicted Values')
# plt.legend(loc='lower right')

# 显示图形
plt.show()