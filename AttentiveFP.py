import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt

# 设置随机种子，保证结果可复现性
torch.manual_seed(42)
np.random.seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# 1. 数据加载 (使用你提供的数据加载代码)
tra_hopv = pd.read_csv("train.csv", header=0)
tra_hopv = tra_hopv.iloc[:, 2:]

tes_hopv = pd.read_csv("test.csv", header=0)
tes_hopv = tes_hopv.iloc[:, 2:]

x_train_raw = tra_hopv.iloc[:, 7:].values.astype(np.float32)
y_train_raw = tra_hopv.iloc[:, 0].values.astype(np.float32)
x_test_raw = tes_hopv.iloc[:, 7:].values.astype(np.float32)
y_test_raw = tes_hopv.iloc[:, 0].values.astype(np.float32)

print(f"x_train_raw shape: {x_train_raw.shape}, y_train_raw shape: {y_train_raw.shape}")
print(f"x_test_raw shape: {x_test_raw.shape}, y_test_raw shape: {y_test_raw.shape}")

# 2. 数据预处理/归一化
# 假设631维特征是连续值描述符，进行标准化。
# 如果它们是二进制指纹，则不需要归一化。
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train_raw)
x_test_scaled = scaler.transform(x_test_raw)

# 将NumPy数组转换为PyTorch Tensor
x_train_tensor = torch.tensor(x_train_scaled, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train_raw, dtype=torch.float32).view(-1, 1) # Reshape to (N, 1) for regression
x_test_tensor = torch.tensor(x_test_scaled, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test_raw, dtype=torch.float32).view(-1, 1) # Reshape to (N, 1)

# 创建PyTorch DataLoader
batch_size = 32
train_dataset = TensorDataset(x_train_tensor, y_train_tensor)
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

test_dataset = TensorDataset(x_test_tensor, y_test_tensor)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False) # Test loader usually doesn't need shuffle

# 3. 定义AttentionFP风格的模型 (使用PyTorch)
class AttentionFingerprintModel(nn.Module):
    def __init__(self, input_dim):
        super(AttentionFingerprintModel, self).__init__()
        
        # 模拟注意力机制：
        # 这里我们用一个全连接层生成与输入维度相同的权重，然后用Softmax归一化
        # 接着将权重应用于另一个全连接层（可以理解为Value变换）的输出
        self.attention_weights_generator = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.Softmax(dim=-1) # 对特征维度进行softmax
        )
        
        self.value_transform = nn.Linear(input_dim, input_dim) # 变换输入，作为注意力作用的对象

        # 后续的全连接层
        self.fc1 = nn.Linear(input_dim, 256)
        self.dropout1 = nn.Dropout(0.2)
        self.fc2 = nn.Linear(256, 128)
        self.dropout2 = nn.Dropout(0.2)
        self.output_layer = nn.Linear(128, 1) # 输出一个回归值

    def forward(self, x):
        # x 是输入的指纹特征 (batch_size, input_dim)
        
        # 1. 生成注意力权重
        attention_weights = self.attention_weights_generator(x) # (batch_size, input_dim)
        
        # 2. 变换输入以生成Value
        value = self.value_transform(x) # (batch_size, input_dim)
        
        # 3. 应用注意力：将权重与Value逐元素相乘
        weighted_fingerprint = attention_weights * value # (batch_size, input_dim)
        
        # 4. 通过后续的全连接层
        x = torch.relu(self.fc1(weighted_fingerprint))
        x = self.dropout1(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout2(x)
        
        # 5. 输出预测
        output = self.output_layer(x)
        return output

# 获取特征维度
input_dim = x_train_scaled.shape[1]
model = AttentionFingerprintModel(input_dim)

# 将模型移动到合适的设备 (CPU 或 GPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
print(f"Using device: {device}")

# 定义损失函数和优化器
criterion = nn.MSELoss() # 均方误差损失
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 4. 模型训练
epochs = 200 # 增加epochs以适应PyTorch的训练循环，配合EarlyStopping
train_losses = []
val_losses = []
best_val_loss = float('inf')
patience = 20 # Early Stopping patience
current_patience = 0

print("\n--- Model Training ---")
for epoch in range(epochs):
    model.train() # 设置模型为训练模式
    running_loss = 0.0
    for batch_x, batch_y in train_loader:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        
        optimizer.zero_grad() # 梯度清零
        
        outputs = model(batch_x) # 前向传播
        loss = criterion(outputs, batch_y) # 计算损失
        
        loss.backward() # 反向传播
        optimizer.step() # 更新参数
        
        running_loss += loss.item() * batch_x.size(0)
    
    epoch_loss = running_loss / len(train_loader.dataset)
    train_losses.append(epoch_loss)

    # 验证阶段
    model.eval() # 设置模型为评估模式
    val_running_loss = 0.0
    with torch.no_grad(): # 在验证阶段不计算梯度
        for batch_x_val, batch_y_val in test_loader: # 使用test_loader作为验证集
            batch_x_val, batch_y_val = batch_x_val.to(device), batch_y_val.to(device)
            val_outputs = model(batch_x_val)
            val_loss = criterion(val_outputs, batch_y_val)
            val_running_loss += val_loss.item() * batch_x_val.size(0)
    
    epoch_val_loss = val_running_loss / len(test_loader.dataset) # 注意这里是test_loader的长度
    val_losses.append(epoch_val_loss)

    print(f"Epoch [{epoch+1}/{epochs}], Train Loss: {epoch_loss:.4f}, Val Loss: {epoch_val_loss:.4f}")

    # Early Stopping
    if epoch_val_loss < best_val_loss:
        best_val_loss = epoch_val_loss
        current_patience = 0
        torch.save(model.state_dict(), 'best_attention_fp_model.pth') # 保存最佳模型
    else:
        current_patience += 1
        if current_patience >= patience:
            print(f"Early stopping at epoch {epoch+1} due to no improvement in validation loss.")
            break

# 加载最佳模型权重
model.load_state_dict(torch.load('best_attention_fp_model.pth'))
print("Loaded best model weights.")

# 5. 模型预测
model.eval() # 设置模型为评估模式
with torch.no_grad(): # 在预测阶段不计算梯度
    # 预测训练集
    y_pred_train_tensor = model(x_train_tensor.to(device))
    y_pred_train = y_pred_train_tensor.cpu().numpy().flatten()
    
    # 预测测试集
    y_pred_test_tensor = model(x_test_tensor.to(device))
    y_pred_test = y_pred_test_tensor.cpu().numpy().flatten()

# 6. 模型评估
def evaluate_model(y_true, y_pred, name=""):
    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    
    residuals = y_true - y_pred
    se = np.std(residuals)
    
    print(f"\n--- {name} Evaluation ---")
    print(f"R^2 Score: {r2:.4f}")
    print(f"MAE: {mae:.4f}")
    print(f"Standard Error (SE): {se:.4f}")
    return r2, mae, se

print("\n--- Training Set Performance ---")
r2_train, mae_train, se_train = evaluate_model(y_train_raw, y_pred_train, "Training Set")

print("\n--- Test Set Performance ---")
r2_test, mae_test, se_test = evaluate_model(y_test_raw, y_pred_test, "Test Set")

# 可视化训练过程 (可选)
plt.figure(figsize=(10, 6))
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss (MSE)')
plt.title('Model Training History')
plt.legend()
plt.grid(True)
plt.show()

# 可视化预测结果 (可选)
plt.figure(figsize=(10, 6))
plt.scatter(y_test_raw, y_pred_test, alpha=0.7)
plt.plot([y_test_raw.min(), y_test_raw.max()], [y_test_raw.min(), y_test_raw.max()], 'r--', lw=2)
plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
plt.title("Actual vs. Predicted Values (Test Set)")
plt.grid(True)
plt.show()

true_values = y_test_raw
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