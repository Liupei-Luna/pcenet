import shap
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.tree import export_graphviz
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler


tra_hopv = pd.read_csv("train.csv", header=0)
tra_hopv = tra_hopv.iloc[:, 2:]
train_array = tra_hopv.to_numpy(dtype=np.float32)

tes_hopv = pd.read_csv("test.csv", header=0)
tes_hopv = tes_hopv.iloc[:, 2:]
test_array = tes_hopv.to_numpy(dtype=np.float32)

data = pd.read_csv("pre_data.csv", header=0)
data = data.iloc[:, 2:]
data_array = data.to_numpy(dtype=np.float32)

#供受体
x_train = tra_hopv.iloc[:,1:].values  # Use iloc for DataFrame indexing
y_train = tra_hopv.iloc[:,0].values    # Extract the last column as target
x_test = tes_hopv.iloc[:,1:].values     # Similarly for the test set
y_test = tes_hopv.iloc[:,0].values      # Extract target from test set

#供体
# x_train = tra_hopv.iloc[:,7:].values  # Use iloc for DataFrame indexing
# y_train = tra_hopv.iloc[:,0].values    # Extract the last column as target
# x_test = tes_hopv.iloc[:,7:].values     # Similarly for the test set
# y_test = tes_hopv.iloc[:,0].values      # Extract target from test set

pre_data = data.iloc[:, 1:].values



# class SelfAttention(nn.Module):
#     def __init__(self, in_dim):
#         super(SelfAttention, self).__init__()
#         self.query_conv = nn.Linear(in_dim, in_dim // 4)
#         self.key_conv = nn.Linear(in_dim, in_dim // 4)
#         self.value_conv = nn.Linear(in_dim, in_dim)
#         self.gamma = nn.Parameter(torch.zeros(1))

#     def forward(self, x):
#         proj_query = self.query_conv(x)
#         proj_key = self.key_conv(x)
#         energy = torch.matmul(proj_query, proj_key.T)
#         attention = F.softmax(energy, dim=-1)
#         proj_value = self.value_conv(x)
#         out = torch.matmul(attention, proj_value)
#         out = self.gamma * out + x
#         return out

class SelfAttention(nn.Module):
    def __init__(self, in_dim, head_dim=None):
        super(SelfAttention, self).__init__()
        self.head_dim = head_dim or in_dim // 4
        self.scale = self.head_dim ** -0.5
        
        self.query_conv = nn.Linear(in_dim, self.head_dim)
        self.key_conv = nn.Linear(in_dim, self.head_dim)
        self.value_conv = nn.Linear(in_dim, in_dim)
        self.gamma = nn.Parameter(torch.zeros(1))
        
        # 添加dropout提高泛化能力
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        batch_size = x.size(0)
        
        proj_query = self.query_conv(x)  # [batch, head_dim]
        proj_key = self.key_conv(x)      # [batch, head_dim]
        proj_value = self.value_conv(x)  # [batch, in_dim]
        
        # 正确的注意力计算
        energy = torch.matmul(proj_query, proj_key.transpose(-2, -1)) * self.scale
        attention = F.softmax(energy, dim=-1)
        attention = self.dropout(attention)
        
        out = torch.matmul(attention, proj_value)
        out = self.gamma * out + x
        return out
    
class ResidualBlock(nn.Module):
    def __init__(self, in_dim, dropout_rate=0.001): # 添加dropout_rate参数
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Linear(in_dim, in_dim)
        self.conv2 = nn.Linear(in_dim, in_dim)
        self.norm1 = nn.BatchNorm1d(in_dim)
        self.norm2 = nn.BatchNorm1d(in_dim)
        self.attention = SelfAttention(in_dim)
        self.dropout = nn.Dropout(p=dropout_rate) # 添加Dropout层

    def forward(self, x):
        identity = x
        out = F.relu(self.norm1(self.conv1(x)))
        out = self.norm2(self.conv2(out))
        out = self.attention(out)
        out = self.dropout(out) # 应用Dropout
        out += identity
        return F.relu(out)

class ResNetWithSelfAttention(nn.Module):
    def __init__(self, input_dim, hidden_dim=59, output_dim=1):
        super(ResNetWithSelfAttention, self).__init__()
        self.input_layer = nn.Linear(input_dim, hidden_dim)
        self.res_blocks = nn.Sequential(
            ResidualBlock(hidden_dim),
            ResidualBlock(hidden_dim),
            ResidualBlock(hidden_dim)
        )
        self.output_layer = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = F.relu(self.input_layer(x))
        x = self.res_blocks(x)
        x = self.output_layer(x)
        return x




x_train = np.array(x_train, dtype=np.float32)
y_train = np.array(y_train, dtype=np.float32)

x_test = np.array(x_test, dtype=np.float32)
y_test = np.array(y_test, dtype=np.float32)

# pre_data = np.array(pre_data, dtype=np.float32)

#转换为 PyTorch 的张量并转化成列向量
x_train_tensor = torch.from_numpy(x_train).float()
y_train_tensor = torch.from_numpy(y_train).float().view(-1, 1) 

x_test_tensor = torch.from_numpy(x_test).float()
y_test_tensor = torch.from_numpy(y_test).float().view(-1, 1)

# pre_data_tensor = torch.from_numpy(pre_data).float()
                                                           
# 创建数据集和数据加载器
dataset = TensorDataset(x_train_tensor, y_train_tensor)
dataloader = DataLoader(dataset, batch_size=64, shuffle=False)  # 设置合适的 batch_size

test_dataset = TensorDataset(x_test_tensor, y_test_tensor)
pre_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# pre_dataset = TensorDataset(pre_data_tensor)
# pre_dataloader = DataLoader(pre_dataset, batch_size=32, shuffle=False)

# 实例化模型
model = ResNetWithSelfAttention(input_dim=x_train.shape[1])

# 定义损失函数和优化器
criterion = nn.MSELoss()  # 均方误差损失函数
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)  # 使用 Adam 优化器
num_epochs = 500


# 训练模型
for epoch in range(num_epochs):
    model.train()
    for inputs, targets in dataloader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
    print(f'Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.4f}')

# 测试模型
torch.manual_seed(42)
np.random.seed(42)

# 定义模型评估函数
def evaluate_model(model, x_train_tensor, y_train, x_test_tensor, y_test, num_evals=1):
    r2_train_list, mae_train_list = [], []
    r2_test_list, mae_test_list = [], []
    se_train_list, se_test_list = [], []

    for _ in range(num_evals):
        model.eval()
        with torch.no_grad():
            y_train_pred = model(x_train_tensor)
            y_test_pred = model(x_test_tensor)

            # 计算R2和MAE
            r2_train = r2_score(y_train, y_train_pred.numpy().flatten())
            mae_train = mean_absolute_error(y_train, y_train_pred.numpy().flatten())
            r2_test = r2_score(y_test, y_test_pred.numpy().flatten())
            mae_test = mean_absolute_error(y_test, y_test_pred.numpy().flatten())

            # 计算标准误差
            se_train = np.sqrt(np.mean((y_train - y_train_pred.numpy().flatten())**2))
            se_test = np.sqrt(np.mean((y_test - y_test_pred.numpy().flatten())**2))

            # 收集指标
            r2_train_list.append(r2_train)
            mae_train_list.append(mae_train)
            r2_test_list.append(r2_test)
            mae_test_list.append(mae_test)
            se_train_list.append(se_train)
            se_test_list.append(se_test)

    # 计算平均值
    results = {
        'train_r2': np.mean(r2_train_list),
        'train_mae': np.mean(mae_train_list),
        'test_r2': np.mean(r2_test_list),
        'test_mae': np.mean(mae_test_list),
        'se_train': np.mean(se_train_list),
        'se_test': np.mean(se_test_list)
    }
    return results

# 调用评估
results = evaluate_model(model, x_train_tensor, y_train, x_test_tensor, y_test)
print(f"Train R^2: {results['train_r2']:.3f}")
print(f"Train MAE: {results['train_mae']:.3f}")
print(f"Test R^2: {results['test_r2']:.3f}")
print(f"Test MAE: {results['test_mae']:.3f}")
print(f"Estimated standard error (train): {results['se_train']:.3f}")
print(f"Estimated standard error (test): {results['se_test']:.3f}")



# 模型训练完成后，对预测数据进行处理和预测
print("开始对预测数据进行预测...")

# 将预测数据转换为张量
pre_data_tensor = torch.from_numpy(pre_data).float()

# 创建预测数据的数据加载器
pre_dataset = TensorDataset(pre_data_tensor)
pre_dataloader = DataLoader(pre_dataset, batch_size=64, shuffle=False)

# 进行预测
model.eval()  # 设置模型为评估模式
predictions = []

with torch.no_grad():  # 关闭梯度计算以节省内存和计算
    for batch in pre_dataloader:
        inputs = batch[0]  # 获取输入数据
        outputs = model(inputs)  # 模型预测
        predictions.extend(outputs.numpy().flatten())  # 将预测结果添加到列表中

# 将预测结果转换为numpy数组
predictions = np.array(predictions)

print(f"预测完成！共预测了 {len(predictions)} 个样本")
print(f"预测值范围: {predictions.min():.4f} 到 {predictions.max():.4f}")
print(f"预测值均值: {predictions.mean():.4f}")

# 创建结果DataFrame
results_df = pd.DataFrame({
    'Sample_Index': range(len(predictions)),
    'Predicted_PCE': predictions
})

# 如果原始数据有其他标识列，可以添加进去
# 假设原始data.csv的前两列是标识信息
original_data = pd.read_csv("pre_data.csv", header=0)
if original_data.shape[1] >= 2:
    # 添加原始数据的前两列作为标识
    results_df['ID_1'] = original_data.iloc[:, 0].values
    results_df['ID_2'] = original_data.iloc[:, 1].values
    # 重新排列列的顺序
    results_df = results_df[['ID_1', 'ID_2', 'Sample_Index', 'Predicted_PCE']]

# 保存预测结果
output_filename = "predicted_pce_results1.csv"
results_df.to_csv(output_filename, index=False)
print(f"预测结果已保存到: {output_filename}")

# 显示前几行预测结果
print("\n前10个预测结果:")
print(results_df.head(10))

# 可选：绘制预测值分布图
plt.figure(figsize=(10, 6))
plt.hist(predictions, bins=50, alpha=0.7, edgecolor='black')
plt.xlabel('Predicted PCE')
plt.ylabel('Frequency')
plt.title('Distribution of Predicted PCE Values')
plt.grid(True, alpha=0.3)
plt.show()

# 可选：保存预测值的统计信息
stats_df = pd.DataFrame({
    'Statistic': ['Count', 'Mean', 'Std', 'Min', '25%', '50%', '75%', 'Max'],
    'Value': [
        len(predictions),
        predictions.mean(),
        predictions.std(),
        predictions.min(),
        np.percentile(predictions, 25),
        np.percentile(predictions, 50),
        np.percentile(predictions, 75),
        predictions.max()
    ]
})

stats_filename = "prediction_statistics.csv"
stats_df.to_csv(stats_filename, index=False)
print(f"\n预测统计信息已保存到: {stats_filename}")
print(stats_df)
