import shap
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from rdkit import Chem
from rdkit.Chem import AllChem
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.tree import export_graphviz
import seaborn as sns

from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.preprocessing import label_binarize
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import ExtraTreeRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import BaggingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import r2_score, mean_absolute_error

train_data = pd.read_csv("new_train_data.csv", header=0)
train_array = train_data.to_numpy(dtype=np.float32)

test_data = pd.read_csv("new_test_data.csv", header=0)
test_array = test_data.to_numpy(dtype=np.float32)

x_train = train_data.iloc[:,:-1].values  # Use iloc for DataFrame indexing
y_train = train_data.iloc[:,-1].values    # Extract the last column as target
x_test = test_data.iloc[:,:-1].values     # Similarly for the test set
y_test = test_data.iloc[:,-1].values      # Extract target from test set




num_runs = 100  # 设置运行次数
r2_train_scores = []
r2_test_scores = []
mae_train_scores = []
mae_test_scores = []
se_train_scores = []
se_test_scores = []

for _ in range(num_runs):
    # 创建 MLPRegressor 实例
    # clf = MLPRegressor(hidden_layer_sizes=(100, 50), activation='relu', learning_rate_init=0.001)
    clf = RandomForestRegressor(n_estimators=100, criterion='squared_error', random_state=42)
    # clf = GradientBoostingRegressor(loss='squared_error', learning_rate=0.1, n_estimators=100, subsample=1.0, criterion='friedman_mse', min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_depth=3, random_state=42)
    # clf = LinearRegression()
    
    # 在训练集上训练模型
    clf.fit(x_train, y_train.ravel())

    # 在训练集和测试集上进行预测
    y_pred_train = clf.predict(x_train)
    y_pred_test = clf.predict(x_test)

    # 计算并保存性能指标
    r2_train_scores.append(r2_score(y_train, y_pred_train))
    r2_test_scores.append(r2_score(y_test, y_pred_test))
    mae_train_scores.append(mean_absolute_error(y_train, y_pred_train))
    mae_test_scores.append(mean_absolute_error(y_test, y_pred_test))
    
    # 计算实际值与预测值的估计标准误差训练集
    residuals_train = y_train - y_pred_train
    mse_train = np.mean(residuals_train**2)
    estimated_standard_error_train = np.sqrt(mse_train)
    se_train_scores.append(estimated_standard_error_train)
    
    # 计算实际值与预测值的估计标准误差测试集
    residuals_test = y_test - y_pred_test
    mse_test = np.mean(residuals_test**2)
    estimated_standard_error_test = np.sqrt(mse_test)
    se_test_scores.append(estimated_standard_error_test)

# 计算平均性能指标
average_r2_train = np.mean(r2_train_scores)
average_r2_test = np.mean(r2_test_scores)
average_mae_train = np.mean(mae_train_scores)
average_mae_test = np.mean(mae_test_scores)
average_se_train = np.mean(se_train_scores)
average_se_test = np.mean(se_test_scores)


print(f'Average MLPRegressor training set R^2: {average_r2_train}')
print(f'Average MLPRegressor training set MAE: {average_mae_train}')
print(f'Average MLPRegressor testing set R^2: {average_r2_test}')
print(f'Average MLPRegressor testing set MAE: {average_mae_test}')
print(f'Estimated standard error for train set (actual vs predicted): {average_se_train}')
print(f'Estimated standard error for test set (actual vs predicted): {average_se_test}')


# 假设 x_test 已经定义，并且 clf 是您的训练模型
# x_test_df = pd.DataFrame(x_test, columns=[f'feature_{i}' for i in range(x_test.shape[1])])
importances = clf.feature_importances_  
indices = np.argsort(importances)[::-1]  # 根据重要性排序的索引

# 可视化所有特征重要性
plt.figure(figsize=(12, 8))
plt.title("Feature Importances")

# 根据所有特征的索引绘制柱状图
plt.bar(range(len(importances)), importances[indices], align="center", color='cyan')

# 设置所有特征名称作为横坐标标签
plt.xticks(range(len(importances)), [test_data.columns[i] for i in indices], rotation='vertical')
plt.xlabel("Molecular Descriptors")
plt.ylabel("Importance Score")
plt.tight_layout()  # 防止标签被剪切
plt.show()

# 创建 SHAP 解释器
explainer = shap.TreeExplainer(clf)

# 计算 SHAP 值
shap_values = explainer.shap_values(x_test)

# SHAP summary plot
plt.figure(figsize=(12, 8))
shap.summary_plot(shap_values, x_test, 
                  feature_names=test_data.columns.tolist(),  # 使用实际的列名
                  show=False)  # 防止立即显示
plt.tight_layout()
plt.show()

# SHAP bar plot
shap.summary_plot(shap_values, x_test, 
                  feature_names=test_data.columns.tolist(), 
                  plot_type='bar')