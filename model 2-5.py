import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# from smiles import x_train, y_train, x_test, y_test

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
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold


# tra_hopv_ = np.array(tra_hopv)
# tes_hopv_ = np.array(tes_hopv)
# x_train = tra_hopv_[:,9:]
# y_train = tra_hopv_[:,2]
# x_test = tes_hopv_[:,9:]
# y_test = tes_hopv_[:,2]

# -------------------- 1. 数据加载与预处理 --------------------
df = pd.read_csv("data.csv", header=0)
df_processed = df.iloc[:, 2:]    # 移除前两列
x = df_processed.iloc[:, 1:].values # 特征数据
y = df_processed.iloc[:, 0].values  # 目标变量

# -------------------- 2. 设置五折交叉验证 --------------------
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# -------------------- 3. 初始化累积所有折叠性能指标的列表 --------------------
all_r2_train = []
all_r2_test = []
all_mae_train = []
all_mae_test = []
all_se_train = []
all_se_test = []

# -------------------- 4. 交叉验证循环 --------------------
for fold, (train_index, test_index) in enumerate(kf.split(x), 1):
    x_train, x_test = x[train_index], x[test_index]
    y_train, y_test = y[train_index], y[test_index]
    
    print(f"--- Fold {fold} ---")
    print(f"训练集样本数: {len(x_train)}")
    print(f"测试集样本数: {len(x_test)}")
    
    # 每次循环都创建新的模型实例以确保独立性
    # 当前选用 GradientBoostingRegressor
    clf = MLPRegressor(hidden_layer_sizes=(100, 50), activation='relu', learning_rate_init=0.001, random_state=42)
    # clf = RandomForestRegressor(n_estimators=100, criterion='squared_error', random_state=42)
    # clf = LinearRegression()
    # clf = GradientBoostingRegressor(loss='squared_error', learning_rate=0.1, n_estimators=100, subsample=1.0, 
    #                                 criterion='friedman_mse', min_samples_split=2, min_samples_leaf=1, 
    #                                 max_depth=3, random_state=42) # random_state 保证可复现性
    
    # 在训练集上训练模型
    clf.fit(x_train, y_train.ravel()) # .ravel() 确保y是一维数组

    # 在训练集和测试集上进行预测
    y_pred_train = clf.predict(x_train)
    y_pred_test = clf.predict(x_test)

    # 计算并保存当前折叠的性能指标
    current_r2_train = r2_score(y_train, y_pred_train)
    current_r2_test = r2_score(y_test, y_pred_test)
    current_mae_train = mean_absolute_error(y_train, y_pred_train)
    current_mae_test = mean_absolute_error(y_test, y_pred_test)
    
    # 计算实际值与预测值的估计标准误差（即RMSE）
    residuals_train = y_train - y_pred_train
    current_se_train = np.sqrt(np.mean(residuals_train**2))
    
    residuals_test = y_test - y_pred_test
    current_se_test = np.sqrt(np.mean(residuals_test**2))
    
    # 将当前折叠的指标添加到总列表中
    all_r2_train.append(current_r2_train)
    all_r2_test.append(current_r2_test)
    all_mae_train.append(current_mae_train)
    all_mae_test.append(current_mae_test)
    all_se_train.append(current_se_train)
    all_se_test.append(current_se_test)

    # 打印当前折叠的详细指标 (可选，如果只想看最终均值可以注释掉)
    print(f'Train R^2: {current_r2_train:.4f}, MAE: {current_mae_train:.4f}, SE(RMSE): {current_se_train:.4f}')
    print(f'Test R^2: {current_r2_test:.4f}, MAE: {current_mae_test:.4f}, SE(RMSE): {current_se_test:.4f}')
    print("-" * 30)

# -------------------- 5. 计算并打印所有折叠的平均性能指标 --------------------
print("\n--- Average Performance Across All Folds ---")
print(f'Average Train R^2: {np.mean(all_r2_train):.4f} ± {np.std(all_r2_train):.4f}')
print(f'Average Train MAE: {np.mean(all_mae_train):.4f} ± {np.std(all_mae_train):.4f}')
print(f'Average Train SE (RMSE): {np.mean(all_se_train):.4f} ± {np.std(all_se_train):.4f}')
print(f'Average Test R^2: {np.mean(all_r2_test):.4f} ± {np.std(all_r2_test):.4f}')
print(f'Average Test MAE: {np.mean(all_mae_test):.4f} ± {np.std(all_mae_test):.4f}')
print(f'Average Test SE (RMSE): {np.mean(all_se_test):.4f} ± {np.std(all_se_test):.4f}')


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_absolute_error
import os
from scipy import stats
from scipy.stats import norm, pearsonr
import warnings
warnings.filterwarnings('ignore')

# 尝试导入seaborn用于更好的可视化
try:
    import seaborn as sns
    sns.set_style('whitegrid')
except ImportError:
    print("Seaborn not found. Using matplotlib for visualization.")
    sns = None

# 对于每个模型，在模型评估结束后保存校准数据
def save_calibration_data(model_name, confidence_levels, observed_coverages, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    
    # 创建数据框
    calibration_df = pd.DataFrame({
        'model': [model_name] * len(confidence_levels),
        'confidence_level': confidence_levels,
        'observed_coverage': observed_coverages
    })
    
    # 保存到CSV
    calibration_df.to_csv(f'{save_dir}/{model_name}_calibration.csv', index=False)
    
    return calibration_df    

# -------------------- 1. 数据加载与预处理 --------------------
df = pd.read_csv("data.csv", header=0)
df_processed = df.iloc[:, 2:]    # 移除前两列
x = df_processed.iloc[:, 1:].values # 特征数据
y = df_processed.iloc[:, 0].values  # 目标变量

# -------------------- 2. 设置五折交叉验证 --------------------
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# -------------------- 3. 初始化累积所有折叠性能指标的列表 --------------------
all_r2_train = []
all_r2_test = []
all_mae_train = []
all_mae_test = []
all_se_train = []
all_se_test = []

# 为不确定性分析收集数据
all_oof_true = []
all_oof_mean_preds = []
all_oof_stds = []

# -------------------- 4. 交叉验证循环 --------------------
for fold, (train_index, test_index) in enumerate(kf.split(x), 1):
    x_train, x_test = x[train_index], x[test_index]
    y_train, y_test = y[train_index], y[test_index]
    
    print(f"--- Fold {fold} ---")
    print(f"训练集样本数: {len(x_train)}")
    print(f"测试集样本数: {len(x_test)}")
    
    # 使用三个分位数训练模型：0.1, 0.5, 0.9
    # 这将允许我们估计90%的预测区间
    quantiles = [0.5, 0.7, 0.8, 0.9, 0.95]
    gb_models = []
    
    for q in quantiles:
        model = GradientBoostingRegressor(
            loss='quantile', 
            alpha=q,
            n_estimators=100, 
            max_depth=3, 
            learning_rate=0.1, 
            min_samples_split=2, 
            min_samples_leaf=1, 
            random_state=42
        )
        
        model.fit(x_train, y_train)
        gb_models.append(model)
    
    # 使用中位数模型(alpha=0.5)作为主要预测模型
    main_model = gb_models[1]  # 索引1对应alpha=0.5
    
    # 在训练集和测试集上进行预测
    y_pred_train = main_model.predict(x_train)
    y_pred_test = main_model.predict(x_test)
    
    # 计算预测区间和不确定性
    y_lower_test = gb_models[0].predict(x_test)  # 10th 分位数
    y_upper_test = gb_models[2].predict(x_test)  # 90th 分位数
    
    # 估计标准差 (基于90%置信区间)
    # 在正态分布中，90%置信区间约为±1.645标准差
    y_std_test = (y_upper_test - y_lower_test) / (2 * 1.645)
    
    # 计算并保存当前折叠的性能指标
    current_r2_train = r2_score(y_train, y_pred_train)
    current_r2_test = r2_score(y_test, y_pred_test)
    current_mae_train = mean_absolute_error(y_train, y_pred_train)
    current_mae_test = mean_absolute_error(y_test, y_pred_test)
    
    # 计算实际值与预测值的估计标准误差（即RMSE）
    residuals_train = y_train - y_pred_train
    current_se_train = np.sqrt(np.mean(residuals_train**2))
    
    residuals_test = y_test - y_pred_test
    current_se_test = np.sqrt(np.mean(residuals_test**2))
    
    # 将当前折叠的指标添加到总列表中
    all_r2_train.append(current_r2_train)
    all_r2_test.append(current_r2_test)
    all_mae_train.append(current_mae_train)
    all_mae_test.append(current_mae_test)
    all_se_train.append(current_se_train)
    all_se_test.append(current_se_test)
    
    # 收集用于不确定性分析的数据
    all_oof_true.extend(y_test)
    all_oof_mean_preds.extend(y_pred_test)
    all_oof_stds.extend(y_std_test)

    # 打印当前折叠的详细指标
    print(f'Train R^2: {current_r2_train:.4f}, MAE: {current_mae_train:.4f}, SE(RMSE): {current_se_train:.4f}')
    print(f'Test R^2: {current_r2_test:.4f}, MAE: {current_mae_test:.4f}, SE(RMSE): {current_se_test:.4f}')
    print(f'Mean Estimated Uncertainty (Std): {np.mean(y_std_test):.4f}')
    print("-" * 30)

# -------------------- 5. 计算并打印所有折叠的平均性能指标 --------------------
print("\n--- Average Performance Across All Folds ---")
print(f'Average Train R^2: {np.mean(all_r2_train):.4f} ± {np.std(all_r2_train):.4f}')
print(f'Average Train MAE: {np.mean(all_mae_train):.4f} ± {np.std(all_mae_train):.4f}')
print(f'Average Train SE (RMSE): {np.mean(all_se_train):.4f} ± {np.std(all_se_train):.4f}')
print(f'Average Test R^2: {np.mean(all_r2_test):.4f} ± {np.std(all_r2_test):.4f}')
print(f'Average Test MAE: {np.mean(all_mae_test):.4f} ± {np.std(all_mae_test):.4f}')
print(f'Average Test SE (RMSE): {np.mean(all_se_test):.4f} ± {np.std(all_se_test):.4f}')

# -------------------- 6. 不确定性量化分析 --------------------
# 创建结果目录
uncertainty_dir = 'uncertainty_results_gbr'
os.makedirs(uncertainty_dir, exist_ok=True)

# 确保数据是numpy数组
all_oof_true = np.array(all_oof_true)
all_oof_mean_preds = np.array(all_oof_mean_preds)
all_oof_stds = np.array(all_oof_stds)

# 计算绝对误差
abs_errors = np.abs(all_oof_true - all_oof_mean_preds)

print("\n" + "="*80)
print("--- Uncertainty Quantification Analysis for Gradient Boosting Model ---")
print("="*80)

# 打印基本统计信息
print("\n--- Basic Statistics ---")
print(f"Total samples: {len(all_oof_true)}")
print(f"Mean prediction: {np.mean(all_oof_mean_preds):.4f}")
print(f"Mean uncertainty (std): {np.mean(all_oof_stds):.4f}")
print(f"Mean absolute error: {np.mean(abs_errors):.4f}")

# --- 6.1 不确定性分布 (直方图) ---
fig, ax = plt.subplots(figsize=(8, 6))
ax.grid(False)  # 确保网格关闭
if sns is not None:
    # 使用seaborn绘制带有黑色边框的直方图
    sns.histplot(all_oof_stds, bins=30, kde=True, color='skyblue', edgecolor='black', linewidth=1, ax=ax)
else:
    # 使用matplotlib绘制带有黑色边框的直方图
    ax.hist(all_oof_stds, bins=30, density=True, alpha=0.7, color='skyblue', edgecolor='black', linewidth=1)
    # 添加KDE曲线
    x_range = np.linspace(min(all_oof_stds), max(all_oof_stds), 1000)
    from scipy.stats import gaussian_kde
    kde = gaussian_kde(all_oof_stds)
    ax.plot(x_range, kde(x_range), 'r-', linewidth=2)

ax.set_xlabel('Prediction Standard Deviation', fontsize=12)
ax.set_ylabel('Frequency', fontsize=12)
# 设置轴线为黑色实线
ax.spines['left'].set_color('black')
ax.spines['bottom'].set_color('black')
ax.spines['left'].set_linewidth(1.0)
ax.spines['bottom'].set_linewidth(1.0)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tight_layout()
plt.savefig(f'{uncertainty_dir}/uncertainty_distribution.png', dpi=300, bbox_inches='tight')
plt.show()

# --- 6.2 预测误差与不确定性的相关性 ---
fig, ax = plt.subplots(figsize=(8, 6))
ax.grid(False)  # 确保网格关闭
# 散点图不使用黑色边框
ax.scatter(all_oof_stds, abs_errors, alpha=0.6, s=20, color='darkgreen')
ax.set_xlabel('Prediction Standard Deviation (Uncertainty)', fontsize=12)
ax.set_ylabel('Absolute Prediction Error |y_true - y_pred_mean|', fontsize=12)

# 计算并显示皮尔逊相关系数
corr_coef = np.corrcoef(all_oof_stds, abs_errors)[0, 1]
ax.text(0.05, 0.95, f'Pearson Correlation (Std Dev vs Abs Error): {corr_coef:.3f}', 
         transform=ax.transAxes, fontsize=12, verticalalignment='top', 
         bbox=dict(boxstyle='round,pad=0.5', fc='wheat', alpha=0.5, edgecolor='black'))
# 设置轴线为黑色实线
ax.spines['left'].set_color('black')
ax.spines['bottom'].set_color('black')
ax.spines['left'].set_linewidth(1.0)
ax.spines['bottom'].set_linewidth(1.0)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tight_layout()
plt.savefig(f'{uncertainty_dir}/error_uncertainty_correlation.png', dpi=300, bbox_inches='tight')
plt.show()

# --- 6.3 校准图 (Calibration Plot) ---
print("\n--- Coverage Statistics (Calibration) ---")
confidence_levels = np.array([0.50, 0.70, 0.80, 0.90, 0.95, 0.99])
observed_coverages = []

fig, ax = plt.subplots(figsize=(8, 6))
ax.grid(False)  # 确保网格关闭
ax.plot([0, 1], [0, 1], 'k--', label='Perfectly Calibrated')

for alpha in confidence_levels:
    # 计算对应于alpha置信水平的Z值
    z_score = norm.ppf(1 - (1 - alpha) / 2)
    # 计算置信区间
    lower_bound = all_oof_mean_preds - z_score * all_oof_stds
    upper_bound = all_oof_mean_preds + z_score * all_oof_stds
    
    # 计算实际覆盖率
    is_covered = (all_oof_true >= lower_bound) & (all_oof_true <= upper_bound)
    observed_coverage = np.mean(is_covered)
    
    # 将实际覆盖率添加到列表中
    observed_coverages.append(observed_coverage)
    
    print(f"  Nominal {alpha*100:.0f}% CI Coverage: Observed {observed_coverage*100:.2f}% (Error: {abs(observed_coverage - alpha):.4f})")

# 保存校准数据
save_dir = 'calibration_data'
save_calibration_data('GBR', confidence_levels, observed_coverages, save_dir)

# 绘制观察到的覆盖率
ax.plot(confidence_levels, observed_coverages, marker='o', color='red', linestyle='-', label='Observed Coverage')
ax.scatter(confidence_levels, observed_coverages, s=80, color='red')

ax.set_xlabel('Nominal Coverage Probability', fontsize=12)
ax.set_ylabel('Observed Coverage Probability', fontsize=12)
ax.legend(fontsize=10)
# 设置轴线为黑色实线
ax.spines['left'].set_color('black')
ax.spines['bottom'].set_color('black')
ax.spines['left'].set_linewidth(1.0)
ax.spines['bottom'].set_linewidth(1.0)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tight_layout()
plt.savefig(f'{uncertainty_dir}/calibration_plot.png', dpi=300, bbox_inches='tight')
plt.show()

# --- 6.4 归一化误差分布 ---
normalized_errors = (all_oof_true - all_oof_mean_preds) / all_oof_stds

fig, ax = plt.subplots(figsize=(8, 6))
ax.grid(False)  # 确保网格关闭
if sns is not None:
    sns.histplot(normalized_errors, kde=True, stat='density', bins=30, 
                 color='lightblue', edgecolor='black', linewidth=1, ax=ax)
else:
    ax.hist(normalized_errors, bins=30, density=True, alpha=0.7, 
             color='lightblue', edgecolor='black', linewidth=1)
    
# 添加标准正态分布曲线
x_range = np.linspace(-4, 4, 1000)
ax.plot(x_range, norm.pdf(x_range, 0, 1), 'r-', linewidth=2, label='Standard Normal')

ax.set_xlabel('Normalized Error: (y_true - y_pred_mean) / std', fontsize=12)
ax.set_ylabel('Density', fontsize=12)
ax.legend(fontsize=10)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tight_layout()
plt.savefig(f'{uncertainty_dir}/normalized_error_distribution.png', dpi=300, bbox_inches='tight')
plt.show()

# 进行Kolmogorov-Smirnov检验
from scipy import stats
ks_statistic, ks_pvalue = stats.kstest(normalized_errors, 'norm')
print(f"\n--- Kolmogorov-Smirnov Test (normalized errors vs. standard normal) ---")
print(f"  KS Statistic: {ks_statistic:.4f}, p-value: {ks_pvalue:.4f}")







# true_values = y_test
# predicted_values = y_pred_test

# # 绘制散点图
# plt.figure(figsize=(8, 8))
# plt.scatter(true_values, predicted_values, color='blue')

# # 绘制y=x线
# min_val = min(min(true_values), min(predicted_values))
# max_val = max(max(true_values), max(predicted_values))
# max_val = 5
# plt.plot([min_val, max_val], [min_val, max_val], 'black')

# # 设置坐标轴范围
# plt.xlim(0, max_val)
# plt.ylim(0, max_val)

# # 添加图例和标签
# plt.xlabel('Experimental PCE/%')
# plt.ylabel('Predicted PCE/%')
# # plt.title('Scatter Plot of True vs Predicted Values')
# # plt.legend(loc='lower right')

# # 显示图形
# plt.show()