import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, auc, make_scorer
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
import time

# 读取数据
print("正在读取数据...")
train_data = pd.read_excel('C:/Users/Administrator/Desktop/train_jianmo1.xlsx')
test_data = pd.read_excel('C:/Users/Administrator/Desktop/test_jianmo1.xlsx')

X_train = train_data.drop(columns=['label'])
y_train = train_data['label']
X_test = test_data.drop(columns=['label'])
y_test = test_data['label']

print(f"训练集形状: {X_train.shape}")
print(f"测试集形状: {X_test.shape}")
print(f"类别分布 - 训练集: {y_train.value_counts().to_dict()}")
print(f"类别分布 - 测试集: {y_test.value_counts().to_dict()}")

# ========== 1. 参数调优 - 网格搜索 ==========
print("\n" + "="*50)
print("随机森林参数调优")
print("="*50)

# 定义AUC评分函数（供网格搜索使用）
auc_scorer = make_scorer(lambda y_true, y_proba: roc_auc_score(y_true, y_proba[:, 1]), 
                          needs_proba=True)

# 检查类别是否平衡，决定class_weight参数
class_ratio = y_train.value_counts()[1] / y_train.value_counts()[0]
if class_ratio < 0.5 or class_ratio > 2:
    print(f"检测到类别不平衡（比例: {class_ratio:.3f}），使用balanced权重")
    class_weight_options = ['balanced', 'balanced_subsample']
else:
    print("类别相对平衡，使用None权重")
    class_weight_options = [None, 'balanced']

# 根据数据规模决定参数范围
n_features = X_train.shape[1]
n_samples = X_train.shape[0]

print(f"特征数量: {n_features}, 样本数量: {n_samples}")

if n_features < 50:  # 特征较少
    max_features_options = ['sqrt', 'log2', None]
elif n_features < 100:  # 中等特征
    max_features_options = ['sqrt', 'log2', 0.3]
else:  # 特征很多
    max_features_options = ['sqrt', 'log2', 0.1]

# 方法1：随机搜索（快速找到大致范围）
print("\n=== 随机搜索（快速探索） ===")

# 定义随机搜索的参数分布
random_param_dist = {
    'n_estimators': [50, 100, 200, 300, 500],
    'max_depth': [None, 10, 20, 30, 50],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': max_features_options,
    'class_weight': class_weight_options,
    'bootstrap': [True, False]
}

# 创建基础模型
base_rf = RandomForestClassifier(
    random_state=42,
    n_jobs=-1,  # 使用所有CPU核心
    verbose=0
)

# 随机搜索
print("开始随机搜索...")
start_time = time.time()

random_search = RandomizedSearchCV(
    estimator=base_rf,
    param_distributions=random_param_dist,
    n_iter=20,  # 随机尝试20组参数
    cv=5,  # 5折交叉验证
    scoring='roc_auc',
    n_jobs=-1,
    verbose=1,
    random_state=42
)

random_search.fit(X_train, y_train)
random_search_time = time.time() - start_time

print(f"随机搜索完成，耗时: {random_search_time:.1f}秒")
print(f"最佳随机搜索参数: {random_search.best_params_}")
print(f"最佳随机搜索AUC: {random_search.best_score_:.4f}")

# 方法2：基于随机搜索结果的精细化网格搜索
print("\n=== 精细化网格搜索 ===")

# 基于随机搜索的最佳参数，定义更精细的搜索网格
best_random_params = random_search.best_params_

# 根据最佳参数调整搜索范围
if best_random_params['max_depth'] is None:
    max_depth_options = [None, 20, 30]
elif best_random_params['max_depth'] >= 20:
    max_depth_options = [None, best_random_params['max_depth']-5, 
                        best_random_params['max_depth'], 
                        best_random_params['max_depth']+5]
else:
    max_depth_options = [best_random_params['max_depth']-5 if best_random_params['max_depth']>5 else 5,
                        best_random_params['max_depth'],
                        best_random_params['max_depth']+5]

# 精细化网格
grid_param = {
    'n_estimators': [best_random_params['n_estimators']-50 if best_random_params['n_estimators']>50 else 50,
                    best_random_params['n_estimators'],
                    best_random_params['n_estimators']+50],
    'max_depth': max_depth_options,
    'min_samples_split': [max(2, best_random_params['min_samples_split']-2),
                         best_random_params['min_samples_split'],
                         best_random_params['min_samples_split']+2],
    'min_samples_leaf': [max(1, best_random_params['min_samples_leaf']-1),
                        best_random_params['min_samples_leaf'],
                        best_random_params['min_samples_leaf']+1],
    'max_features': [best_random_params['max_features']],
    'class_weight': [best_random_params['class_weight']],
    'bootstrap': [best_random_params['bootstrap']]
}

print("精细化网格搜索参数范围:")
for key, value in grid_param.items():
    print(f"  {key}: {value}")

print("\n开始精细化网格搜索...")
start_time = time.time()

# 使用分层交叉验证
cv_stratified = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

grid_search = GridSearchCV(
    estimator=RandomForestClassifier(random_state=42, n_jobs=-1, verbose=0),
    param_grid=grid_param,
    cv=cv_stratified,
    scoring='roc_auc',
    n_jobs=-1,  # 并行计算
    verbose=1,
    refit=True  # 用最佳参数重新训练模型
)

grid_search.fit(X_train, y_train)
grid_search_time = time.time() - start_time

print(f"网格搜索完成，耗时: {grid_search_time:.1f}秒")
print(f"最佳参数: {grid_search.best_params_}")
print(f"最佳交叉验证AUC: {grid_search.best_score_:.4f}")

# ========== 2. 使用最佳参数进行完整评估 ==========
print("\n" + "="*50)
print("使用最佳参数进行模型评估")
print("="*50)

# 获取最佳参数
best_params = grid_search.best_params_
print(f"最终使用的参数: {best_params}")

# 使用最佳参数创建最终模型
final_model = RandomForestClassifier(**best_params, random_state=42, n_jobs=-1)

# 创建绘制ROC曲线的图表
plt.figure(figsize=(12, 10))

# 设置十折交叉验证（用于训练集评估）
kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
mean_fpr = np.linspace(0, 1, 100)

# ========== 训练集交叉验证 ==========
print("\n=== 训练集10折交叉验证 ===")
train_tprs = []
train_aucs = []
fold_times = []

for fold, (train_index, val_index) in enumerate(kf.split(X_train, y_train), 1):
    fold_start = time.time()
    
    # 获取训练集和验证集
    X_cv_train, X_cv_val = X_train.iloc[train_index], X_train.iloc[val_index]
    y_cv_train, y_cv_val = y_train.iloc[train_index], y_train.iloc[val_index]
    
    # 训练模型
    fold_model = RandomForestClassifier(**best_params, random_state=fold, n_jobs=-1)
    fold_model.fit(X_cv_train, y_cv_train)
    
    # 预测概率
    y_probs = fold_model.predict_proba(X_cv_val)[:, 1]
    
    # 计算ROC曲线
    fpr, tpr, _ = roc_curve(y_cv_val, y_probs)
    roc_auc = auc(fpr, tpr)
    
    # 将当前轮次的FPR和TPR插值到均匀间隔点上
    train_tprs.append(np.interp(mean_fpr, fpr, tpr))
    train_tprs[-1][0] = 0.0
    train_aucs.append(roc_auc)
    
    fold_time = time.time() - fold_start
    fold_times.append(fold_time)
    
    print(f"Fold {fold}: AUC = {roc_auc:.4f}, 耗时: {fold_time:.1f}秒")

# 计算训练集交叉验证的统计量
train_auc_mean = np.mean(train_aucs)
train_auc_std = np.std(train_aucs)
train_auc_ci = np.percentile(train_aucs, [2.5, 97.5])
avg_fold_time = np.mean(fold_times)

print(f"\n训练集交叉验证汇总:")
print(f"平均AUC: {train_auc_mean:.4f} (±{train_auc_std:.4f})")
print(f"95%置信区间: ({train_auc_ci[0]:.4f}, {train_auc_ci[1]:.4f})")
print(f"每折平均训练时间: {avg_fold_time:.1f}秒")

# ========== 用全部训练数据训练最终模型 ==========
print("\n=== 使用全部训练数据训练最终模型 ===")
start_time = time.time()
final_model.fit(X_train, y_train)
final_train_time = time.time() - start_time
print(f"最终模型训练完成，耗时: {final_train_time:.1f}秒")

# ========== 在独立测试集上评估 ==========
print("\n=== 独立测试集评估 ===")
test_start_time = time.time()

# 测试集预测
y_test_probs = final_model.predict_proba(X_test)[:, 1]
y_test_pred = final_model.predict(X_test)

# 计算测试集ROC曲线
test_fpr, test_tpr, _ = roc_curve(y_test, y_test_probs)
test_auc = auc(test_fpr, test_tpr)

# 插值到均匀间隔点
test_tpr_interp = np.interp(mean_fpr, test_fpr, test_tpr)
test_tpr_interp[0] = 0.0

# 计算其他测试集指标
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

test_accuracy = accuracy_score(y_test, y_test_pred)
test_precision = precision_score(y_test, y_test_pred)
test_recall = recall_score(y_test, y_test_pred)
test_f1 = f1_score(y_test, y_test_pred)

print(f"测试集预测完成，耗时: {time.time()-test_start_time:.1f}秒")
print(f"测试集AUC: {test_auc:.4f}")
print(f"测试集准确率: {test_accuracy:.4f}")
print(f"测试集精确率: {test_precision:.4f}")
print(f"测试集召回率: {test_recall:.4f}")
print(f"测试集F1分数: {test_f1:.4f}")

# 计算测试集AUC的置信区间（使用bootstrap方法）
print("\n正在计算测试集AUC置信区间...")
n_bootstraps = 1000
bootstrap_aucs = []

for i in range(n_bootstraps):
    # 有放回抽样
    indices = np.random.choice(len(y_test), len(y_test), replace=True)
    if len(np.unique(y_test.iloc[indices])) < 2:
        continue
    
    fpr_boot, tpr_boot, _ = roc_curve(y_test.iloc[indices], y_test_probs[indices])
    bootstrap_aucs.append(auc(fpr_boot, tpr_boot))
    
    # 进度显示
    if (i+1) % 200 == 0:
        print(f"Bootstrap进度: {i+1}/{n_bootstraps}")

test_auc_ci = np.percentile(bootstrap_aucs, [2.5, 97.5]) if bootstrap_aucs else [np.nan, np.nan]
print(f"测试集AUC 95%置信区间: ({test_auc_ci[0]:.4f}, {test_auc_ci[1]:.4f})")

# ========== 绘制ROC曲线 ==========
print("\n正在绘制ROC曲线...")
# 绘制训练集交叉验证的平均ROC曲线（带标准差阴影）
mean_train_tpr = np.mean(train_tprs, axis=0)
std_train_tpr = np.std(train_tprs, axis=0)

plt.plot(mean_fpr, mean_train_tpr, 
         label=f'Train (CV) - AUC: {train_auc_mean:.4f} ({train_auc_ci[0]:.4f}, {train_auc_ci[1]:.4f})',
         color='blue', lw=2.5)

# 添加训练集的标准差阴影
tprs_upper = np.minimum(mean_train_tpr + std_train_tpr, 1)
tprs_lower = np.maximum(mean_train_tpr - std_train_tpr, 0)
plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='blue', alpha=0.2)

# 绘制测试集的ROC曲线
plt.plot(mean_fpr, test_tpr_interp, 
         label=f'Test - AUC: {test_auc:.4f} ({test_auc_ci[0]:.4f}, {test_auc_ci[1]:.4f})',
         color='red', lw=2.5, linestyle='--')

# 添加随机猜测线
plt.plot([0, 1], [0, 1], color='gray', linestyle=':', lw=2, label='Random Guess')

# 设置图表
plt.xlabel('False Positive Rate', fontsize=12)
plt.ylabel('True Positive Rate', fontsize=12)
plt.title(f'ROC Curve - Random Forest (Optimized)\nBest Params: n_estimators={best_params.get("n_estimators", "N/A")}', fontsize=14)
plt.legend(loc='lower right', fontsize=11)
plt.grid(True, alpha=0.3)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.tight_layout()

# 保存图表
plt.savefig('C:/Users/Administrator/Desktop/random_forest_roc_optimized.png', dpi=300, bbox_inches='tight')
plt.show()

# ========== 特征重要性分析 ==========
print("\n=== 特征重要性分析 ===")
feature_importances = final_model.feature_importances_
feature_names = X_train.columns

# 创建特征重要性DataFrame
importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': feature_importances
}).sort_values('Importance', ascending=False)

# 显示前20个重要特征
print(f"\n特征重要性排名（前20）:")
print(importance_df.head(20).to_string(index=False))

# 绘制特征重要性图
plt.figure(figsize=(14, 10))
top_n = min(20, len(importance_df))
plt.barh(range(top_n), importance_df['Importance'].head(top_n)[::-1])
plt.yticks(range(top_n), importance_df['Feature'].head(top_n)[::-1], fontsize=10)
plt.xlabel('Feature Importance', fontsize=12)
plt.title(f'Top {top_n} Feature Importances - Optimized Random Forest', fontsize=14)
plt.grid(True, alpha=0.3, axis='x')
plt.tight_layout()

# 保存特征重要性图
plt.savefig('C:/Users/Administrator/Desktop/feature_importances.png', dpi=300, bbox_inches='tight')
plt.show()

# ========== 保存最终模型 ==========
import joblib
import os

# 创建模型保存路径
model_path = 'C:/Users/Administrator/Desktop/final_random_forest_optimized.pkl'
joblib.dump(final_model, model_path)
print(f"\n最终模型已保存到: {model_path}")
print(f"模型文件大小: {os.path.getsize(model_path)/1024/1024:.2f} MB")

# ========== 保存特征重要性到Excel ==========
importance_path = 'C:/Users/Administrator/Desktop/feature_importances.xlsx'
importance_df.to_excel(importance_path, index=False)
print(f"特征重要性已保存到: {importance_path}")

# ========== 模型性能完整摘要 ==========
print("\n" + "="*60)
print("随机森林模型（调优后）性能完整摘要")
print("="*60)

print(f"\n【最佳参数】")
for key, value in best_params.items():
    print(f"  {key}: {value}")

print(f"\n【训练集性能（10折交叉验证）】")
print(f"  平均AUC: {train_auc_mean:.4f} (±{train_auc_std:.4f})")
print(f"  95%置信区间: ({train_auc_ci[0]:.4f} - {train_auc_ci[1]:.4f})")

print(f"\n【测试集性能】")
print(f"  AUC: {test_auc:.4f}")
print(f"  AUC 95%置信区间: ({test_auc_ci[0]:.4f} - {test_auc_ci[1]:.4f})")
print(f"  准确率: {test_accuracy:.4f}")
print(f"  精确率: {test_precision:.4f}")
print(f"  召回率: {test_recall:.4f}")
print(f"  F1分数: {test_f1:.4f}")

print(f"\n【泛化性能】")
print(f"  测试集AUC/训练集AUC: {test_auc/train_auc_mean:.2%}")

# 计算过拟合程度
overfitting_ratio = (train_auc_mean - test_auc) / train_auc_mean
if overfitting_ratio < 0.05:
    overfitting_status = "轻微过拟合"
elif overfitting_ratio < 0.1:
    overfitting_status = "中度过拟合"
else:
    overfitting_status = "严重过拟合"

print(f"  过拟合程度: {overfitting_ratio:.2%} ({overfitting_status})")

print(f"\n【计算时间】")
print(f"  随机搜索: {random_search_time:.1f}秒")
print(f"  网格搜索: {grid_search_time:.1f}秒")
print(f"  每折交叉验证平均时间: {avg_fold_time:.1f}秒")
print(f"  最终模型训练时间: {final_train_time:.1f}秒")

print(f"\n【特征重要性】")
print(f"  最重要的5个特征:")
for i, row in importance_df.head(5).iterrows():
    print(f"    {i+1}. {row['Feature']}: {row['Importance']:.4f}")

print(f"\n【文件保存】")
print(f"  模型文件: {model_path}")
print(f"  ROC曲线图: C:/Users/Administrator/Desktop/random_forest_roc_optimized.png")
print(f"  特征重要性图: C:/Users/Administrator/Desktop/feature_importances.png")
print(f"  特征重要性表: {importance_path}")

print("\n" + "="*60)
print("随机森林参数调优完成！")
print("="*60)

# ========== 可选：混淆矩阵可视化 ==========
print("\n生成混淆矩阵...")
conf_matrix = confusion_matrix(y_test, y_test_pred)

plt.figure(figsize=(8, 6))
plt.imshow(conf_matrix, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Confusion Matrix - Test Set', fontsize=14)
plt.colorbar()

# 添加数值标签
thresh = conf_matrix.max() / 2
for i in range(conf_matrix.shape[0]):
    for j in range(conf_matrix.shape[1]):
        plt.text(j, i, format(conf_matrix[i, j], 'd'),
                 horizontalalignment="center",
                 color="white" if conf_matrix[i, j] > thresh else "black",
                 fontsize=12)

plt.ylabel('True Label', fontsize=12)
plt.xlabel('Predicted Label', fontsize=12)
plt.xticks([0, 1], ['Class 0', 'Class 1'])
plt.yticks([0, 1], ['Class 0', 'Class 1'])
plt.tight_layout()

# 保存混淆矩阵图
plt.savefig('C:/Users/Administrator/Desktop/confusion_matrix.png', dpi=300, bbox_inches='tight')
plt.show()
