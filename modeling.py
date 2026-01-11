import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

# 读取数据
train_data = pd.read_excel('C:/Users/Administrator/Desktop/train_jianmo3.xlsx')
test_data = pd.read_excel('C:/Users/Administrator/Desktop/test_jianmo3.xlsx')

X_train = train_data.drop(columns=['label'])
y_train = train_data['label']
X_test = test_data.drop(columns=['label'])
y_test = test_data['label']

# 初始化模型列表
models = [
    ('Logistic Regression', LogisticRegression(max_iter=1000, random_state=42)),
    ('Random Forest', RandomForestClassifier(random_state=42)),
    ('SVM', SVC(probability=True, random_state=42)),
    ('AdaBoost', AdaBoostClassifier(random_state=42)),
    ('K-Nearest Neighbors', KNeighborsClassifier()),
    ('Decision Tree', DecisionTreeClassifier(random_state=42)),
    ('Naive Bayes', GaussianNB()),
    ('Neural Network', MLPClassifier(random_state=42))
]

# 创建绘制ROC曲线的图表
plt.figure(figsize=(10, 8))

# 设置十折交叉验证
kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

# 初始化均匀间隔点
mean_fpr = np.linspace(0, 1, 100)

# 遍历每个模型并进行交叉验证
for name, model in models:
    tprs = []
    aucs = []

    # 进行十折交叉验证
    for train_index, val_index in kf.split(X_train, y_train):
        # 获取训练集和验证集
        X_cv_train, X_cv_val = X_train.iloc[train_index], X_train.iloc[val_index]
        y_cv_train, y_cv_val = y_train.iloc[train_index], y_train.iloc[val_index]

        # 训练模型
        model.fit(X_cv_train, y_cv_train)

        # 预测概率
        y_probs = model.predict_proba(X_cv_val)[:, 1]

        # 计算ROC曲线
        fpr, tpr, _ = roc_curve(y_cv_val, y_probs)
        roc_auc = auc(fpr, tpr)

        # 将当前轮次的FPR和TPR插值到均匀间隔点上
        tprs.append(np.interp(mean_fpr, fpr, tpr))
        tprs[-1][0] = 0.0
        aucs.append(roc_auc)

    # 计算95%置信区间
    auc_ci = np.percentile(aucs, [2.5, 97.5])
    print(f'{name} - AUC: {np.mean(aucs):.4f} ({auc_ci[0]:.4f}, {auc_ci[1]:.4f})')

    # 绘制ROC曲线
    plt.plot(mean_fpr, np.mean(tprs, axis=0), label=f'{name} - AUC: {np.mean(aucs):.4f} ({auc_ci[0]:.4f}, {auc_ci[1]:.4f})')

# 添加随机猜测线
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')

# 设置图表标签
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc='lower right')


# 保存图片到桌面并显示
output_folder = 'C:/Users/Administrator/Desktop'
if not os.path.exists(output_folder):
    os.makedirs(output_folder)
plt.savefig(os.path.join(output_folder, 'roc_curve.png'))
plt.show()
