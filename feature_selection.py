import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LassoCV
from sklearn.model_selection import train_test_split, cross_val_score, KFold, RepeatedKFold, GridSearchCV
from scipy.stats import pearsonr, ttest_ind, levene
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
import seaborn as sns # 作图包
import matplotlib.pyplot as plt # 作图包
from sklearn.metrics import roc_curve, roc_auc_score, classification_report # ROC 曲线 AUC 分类报告
import matplotlib.ticker as ticker   #设置横纵坐标的间隔的函数



xlsx1_filepath = 'C:/Users/Administrator/Desktop/lz3IAC.xlsx'
xlsx2_filepath = 'C:/Users/Administrator/Desktop/lz3MIA.xlsx'
data_1 = pd.read_excel(xlsx1_filepath)
data_2 = pd.read_excel(xlsx2_filepath)

rows_1, _ = data_1.shape
rows_2, _ = data_2.shape

data_1.insert(0, 'label', [0] * rows_1)
data_2.insert(0, 'label', [1] * rows_2)

data = pd.concat([data_1, data_2])

# # Reset the index to ensure unique row indices after shuffling and concatenation此处不应该混序，不然每次运行得到的data都是不一样的顺序，
# 后续的分组即使选择了random_state也不能得到固定结果，每次分组都是随机的
# data = shuffle(data).reset_index(drop=True)

# data = data.fillna(0)


# # Save the data DataFrame as an Excel file on the desktop
# desktop_filepath = 'C:/Users/Administrator/Desktop/data.xlsx'
# data.to_excel(desktop_filepath, index=False)

# print("Data DataFrame has been saved to:", desktop_filepath)

# random_state的作用是第一次分组是随机的，以后的每次运行都是得到第一次的随机分组结果，这样实现实验的可重复
data_train, data_test = train_test_split(data, test_size=0.3,random_state=15,stratify=data['label'])

# 显示训练集和测试集的0和1的个数
data_train_a = data_train[:][data_train['label'] == 0]
data_train_b = data_train[:][data_train['label'] == 1]
data_test_a = data_test[:][data_test['label'] == 0]
data_test_b = data_test[:][data_test['label'] == 1]
print(data_train_a.shape)
print(data_train_b.shape)
print(data_test_a.shape)
print(data_test_b.shape)




print(data_train.shape,data_test.shape)

print(data_train.head())

# # Save the data_train as an Excel file on the desktop
# desktop_filepath = 'C:/Users/Administrator/Desktop/data_train.xlsx'
# data_train.to_excel(desktop_filepath, index=False)

# print("Data DataFrame has been saved to:", desktop_filepath)


# Separate the labels from the features in data_train
# X_train = data_train.drop(columns=['label'])
X_train = data_train[data.columns[1:]]
y_train = data_train['label']

# Standardize the features using Z-score normalization
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# Convert the scaled features back to a DataFrame
data_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns)

# Combine the scaled features with the labels
data_train_scaled['label'] = y_train.tolist()

# 将label列移到第一列
data_train_scaled = data_train_scaled[['label'] + [col for col in data_train_scaled.columns if col != 'label']]

# Now, data_train_scaled contains the Z-score normalized data_train
print(data_train_scaled.head())

# # Save data_train_scaled DataFrame as an Excel file
# output_filepath_train = 'C:/Users/Administrator/Desktop/data_train_scaled.xlsx'
# data_train_scaled.to_excel(output_filepath_train, index=False)

# print("data_train_scaled has been saved to:", output_filepath_train)



#t检验特征筛选，从data_train_scaled里
# Separate the labels from the features in data_train_scaled
X_train_scaled = data_train_scaled[data.columns[1:]]
y_train = data_train_scaled['label']

# Perform t-test feature selection
t_test_p_values = []
selected_features_t_test = []
for col in X_train_scaled.columns:
    t_stat, p_value = ttest_ind(X_train_scaled[y_train == 0][col], X_train_scaled[y_train == 1][col])
    if p_value < 0.05:
        t_test_p_values.append((col, p_value))
        selected_features_t_test.append(col)

# Display the selected features based on t-test p-values
print("Selected Features based on t-test p-values:")
print(selected_features_t_test)

print(len(selected_features_t_test))



# Extract selected features from the standardized train dataset
X_train_selected = data_train_scaled[selected_features_t_test]
train_t_test = pd.concat([X_train_selected, data_train_scaled['label']], axis=1)

# 将label列移到第一列
train_t_test = data_train_scaled[['label'] + [col for col in data_train_scaled.columns if col != 'label']]

# Save train_jianmo DataFrame as an Excel file on the desktop
desktop_filepath_train_t_test = 'C:/Users/Administrator/Desktop/train_t_test.xlsx'
train_t_test.to_excel(desktop_filepath_train_t_test, index=False)

print("train_t_test DataFrame has been saved to:", desktop_filepath_train_t_test)


import pandas as pd
import mrmr
from sklearn.datasets import make_classification
from mrmr import mrmr_classif

# 读入数据
data = pd.read_excel("C:/Users/Administrator/Desktop/train_t_test.xlsx")

# 假设目标变量列名为'label'
X = data.drop('label', axis=1)
y = data['label']

# 使用mrmr进行特征选择
selected_features = mrmr_classif(X=X, y=y, K=20)  # K表示要选择的特征数量，可以根据需要调整

print("Selected features:", selected_features)

# Separate the selected features from X_train_scaled

Selected_features = selected_features
X_train_selected = X_train_scaled[Selected_features]

# Perform Lasso feature selection on the selected features
alphas = np.logspace(-3, 1, 50)
model_lassoCV = LassoCV(alphas=alphas, cv=10, max_iter=100000).fit(X_train_selected, y_train)

# Get the selected features after Lasso feature selection
selected_features_lasso = X_train_selected.columns[model_lassoCV.coef_ != 0]

# Display the selected features after Lasso feature selection
print("Selected Features after Lasso feature selection:")
print(selected_features_lasso)


print(model_lassoCV.alpha_)
coef = pd.Series(model_lassoCV.coef_, index=X_train_selected.columns)
print("Lasso picked" + str(sum(coef !=0)) + "variables and eliminated the other" + str(sum(coef == 0)))


# 画lasso的图# 画lasso的图# 画lasso的图# 画lasso的图# 画lasso的图# 画lasso的图


MSEs = (model_lassoCV.mse_path_)

MSEs_mean = np.apply_along_axis(np.mean,1,MSEs)
MSEs_std = np.apply_along_axis(np.std,1,MSEs)


plt.figure(dpi=300)    #dpi = 300   #dpi是发文章时需要放到括号里，输出高清图，可直接复制粘贴到文章

# Plot Lasso path   # LASSO 模型中 Lambda 选值图
coefs = model_lassoCV.path(X_train_selected, y_train, alphas=alphas)[1].T

plt.errorbar(model_lassoCV.alphas_,MSEs_mean      #x,y数据，一一对应
            , yerr=MSEs_std    #y误差范围
            , fmt="o"  #数据点标记
            , ms=3   #数据点大小
            , mfc="r"   #数据点颜色
            , mec="r"   #数据点边缘颜色
            , ecolor="lightblue"   #误差棒颜色
            , elinewidth=2     #误差棒线宽
            , capsize=4   #误差棒边界线长度
            , capthick=1)  #误差棒边界线厚度
plt.semilogx()
plt.axvline(model_lassoCV.alpha_,color = 'black',ls="--")
plt.xlabel('Lambda')
plt.ylabel('MSE')
ax=plt.gca()
y_major_locator=ticker.MultipleLocator(0.05)  #设置y轴的单位分隔 
ax.yaxis.set_major_locator(y_major_locator)

# 保存图片到桌面并显示
import os
output_folder = 'C:/Users/Administrator/Desktop'
if not os.path.exists(output_folder):
    os.makedirs(output_folder)
plt.savefig(os.path.join(output_folder, 'Lambda选值图.png'))
plt.show()



# 特征系数随 Lambda 变化曲线
coefs = model_lassoCV.path(X_train_selected, y_train, alphas=alphas)[1].T
plt.figure(dpi=300)    #dpi = 300   #dpi是发文章时需要放到括号里，输出高清图，可直接复制粘贴到文章
plt.semilogx(model_lassoCV.alphas_,coefs, '-')
plt.axvline(model_lassoCV.alpha_,color = 'black',ls="--")
plt.xlabel('Lambda')
plt.ylabel('Coefficients')

# 保存图片到桌面并显示
import os
output_folder = 'C:/Users/Administrator/Desktop'
if not os.path.exists(output_folder):
    os.makedirs(output_folder)
plt.savefig(os.path.join(output_folder, 'Lasso图.png'))
plt.show()



# 特征权重图
plt.figure(dpi=300)
x_values = np.arange(len(selected_features_lasso))  # X 值的范围（0-特征数长度）
y_values = coef[coef != 0]        # Y 值 LASSO 筛选后系数不为零的值

# 纵向的柱状图 bar()
plt.bar(x_values, y_values        #横向柱状图使用：barh()
        , color = 'lightblue'     #设置 bar 的颜色
        , edgecolor = 'black'     #设置 bar 边框颜色
        , alpha = 0.8             #设置不透明度
       )
# X 轴的特征名称设置
plt.xticks(x_values               # X 值
           , selected_features_lasso  # 特征名
           , rotation=45       # 旋转 xticks 旋转 45°
           , ha = 'right'         # xticks 的水平对齐方式
           , va = 'top'           # xticks 的垂直对齐方式
          )
plt.xlabel("feature")             # 横轴名称
plt.ylabel("weight")              # 纵轴名称

# 保存图片到桌面并显示
import os
output_folder = 'C:/Users/Administrator/Desktop'
if not os.path.exists(output_folder):
    os.makedirs(output_folder)
plt.savefig(os.path.join(output_folder, '权重图.png'))
plt.show()




# 特征相关热图

# Extract the selected features from X_train_selected
X_train_selected_lasso = X_train_selected[selected_features_lasso]

plt.figure(figsize=(12,10), dpi= 300)  # figsize 图片宽高，dpi 分辨率
sns.heatmap(X_train_selected_lasso.corr()            # 计算特征间的相关性
            , xticklabels=X_train_selected_lasso.corr().columns
            , yticklabels=X_train_selected_lasso.corr().columns
            , cmap='RdYlGn'   # 配色方案
            , center=0.5      # 颜色分布的中间值
            , annot=True)     # 是否标出相关性系数，默认 False
plt.title('Correlogram of features', fontsize=22) 
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)


# 保存图片到桌面并显示
import os
output_folder = 'C:/Users/Administrator/Desktop'
if not os.path.exists(output_folder):
    os.makedirs(output_folder)
plt.savefig(os.path.join(output_folder, '热图.png'))
plt.show()

