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



xlsx1_filepath = os.path.join(os.path.expanduser('~'), 'Desktop', '666', 'MIA.xlsx')
xlsx2_filepath = os.path.join(os.path.expanduser('~'), 'Desktop', '666', 'IAC.xlsx')
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


# Save the data DataFrame as an Excel file on the desktop
desktop_filepath = os.path.join(os.path.expanduser('~'), 'Desktop', '666', 'MIA和IAC混合.xlsx')
data.to_excel(desktop_filepath, index=False)

print("Data DataFrame has been saved to:", desktop_filepath)

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

# Save the data_train as an Excel file on the desktop
desktop_filepath = os.path.join(os.path.expanduser('~'), 'Desktop', '666', '训练组.xlsx')
data_train.to_excel(desktop_filepath, index=False)

print("Data DataFrame has been saved to:", desktop_filepath)

# Save the data_test as an Excel file on the desktop
desktop_filepath = os.path.join(os.path.expanduser('~'), 'Desktop', '666', '测试组.xlsx')
data_test.to_excel(desktop_filepath, index=False)

print("Data DataFrame has been saved to:", desktop_filepath)
