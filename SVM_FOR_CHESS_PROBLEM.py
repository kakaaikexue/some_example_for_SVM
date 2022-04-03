from numpy import float32
import pandas as pd
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix,roc_curve,auc
import seaborn as sns
import matplotlib.pyplot as plt
# 读取数据
data = pd.read_csv('krkop.DATA',header=None)
data.dropna(inplace=True)
# 将样本数值化
for i in [0,2,4]:
    data.loc[data[i]=='a',i] = 1
    data.loc[data[i]=='b',i] = 2
    data.loc[data[i]=='c',i] = 3
    data.loc[data[i]=='d',i] = 4
    data.loc[data[i]=='e',i] = 5
    data.loc[data[i]=='f',i] = 6
    data.loc[data[i]=='g',i] = 7
    data.loc[data[i]=='h',i] = 8
# 将标签数值化
data.loc[data[6]!='draw',6] = -1    
data.loc[data[6]=='draw',6] = 1
# 归一化处理
for i in range(6):
    data[i] = (data[i]-data[i].mean())/data[i].std()  
# 拆分训练集和测试集
X_train, X_test, y_train, y_test =  train_test_split(data.iloc[:,:6],data[6],test_size=0.82178500142572)
X_train = X_train.to_numpy()
y_train = y_train.to_numpy().astype(int)
X_test = X_test.to_numpy()
y_test = y_test.to_numpy().astype(int)
# 寻找C和gamma的粗略范围
CScale = [i for i in range(100,201,10)]
gammaScale = [i/10 for i in range(1,11)]
cv_scores = 0

for i in CScale:
    for j in gammaScale:
        model = SVC(kernel = 'rbf', C = i,gamma=j)
        scores = cross_val_score(model,X_train, y_train,cv =5,scoring = 'accuracy', error_score='raise')
        if scores.mean()>cv_scores:
            cv_scores = scores.mean()
            savei = i
            savej = j*100
        

# 找到更精确的C和gamma
CScale = [i for i in range(savei-5,savei+5)]
gammaScale = [i/100+0.01 for i in range(int(savej)-5,int(savej)+5)]
cv_scores = 0
for i in CScale:
    for j in gammaScale:
        model = SVC(kernel = 'rbf', C = i,gamma=j)
        scores = cross_val_score(model,X_train, y_train,cv =5,scoring = 'accuracy')
        if scores.mean()>cv_scores:
            cv_scores = scores.mean()
            savei = i
            savej = j
#将确定好的参数重新建立svm模型
model = SVC(kernel = 'rbf', C=savei,gamma=savej)
model.fit(X_train, y_train)
pre = model.predict(X_test)
model.score(X_test,y_test)
# 绘制AUC和EER图形
cm = confusion_matrix(y_test, pre, labels=[-1, 1], sample_weight=None)
sns.set()
f,ax=plt.subplots()
sns.heatmap(cm,annot=True,ax=ax) #画热力图
ax.set_title('confusion matrix') #标题
ax.set_xlabel('predict') #x轴
ax.set_ylabel('true') #y轴
fpr,tpr,threshold = roc_curve(y_test, pre) ###计算真正率和假正率
roc_auc = auc(fpr,tpr) ###计算auc的值，auc就是曲线包围的面积，越大越好
# plt.figure()
lw = 2
plt.figure(figsize=(10,10))
plt.plot(fpr, tpr, color='darkorange',lw=lw, label='ROC curve (area = %0.2f)' % roc_auc) ###假正率为横坐标，真正率为纵坐标做曲线
plt.plot([0, 1], [1, 0], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")
plt.show()



