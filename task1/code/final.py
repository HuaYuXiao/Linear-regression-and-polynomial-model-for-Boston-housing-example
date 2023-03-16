import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split  # 导入数据集划分模块
from sklearn.metrics import r2_score
from skimage.metrics import mean_squared_error
 
#数据集
path = 'data.csv'
# 使用pandas读入
data = pd.read_csv(path) #读取文件中所有数据

# 按列分离数据
x = data[['CRIM']]#读取某列
x=np.array(x).astype('float32')

y = data[['MEDV']]#读取某列
y=np.array(y).astype('float32')

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.5, random_state=0)
 
# 线性回归模型
model = LinearRegression()
model.fit(x_train, y_train)  # 训练数据,学习模型参数
y_predict = model.predict(x_test)  # 预测

# 与验证值作比较
error = mean_squared_error(y_test, y_predict).round(5)  # 平方差
score = r2_score(y_test, y_predict).round(5)  # 相关系数

#画出结果
print("d=",error)
print("R^2=",score)
print("w=",model.coef_)
print("b=",model.intercept_)

plt.plot(x_train,y_train,'o')
plt.plot(x_test,y_predict)
plt.legend(['true', 'predict'])
plt.show()
