import numpy as np
from skimage.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression  # 导入线性模型
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split  # 导入数据集划分模块
import matplotlib.pyplot as plt
import pandas as pd

#数据集
path = 'data.csv'
# 使用pandas读入
data = pd.read_csv(path) #读取文件中所有数据

# 按列分离数据
x = data[['CRIM','RM','AGE']]#读取某三列
x=np.array(x).astype('float32')

y = data[['MEDV']]#读取某列
y=np.array(y).astype('float32')

# 将数据进行拆分，一份用于训练，一份用于测试和验证
# 测试集大小为30%,防止过拟合
# 这里的random_state就是为了保证程序每次运行都分割一样的训练集和测试集。
# 否则，同样的算法模型在不同的训练集和测试集上的效果不一样。
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.5, random_state=0)

# 线性回归模型
model = LinearRegression()
model.fit(x_train, y_train)  # 训练数据,学习模型参数
y_predict = model.predict(x_test)  # 预测

# 与验证值作比较
error = mean_squared_error(y_test, y_predict).round(5)  # 平方差
score = r2_score(y_test, y_predict).round(5)  # 相关系数

# 绘制真实值和预测值的对比图
print("d=",error)
print("R^2=",score)
print("w=",model.coef_)
print("b=",model.intercept_)

xx = np.arange(0, 50)
yy = xx
plt.plot(xx, yy)
plt.scatter(y_test, y_predict,color='red')
plt.show()
