# Linear-regression-and-polynomial-model-for-Boston-housing-example
SUSTech EE271 Artificial Intelligence and Machine Learning

Each record in the database (boston_housing_data.csv) describes a Boston suburb or town. The data was drawn from the Boston Standard Metropolitan Statistical Area (SMSA) in 1970. The attributes are defined as follows. 

CRIM: per capita crime rate by town
ZN: proportion of residential land zoned for lots over 25,000 sq.ft.
INDUS: proportion of non-retail business acres per town
CHAS: Charles River dummy variable (= 1 if tract bounds river; 0 otherwise)
NOX: nitric oxides concentration (parts per 10 million)
RM: average number of rooms per dwelling
AGE: proportion of owner-occupied units built prior to 1940
DIS: weighted distances to ﬁve Boston employment centers
RAD: index of accessibility to radial highways
TAX: full-value property-tax rate per $10,000
PTRATIO: pupil-teacher ratio by town 
B: 1000(Bk−0.63)2 where Bk is the proportion of blacks by town 
LSTAT: % lower status of the population
MEDV: Median value of owner-occupied homes in $1000s

Homework Assignment:

Consider only one attribute for the Boston housing example (e.g., picking CRIM: per captia crime rate by town, as the unique input) and the continuous output is MEDV (Median value of owner-occupied homes in $1000s). 

1. Get data from boston_housing_data.csv and write a Class called OneDimBostonHousingDataset with CRIM as the input and MEDV as the target by inheriting from torch.utils.data.Dataset.
2. Use DataLoader to make a training set called train_loader and a validation set called validation_loader.
3. Create a linear regression model and MSE loss function from torch.nn, and use automatic differentiation from PyTorch to train the model. Then plot the loss function for both the training data and the validation data.
d.	Create a polynomial model with an appropriate order, define a MSE loss function, and write the gradient descent update law for the learning weights by yourself, without using torch.nn, torch.autograd, torch.optim. Then plot the loss function for both the training data and the validation data.

Consider three-dimensional inputs for the Boston housing example (namely, picking CRIM, RM, and AGE as the three inputs) and the continuous output is MEDV (Median value of owner-occupied homes in $1000s). 

1. Get data from boston_housing_data.csv and write a Class called ThreeDimBostonHousingDataset with CRIM, RM and AGE as the inputs and MEDV as the target by inheriting from torch.utils.data.Dataset.
2. Use DataLoader to make a training set called train_loader and a validation set called validation_loader.
3. Create a polynomial model with an appropriate order for three dimensional inputs, define a MSE loss function, and write the gradient descent update law for the learning weights by yourself, without using torch.nn, torch.autograd, torch.optim. Then plot the loss function for both the training data and the validation data.

For more information, please contact linzy@sustech.edu.cn or 1628280289@qq.com.
