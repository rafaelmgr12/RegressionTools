# Regression Tool
A multi-parameter, multi-model k-fold grid search for Machine Learning algorithims. This code search for single-values solutions as well estimate a Probability Density Function for each solution. Also, we have a implemented an artificial Neural Network.
## Modules

### data_preprocessing.py
A template to manipulate data and discretize the y (target). The bins default is 100.

### Regression.py
The regression module containing the regression class and the PDF as well.

### NNRegression.py
The regression artificial neural network module for  module containing the regression class and the PDF as well. Depending on the dataset, there is need a customization.

Current compatible regression and classifiers.

- DecisionTreeClassifier
- KNeighborsClassifier  
- ExtraTreeClassifier 
- DecisionTreeRegressor
- RandomForestRegressor
- GradientBoostingRegressor
- AdaBoostRegressor
- XGBRegressor

## Run
### NNRegression.py
Generate a statistics.csv with the metrics and parameters of the hyper parametrization
### Tutorial.ipynb
There is a tutorial for how to implement both methods and how to use the PDF to implement another estimation methods. Here, we use the information from the NNRegression.py and create and customization.

You need to set:

- Classifiers and Regressors(comment out the ones you do not wish to include)
- Grid search parameters