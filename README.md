# Regression Tool
A multi-parameter, multi-model k-fold grid search. This code search for single-values solutions as well estimate a Probability Density Function(PDF) for each solution. Also, we have a implemented an artificial Neural Network.
## Modules

### data_preprocessing.py
A template to manipulate data and discretize the y (target). The number of bins is ,by default, 100.

### Regression.py
The regression module containing the regression class and the PDF as well.

### NNRegression.py
The regression artificial neural network module for  module containing the regression class and the PDF as well. Depending on the dataset, there is a need a customization.

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
Generate a statistics.csv with the metrics and parameters of the hyper parametrization ANN.
### Tutorial.ipynb
There is a tutorial for how to implement both methods and how to use the PDF to implement for another estimation method. Here, we use the information from the NNRegression.py and create and customize.

You need to set:

- Classifiers and Regressors(comment out the ones you do not wish to include)
- Grid search parameters

See the Tutorial.