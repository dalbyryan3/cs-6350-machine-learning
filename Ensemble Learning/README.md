# Ensemble Learning Directory
This is an implementation of various ensemble methods using binary decision trees as the base predictor.

## ensemble.py
### DecisionStump 
A `DecisionStump` is essentially a light wrapper around a `DecisionTree` with a max depth of 1.
See docstring for more information as well as the README.md in the DecisionTree directory.
## AdaBoost
AdaBoosted binary decision stump based model for training and prediction. 
See docstring for more information.
## BaggedDecisionTree
Bagged binary decision tree model built from out-of-bag samples for training and prediction. 
See docstring for more information.
## RandomForest
Random forest with binary decision trees built from out-of-bag samples and using random small attribute subsets as candidates to split on. 
See docstring for more information.

## homework2_ensemble.py
Uses `AdaBoost`, `BaggedDecisionTree`, and `RandomForest` ensemble methods to complete the tasks outlined in homework 2 part 2 questions 2 the bank dataset.
This file is meant to be run interactively running cells sequentially and can even be converted to a Jupyter notebook if desired (cells are denoted by # %%).
Cells with long run times are indicated in all-caps.
The results from these cells were saved as a pickle object and can be loaded using specified cells (meaning can run rest of code).
The file can be run as a .py file as well but take very long due to cells with long run times.