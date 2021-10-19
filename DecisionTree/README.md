# Decision Tree Directory
This is an implementation of an ID3 decision tree algorithm.
It is possible to use heuristics of entropy, majority error, and gini index to determine what attribute to split the data at each branch.

## decision_tree.py
### DecisionTree 
Can construct a `DecisionTree` object given a description of all possible attribute values for any datasset. 
Then can train by calling `train` given examples to train on (`S`), a definition of all possible attribute values in this training data (`attributes`), labels for each example in S (`labels`), and optionally a desired hueristic (`metric`) and a max depth to grow the tree (`max_depth`).
See docstring for more information.
(Side note: Contains optional parameters to specify weighted construction of a decision tree and/or random tree attribute sampling, the docstring provides more information on this, or the ensemble methods of the Ensemble Learning directory utilize these configurations of a decision tree.) 
### Node 
Create a Node that can be used to build trees. 
A Node can have multiple children and each child has an associated weight that defines the path connecting this Node to the child Node.
See docstring for more information.

## decision_tree_test.py
Contains unit testing with simple decision tree example.

## homework1.py
Uses `DecisionTree` to train a decision tree on a bank and car dataset as required in homework 1.
This file is meant to be run interactively running cells sequentially and can even be converted to a Jupyter notebook if (cells are denotes by # %%).
The file can be run as a .py file though.
