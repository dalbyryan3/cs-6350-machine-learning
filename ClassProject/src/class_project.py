# %%
import pickle
from datetime import datetime
import numpy as np
import pandas as pd
from pathlib import Path
import itertools
import matplotlib.pyplot as plt

from sklearn import preprocessing
from sklearn import linear_model
from sklearn import model_selection
from sklearn import metrics
from sklearn import svm
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV

# %% Config variables
data_root_path = Path('../data')
kaggle_submissions_path = Path('../kaggle_submissions')
models_path = Path('../models')

# %% General functions
def output_kaggle_prediction_and_save_model(model, X_final_test, final_idxs):
    y_final = model.predict_proba(X_final_test)[:,1]
    final_test_pred_df = pd.DataFrame({'ID':final_idxs, 'Prediction':y_final})
    filename = datetime.now().strftime('%Y-%m-%d-%H:%M:%S-submission')
    final_test_pred_df.to_csv(kaggle_submissions_path / '{0}'.format(filename+'.csv'), index=False)
    with open(models_path / '{0}'.format(filename+'.pkl'),'wb') as f:
        pickle.dump(model,f)
    return filename

def load_model(filename):
    with open(models_path / '{0}'.format(filename+'.pkl'),'rb') as f:
        return pickle.load(f)

def encode_data(X, categorical_cols_idxs, enc):
    X_categorical = X[:,categorical_cols_idxs]
    X_non_categorical = np.delete(X, categorical_cols_idxs, axis=1)
    X_categorical_enc = enc.transform(X_categorical)
    X_enc = np.concatenate((X_non_categorical,X_categorical_enc), axis=1)
    return X_enc

def evaluate_model(model, X_train, y_train, X_test, y_test, model_name_str='', should_print=True):
    # Evaluates model in terms of test and train accuracy 
    # Train data evaluation
    y_train_pred = model.predict(X_train)
    y_train_pred_prob = model.predict_proba(X_train)[:,1]
    train_acc = metrics.accuracy_score(y_train, y_train_pred)
    train_auc = metrics.roc_auc_score(y_train, y_train_pred_prob)

    # Test data evaluation
    y_test_pred = model.predict(X_test)
    y_test_pred_prob = model.predict_proba(X_test)[:,1]
    test_acc = metrics.accuracy_score(y_test, y_test_pred)
    test_auc = metrics.roc_auc_score(y_test, y_test_pred_prob)

    if should_print:
        print('{0} model: train accuracy = {1}, train AUC = {2} \ntest accuracy = {3}, test AUC = {4}'.format(model_name_str, train_acc, train_auc, test_acc, test_auc))
    return train_acc, train_auc, test_acc, test_auc

# Learning curve
def create_learning_curves(model, X, y, num_data_points=5, model_name_str=''):
    num_samples_vals, train_scores_5cv, val_scores_5cv = model_selection.learning_curve(model, X, y, train_sizes=np.linspace(0.1, 1.0, num_data_points))
    # Get mean of 5-fold cross validation
    avg_train_scores = np.mean(train_scores_5cv, axis=1)
    avg_val_scores = np.mean(val_scores_5cv, axis=1)
    plt.figure()
    plt.plot(num_samples_vals, avg_train_scores, label='Training score')
    plt.plot(num_samples_vals, avg_val_scores, label='Validation score')
    plt.legend()
    plt.xlabel('Number of examples')
    plt.ylabel('Score (accuracy)')
    plt.title('Learning Curves for {0}'.format(model_name_str))
    plt.show()
    return avg_train_scores, avg_val_scores 
# %%
###
# %%
# Load path 
loaded_train_data = pd.read_csv(data_root_path / 'train_final.csv')
loaded_test_data = pd.read_csv(data_root_path / 'test_final.csv')
X_full_train = loaded_train_data.iloc[:,0:-1].to_numpy()
y_full_train = loaded_train_data.iloc[:,-1].to_numpy()
X_final_test = loaded_test_data.iloc[:,1:].to_numpy()
final_idxs = loaded_test_data.iloc[:,0].to_numpy()

# %%
###
# %%
# Data preprocessing 
is_categorical_data = (loaded_train_data.dtypes == 'object')
categorical_cols = list(is_categorical_data[is_categorical_data].index)
categorical_cols_idxs = [loaded_train_data.columns.get_loc(c) for c in categorical_cols if c in loaded_train_data]
enc = preprocessing.OneHotEncoder(sparse=False, dtype=int)
enc.fit(np.concatenate((X_full_train[:,categorical_cols_idxs],X_final_test[:,categorical_cols_idxs]), axis=0))

# One hot encoding 
X_full_train_enc = encode_data(X_full_train, categorical_cols_idxs, enc)
X_final_test_enc = encode_data(X_final_test, categorical_cols_idxs, enc)

# Make zero mean and unit variance 
data_scaler = preprocessing.StandardScaler()
# Only fit to training data
X_full_train_enc_norm = data_scaler.fit_transform(X_full_train_enc)
X_final_test_enc_norm = data_scaler.transform(X_final_test_enc)

X_train, X_test, y_train, y_test = model_selection.train_test_split(X_full_train_enc_norm, y_full_train, test_size=0.20)

# %%
###
# %%
# Logistic regression
log_model = linear_model.LogisticRegression(solver='lbfgs', max_iter=500) 
log_model_name_str = 'Logistic Regression Classification'
# log_model = linear_model.SGDClassifier(loss='log') 
# SGD fails to converge to good optima sometimes, so lbfgs is used which finds optima without being too computationally expensive for this data
log_model.fit(X_train, y_train)

# %%
# Evaluate logistic regression
log_model_eval = evaluate_model(log_model, X_train, y_train, X_test, y_test, model_name_str=log_model_name_str)

# %% Learning curves 
log_model_train_learning_curve, log_model_val_learning_curve = create_learning_curves(log_model, X_train, y_train, model_name_str=log_model_name_str, num_data_points=100)

# %% Save learning curves 
with open( '{0}'.format('log_learning_curve.pkl'),'wb') as f:
    pickle.dump([log_model_train_learning_curve, log_model_val_learning_curve],f)

# %% 
# Output prediction and save model
output_kaggle_prediction_and_save_model(log_model, X_final_test_enc_norm, final_idxs)

# %%
###
# %%
# Linear support vector machine classification 
svc_model = svm.SVC(kernel='rbf', C=10.0, verbose=True, probability=True)
svc_model.fit(X_train, y_train)
svc_model_name_str = 'Support Vector Machine Classification'

# %%
# Evaluate linear support vector classification 
svc_model_eval = evaluate_model(svc_model, X_train, y_train, X_test, y_test, model_name_str=svc_model_name_str)

# %% Learning Curves 
svc_model_train_learning_curve, svc_model_val_learning_curve = create_learning_curves(svc_model, X_train, y_train, model_name_str=svc_model_name_str, num_data_points=5)

# %% Save learning curves 
with open( '{0}'.format('svc_learning_curve.pkl'),'wb') as f:
    pickle.dump([svc_model_train_learning_curve, svc_model_val_learning_curve],f)

# %% 
# Output prediction and save model
output_kaggle_prediction_and_save_model(svc_model, X_final_test_enc_norm, final_idxs)

# %%
###
# %%
# Logistic regression bagging model
log_bag_model = BaggingClassifier(linear_model.LogisticRegression(solver='lbfgs', max_iter=500), max_samples=1.0, max_features=1.0, bootstrap=True)
log_bag_model.fit(X_train, y_train)
log_bag_model_name_str = 'Bagged Logistic Regression Classification'

log_bag_model_eval = evaluate_model(log_bag_model, X_train, y_train, X_test, y_test, model_name_str=log_bag_model)

# %% Learning Curves 
log_bag_model_train_learning_curve, log_bag_model_val_learning_curve = create_learning_curves(log_bag_model, X_train, y_train, model_name_str=log_bag_model_name_str, num_data_points=10)

# %% Save learning curves 
with open( '{0}'.format('log_bag_learning_curve.pkl'),'wb') as f:
    pickle.dump([log_bag_model_train_learning_curve, log_bag_model_val_learning_curve],f)

# %% 
# Output prediction and save model
output_kaggle_prediction_and_save_model(log_bag_model, X_final_test_enc_norm, final_idxs)

# %%
###

# %%
hidden_layer_size_list = [(100,50)]
learning_rate_init_list = [0.05]
alpha_list = [0.005]
batch_size_list = [256, 512]

best_test_auc = 0
best_params = ()

for hl, lr, alpha, bs in itertools.product(hidden_layer_size_list, learning_rate_init_list, alpha_list, batch_size_list):
    mlp_model = MLPClassifier(hidden_layer_sizes=hl, activation='relu', solver='adam', alpha=alpha, batch_size=bs, learning_rate_init=lr, max_iter=300, early_stopping=True)
    mlp_model.fit(X_train, y_train)
    mlp_model_name_str = 'Fully Connected Neural Network (MLP) Classifier'
    print('Hidden layer shape of {0}, learning rate of {1}, alpha of {2}, and batch size of {3}:'.format(hl, lr, alpha, bs))
    mlp_model_eval = evaluate_model(mlp_model, X_train, y_train, X_test, y_test, model_name_str=mlp_model_name_str)
    _,_,_,test_auc = mlp_model_eval
    print('-----------')
    print()
    if test_auc > best_test_auc:
        best_test_auc =  test_auc
        best_params = (hl, lr, alpha, bs)
print('Best test auc is {0} from hidden layer shape of {1}, learning rate of {2}, alpha of {3}, and batch size of {4}'.format(best_test_auc, *best_params))

# %%
mlp_model = MLPClassifier(hidden_layer_sizes=(100,50), activation='relu', solver='adam', alpha=0.0005, batch_size=128, learning_rate_init=0.008, max_iter=300, early_stopping=True)
mlp_model.fit(X_train, y_train)
mlp_model_name_str = 'Fully Connected Neural Network (MLP) Classifier'
mlp_model_eval = evaluate_model(mlp_model, X_train, y_train, X_test, y_test, model_name_str=mlp_model_name_str)

# %% Learning Curves 
mlp_model_train_learning_curve, mlp_model_val_learning_curve = create_learning_curves(mlp_model, X_train, y_train, model_name_str=mlp_model_name_str, num_data_points=10)

# %% Save learning curves 
with open( '{0}'.format('mlp_learning_curve.pkl'),'wb') as f:
    pickle.dump([mlp_model_train_learning_curve, mlp_model_val_learning_curve],f)


# %% 
# Output prediction and save model
output_kaggle_prediction_and_save_model(mlp_model, X_final_test_enc_norm, final_idxs)

# %%
###
# %%
# Histogram Gradient Boosting Classifier
learning_rate = [0.01, 0.1, 0.15]
param_grid = [
    {'learning_rate':[0.07, 0.1], 'max_depth':[31,61], 'l2_regularization':[1, 15, 30], 'max_leaf_nodes':[15], 'min_samples_leaf':[10,30]}
]

hist_grad_boost_model_grid_search = HistGradientBoostingClassifier(loss='binary_crossentropy', max_iter=200, early_stopping=True)
hist_grad_boost_grid_search = GridSearchCV(hist_grad_boost_model_grid_search, param_grid, scoring='roc_auc', n_jobs=-1)
hist_grad_boost_grid_search_results = hist_grad_boost_grid_search.fit(X_train, y_train)

# %%
print(hist_grad_boost_grid_search_results.best_params_)

# %%
hist_grad_boost_model = HistGradientBoostingClassifier(loss='binary_crossentropy', learning_rate=0.1, max_iter=200, max_depth=31, l2_regularization=15, early_stopping=True, max_leaf_nodes=15, min_samples_leaf=10)
hist_grad_boost_model.fit(X_train, y_train)
hist_grad_boost_model_name_str = 'Histogram Based Gradient Boosting Classifier'

hist_grad_boost_model_eval = evaluate_model(hist_grad_boost_model, X_train, y_train, X_test, y_test, model_name_str=hist_grad_boost_model_name_str)

# %% Learning Curves 
hist_grad_boost_model_train_learning_curve, hist_grad_boost_model_val_learning_curve = create_learning_curves(hist_grad_boost_model, X_train, y_train, model_name_str=hist_grad_boost_model_name_str, num_data_points=10)

# %% Save learning curves 
with open( '{0}'.format('hist_grad_boost_learning_curve.pkl'),'wb') as f:
    pickle.dump([hist_grad_boost_model_train_learning_curve, hist_grad_boost_model_val_learning_curve],f)

# %% 
# Output prediction and save model
output_kaggle_prediction_and_save_model(hist_grad_boost_model, X_final_test_enc_norm, final_idxs)

# %%
###

# %%
# Bagged Histogram Gradient Boosting Classifier 

bag_hist_grad_boost_model = BaggingClassifier(HistGradientBoostingClassifier(loss='binary_crossentropy', learning_rate=0.1, max_iter=200, max_depth=31, l2_regularization=15, early_stopping=True, max_leaf_nodes=15, min_samples_leaf=10), max_samples=1.0, max_features=1.0, bootstrap=False, n_jobs=-1)

bag_hist_grad_boost_model.fit(X_train, y_train)
bag_hist_grad_boost_model_name_str = 'Bagged Histogram Based Gradient Boosting Classifier'

bag_hist_grad_boost_model_eval = evaluate_model(bag_hist_grad_boost_model, X_train, y_train, X_test, y_test, model_name_str=bag_hist_grad_boost_model_name_str)

# %% Learning Curves 
bag_hist_grad_boost_model_train_learning_curve, bag_hist_grad_boost_model_val_learning_curve = create_learning_curves(bag_hist_grad_boost_model, X_train, y_train, model_name_str=bag_hist_grad_boost_model_name_str, num_data_points=10)

# %% Save learning curves 
with open( '{0}'.format('bag_hist_grad_boost_learning_curve.pkl'),'wb') as f:
    pickle.dump([bag_hist_grad_boost_model_train_learning_curve, bag_hist_grad_boost_model_val_learning_curve],f)

# %% 
# Output prediction and save model
output_kaggle_prediction_and_save_model(bag_hist_grad_boost_model, X_final_test_enc_norm, final_idxs)