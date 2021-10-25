# %%
import pickle
from datetime import datetime
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn import preprocessing
from sklearn import linear_model
from sklearn import model_selection
from sklearn import metrics
from sklearn import svm
import matplotlib.pyplot as plt

# %% Config variables
data_root_path = Path('../data')
kaggle_submissions_path = Path('../kaggle_submissions')
models_path = Path('../models')

# %% General functions
def output_kaggle_prediction_and_save_model(model, X_final_test, final_idxs):
    y_final = model.predict(X_final_test)
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
    train_acc = metrics.accuracy_score(y_train, y_train_pred)
    train_auc = metrics.roc_auc_score(y_train, y_train_pred)

    # Test data evaluation
    y_test_pred = model.predict(X_test)
    test_acc = metrics.accuracy_score(y_test, y_test_pred)
    test_auc = metrics.roc_auc_score(y_test, y_test_pred)

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
# Load path 
loaded_train_data = pd.read_csv(data_root_path / 'train_final.csv')
loaded_test_data = pd.read_csv(data_root_path / 'test_final.csv')
X_full_train = loaded_train_data.iloc[:,0:-1].to_numpy()
y_full_train = loaded_train_data.iloc[:,-1].to_numpy()
X_final_test = loaded_test_data.iloc[:,1:].to_numpy()
final_idxs = loaded_test_data.iloc[:,0].to_numpy()

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
svc_model = svm.SVC(kernel='rbf', C=10.0, verbose=True)
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
