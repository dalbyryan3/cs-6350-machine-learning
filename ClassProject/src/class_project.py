# %%
import pickle
from datetime import date, datetime
import numpy as np
import pandas as pd
from pathlib import Path
from pandas.core.arrays import categorical
from sklearn import preprocessing
from sklearn import linear_model
from sklearn import svm
from sklearn import model_selection

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

def encode_data(X, categorical_cols_idxs, enc):
    X_categorical = X[:,categorical_cols_idxs]
    X_non_categorical = np.delete(X, categorical_cols_idxs, axis=1)
    X_categorical_enc = enc.transform(X_categorical)
    X_enc = np.concatenate((X_non_categorical,X_categorical_enc), axis=1)
    return X_enc
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

X_full_train_enc = encode_data(X_full_train, categorical_cols_idxs, enc)
X_final_test_enc = encode_data(X_final_test, categorical_cols_idxs, enc)

X_train, X_test, y_train, y_test = model_selection.train_test_split(X_full_train_enc, y_full_train, test_size=0.25)

# %%
# Logistic regression
log_model = linear_model.SGDClassifier(loss='log')
log_model.fit(X_train, y_train)
train_acc = log_model.score(X_train, y_train)
test_acc = log_model.score(X_test, y_test)
print('Logistic regression: test accuracy = {0}, train accuracy = {1}'.format(train_acc, test_acc))

# %% 
# Output prediction and save model
output_kaggle_prediction_and_save_model(log_model, X_final_test_enc, final_idxs)

# %%
# Linear support vector machine classification 
svc_model = linear_model.SGDClassifier(loss='hinge')
svc_model.fit(X_train, y_train)
train_acc = svc_model.score(X_train, y_train)
test_acc = svc_model.score(X_test, y_test)
print('Linear support vector classification: test accuracy = {0}, train accuracy = {1}'.format(train_acc, test_acc))

# %% 
# Output prediction and save model
output_kaggle_prediction_and_save_model(svc_model, X_final_test_enc, final_idxs)
# %%
