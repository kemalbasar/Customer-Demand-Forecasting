




#pip install --upgrade pip

# Libraries
#########################################################

import numpy as np
from numpy import loadtxt
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, accuracy_score, r2_score, classification_report, roc_auc_score
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import train_test_split, GridSearchCV, cross_validate, cross_val_score, validation_curve, KFold
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn import model_selection, neighbors

pip install catboost
pip install xgboost
pip install lightgbm
pip install pydotplus
pip install skompiler
pip install astor
pip install joblib

pip install libomp
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from xgboost import XGBClassifier, XGBRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier, export_graphviz, export_text
from math import sqrt
from skompiler import skompile

import warnings
import joblib
import pydotplus
import graphviz

pd.set_option('display.max_columns', 30)
pd.set_option('display.width', 170)
pd.set_option('display.max_rows', 50)
pd.set_option('display.float_format', lambda x: '%.3f' % x)

from pandas.core.common import SettingWithCopyWarning
from sklearn.exceptions import ConvergenceWarning

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action="ignore", category=ConvergenceWarning)
warnings.simplefilter(action="ignore", category=SettingWithCopyWarning)

from sklearn.model_selection import GridSearchCV
from tseries_tools import lgbm_smape

orders_final = pd.read_csv(r"data_sources/final_before_model.csv")
# Model Deployment
train = orders_final[orders_final["year_2022"] != 1]
test = orders_final[orders_final["year_2022"] == 1]

features = [col for col in orders_final.columns if col != "GRANDTOTAL_NEW"]

X_train = train[features]
Y_train = train["GRANDTOTAL_NEW"]

X_val = test[features]
Y_val = test["GRANDTOTAL_NEW"]

# Parameter Optimization
# lgb_params = {'num_leaves': 10,
#               'learning_rate': 0.02,
#               'feature_fraction': 0.8,
#               'max_depth': 5,
#               'verbose': 0,
#               'nthread': -1,
#               "num_boost_round": model.best_iteration}

lgbtrain = lgb.Dataset(data=X_train, label=Y_train, feature_name=X_train.columns)
lgbtest = lgb.Dataset(data=X_val, label=Y_val, reference=lgbtrain, feature_name=X_val.columns)

lgbm_model = LGBMRegressor(random_state=46,)

lgb_params = {'num_leaves': [2, 10],
              'learning_rate': [0.01, 0.50],
              'feature_fraction': [0.4, 0.12],
              'max_depth': [2, 8],
              'num_boost_round': [10000, 40000],
              'early_stopping_rounds': [200,250],
              'nthread': [-1]}

lgbm_gs_best_grid = GridSearchCV(lgbm_model,
                                 lgb_params,
                                 cv=3,
                                 n_jobs=-1,
                                 verbose=True).fit(X_train, Y_train)

lgbm_gs_best_grid.best_params_

lgbm_gs_best_grid.best_score_

model = lgb.train(lgb_params, lgbtrain,
                  valid_sets=[lgbtrain, lgbtest],
                  num_boost_round=lgb_params['num_boost_round'],
                  early_stopping_rounds=lgb_params['early_stopping_rounds'],
                  feval=lgbm_smape,
                  verbose_eval=100)

test_preds = model.predict(X_val, num_iteration=model.best_iteration)
