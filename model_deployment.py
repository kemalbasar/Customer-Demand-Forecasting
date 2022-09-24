import warnings

import numpy as np
import pandas as pd
import lightgbm as lgb
from lightgbm import LGBMRegressor
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Lasso, Ridge, LinearRegression, ElasticNet
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
import pre_tools as pt
import mltools as ml

warnings.filterwarnings('ignore')

from tseries_tools import lgbm_smape

orders_final = pd.read_csv(r"C:\Users\kereviz\PycharmProjects\Customer Demand "
                           r"Forecasting\data_sources\final_before_model.csv")

pt.replace_missing_values(orders_final, '0')

# Model Deployment
train = orders_final[orders_final["year_2022"] != 1]
test = orders_final[orders_final["year_2022"] == 1]

features = [col for col in orders_final.columns if col != "GRANDTOTAL_NEW"]

X_train = train[features]
Y_train = train["GRANDTOTAL_NEW"]

X_val = test[features]
Y_val = test["GRANDTOTAL_NEW"]

#############################################################################

models = [("Lasso", Lasso()),
          ("ElasticNet", ElasticNet()),
          ('CART', DecisionTreeRegressor()),
          ('RF', RandomForestRegressor()),
          ('GBM', GradientBoostingRegressor())]
# ("CatBoost", CatBoostRegressor(verbose=False))]

for name, regressor in models:
    rmse = np.mean(np.sqrt(-cross_val_score(regressor, X_train, Y_train, cv=5, scoring="neg_mean_squared_error")))
    print(f"RMSE: {round(rmse, 4)} ({name}) ")

# RMSE Scores

# RMSE: 1.4876 (Lasso)
# RMSE: 1.4875 (ElasticNet)
# RMSE: 1.0495 (CART)
# RMSE: 0.8231 (RF)
# RMSE: 0.8878 (GBM)

for name, regressor in models:
    r2 = cross_val_score(regressor, X_train, Y_train, cv=5, scoring="r2")
    print(f"r2: {r2.mean()} ({name}) ")

# r2 Scores
# r2: 0.049893890420212084 (Lasso)
# r2: 0.05005194680266856 (ElasticNet)
# r2: 0.5250222175158527 (CART)
# r2: 0.6966254272430534 (RF)
# r2: 0.6636575032942984 (GBM)

#############################################################################
# LGB Grid Search Optimization

orders_final = pd.read_csv(r"C:\Users\kereviz\PycharmProjects\Customer Demand "
                           r"Forecasting\data_sources\final_before_model.csv")

lgbtrain = lgb.Dataset(data=X_train, label=Y_train, feature_name=X_train.columns)
lgbtest = lgb.Dataset(data=X_val, label=Y_val, reference=lgbtrain, feature_name=X_val.columns)

lgbm_model = LGBMRegressor()

lgb_params = {'num_leaves': [2, 10],
              'learning_rate': [0.01, 0.50],
              'feature_fraction': [0.4, 0.12],
              'max_depth': [4, 5],
              'num_boost_round': [10000, 40000],
              'nthread': [-1]}

lgbm_gs_best_grid = GridSearchCV(lgbm_model,
                                 lgb_params,
                                 cv=3,
                                 scoring='r2',
                                 n_jobs=-1,
                                 verbose=True).fit(X_train, Y_train, eval_set=(X_val, Y_val))

lgbm_gs_best_grid.best_params_

# Best Params.
# {'feature_fraction': 0.4,
#  'learning_rate': 0.01,
#  'max_depth': 1,
#  'nthread': -1,
#  'num_boost_round': 10000,
#  'num_leaves': 2}

lgbm_gs_best_grid.best_score_
# Best Score
# 0.5699196578038311

# model = lgbm_model.set_params(**lgbm_gs_best.best_params_).fit(X, y
# model = lgb.train(lgb_params, lgbtrain,
#                   valid_sets=[lgbtrain, lgbtest],
#                   num_boost_round=lgb_params['num_boost_round'],
#                   early_stopping_rounds=lgb_params['early_stopping_rounds'],
#                   feval=lgbm_smape,
#                   verbose_eval=100)
#
# test_preds = model.predict(X_val, num_iteration=model.best_iteration)

##################################################################################
##################################################################################
# Random Forest Regressor

## Random Forest Model

# Initializing the Random Forest Regression model with 10 decision trees
rf_model = RandomForestRegressor(n_estimators=10, random_state=0)

# Fitting the Random Forest Regression model to the data
# rf_model.fit(X, y)
rf_model.fit(X_train, Y_train)

# Predicting the target values of the test set
y_pred = rf_model.predict(X_val)

# RMSE (Root Mean Square Error)
rmse = float(format(np.sqrt(mean_squared_error(Y_val, y_pred)), '.3f'))
r2 = float(format(r2_score(Y_val, y_pred), '.3f'))
print("RMSE: ", rmse)
print('r2: ', r2)

rf_model.get_params()

rf_params = {"max_depth": [3],
             "max_features": [10,12,14,16,18,21,25,"auto",42],
             "min_samples_split": [8],
             "n_estimators": [50]}

rf_best_grid = GridSearchCV(rf_model, rf_params, cv=5, n_jobs=-1, verbose=True).fit(X_train, Y_train)

rf_best_grid.best_params_
rf_best_grid.best_score_

rf_final = rf_model.set_params(**rf_best_grid.best_params_, random_state=17).fit(X_train, Y_train)

# RMSE (Root Mean Square Error)
Y_pred = pd.DataFrame(rf_final.predict(X_val))
Y_pred.columns = ['GRANDTOTAL_NEW']
rmse = float(format(np.sqrt(mean_squared_error(Y_val, y_pred)), '.3f'))
r2 = float(format(r2_score(Y_val, y_pred), '.3f'))
print("RMSE: ", rmse)
print('r2: ', r2)

Y_val = Y_val.reset_index()
Y_val.drop("index",axis=1,inplace=True)
ml.plot_co2(train,Y_val,Y_pred,"Pred Results")

