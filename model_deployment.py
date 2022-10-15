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
from tools import mltools as ml, pre_tools as pt

warnings.filterwarnings('ignore')

from tools.tseries_tools import lgbm_smape

orders_final = pd.read_excel(r"C:\Users\kereviz\PycharmProjects\Customer Demand "
                           r"Forecasting\data_sources\final_before_model.xlsx")

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

#RMSE: 1.6529 (Lasso)
#RMSE: 1.6743 (ElasticNet)
#RMSE: 0.8229 (CART)
#RMSE: 0.5999 (RF)
#RMSE: 0.6387 (GBM)


for name, regressor in models:
    r2 = cross_val_score(regressor, X_train, Y_train, cv=5, scoring="r2")
    print(f"r2: {r2.mean()} ({name}) ")

# r2 Scores
#r2: -0.7485149322484936 (Lasso)
#r2: -0.8129844275249669 (ElasticNet)
#r2: 0.6181876138949548 (CART)
#r2: 0.7945008739474475 (RF)
#r2: 0.7652897054643322 (GBM)



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

rf_params = {"max_depth": [3,4,5,6,8,10,12,15],
             "max_features": [3,5,10,20,30,40,50,"auto"],
             "min_samples_split": [2,3,5,7,10,13,15,20],
             "n_estimators": [10,20,30,40,50,60,70,80,90]}

rf_best_grid = GridSearchCV(rf_model, rf_params, cv=5, n_jobs=-1, verbose=True).fit(X_train, Y_train)

rf_best_grid.best_params_
#{'max_depth': 4, 'max_features': 50, 'min_samples_split': 3, 'n_estimators': 20}
rf_best_grid.best_score_
#0.8038101466446218

rf_final = rf_model.set_params(**rf_best_grid.best_params_, random_state=17).fit(X_train, Y_train)

# RMSE (Root Mean Square Error)
Y_pred = pd.DataFrame(rf_final.predict(X_val))
Y_pred.columns = ['GRANDTOTAL_NEW']
rmse = float(format(np.sqrt(mean_squared_error(Y_val, Y_pred)), '.3f'))
r2 = float(format(r2_score(Y_val, Y_pred), '.3f'))
print("RMSE: ", rmse)
print('r2: ', r2)

#RMSE:  0.617
#r2:  0.754

Y_val = Y_val.reset_index()
Y_val.drop("index",axis=1,inplace=True)
#ml.plot_co2(train,Y_val,Y_pred,"Pred Results")
ml.plot_importance(rf_final, X_train, len(X_val), save=False)

#RMSE:  1.201
#r2:  0.664

