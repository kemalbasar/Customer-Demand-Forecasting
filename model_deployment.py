import pandas as pd
import lightgbm as lgb
from lightgbm import LGBMRegressor
from sklearn.model_selection import GridSearchCV
from tseries_tools import lgbm_smape

orders_final = pd.read_csv(r"C:\Users\kereviz\PycharmProjects\Customer Demand "
                           r"Forecasting\data_sources\final_before_model.csv")
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

lgbm_model = LGBMRegressor()

lgb_params = {'num_leaves': [2, 10],
              'learning_rate': [0.01, 0.50],
              'feature_fraction': [0.4, 0.12],
              'max_depth': [1, 5],
              'num_boost_round': [10000, 40000],
              'nthread': [-1]}

lgbm_gs_best_grid = GridSearchCV(lgbm_model,
                                 lgb_params,
                                 cv=3,
                                 scoring='r2',
                                 n_jobs=-1,
                                 verbose=True).fit(X_train, Y_train, eval_set = (X_val,Y_val))

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
