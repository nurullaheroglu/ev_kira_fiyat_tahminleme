import numpy as np
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split, GridSearchCV


def generate_params(lgbm=True, lgbm_learning_rate=[0.01, 0.1, 0.05], lgbm_max_depth=[3, 8],
                    lgbm_n_estimators=[500, 1000], lgbm_subsample=[1, 0.5, 0.7]):
    if lgbm:
        lgbm_params = {"learning_rate": lgbm_learning_rate,
                       "max_depth": lgbm_max_depth,
                       "n_estimators": lgbm_n_estimators,
                       "subsample": lgbm_subsample}

    regressors = [("LightGBM", LGBMRegressor(random_state=17), lgbm_params)]

    return regressors


def hyperparameter_optimization(X, y, cv=3, lgbm=True, lgbm_learning_rate=[0.01, 0.1, 0.05], lgbm_max_depth=[3, 8],
                                lgbm_n_estimators=[500, 1000], lgbm_subsample=[1, 0.5, 0.7]):
    regressors = generate_params(lgbm, lgbm_learning_rate, lgbm_max_depth,
                                 lgbm_n_estimators, lgbm_subsample)

    print("Hyperparameter Optimization....")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)
    for name, regressor, params in regressors:
        print(f"########## {name} ##########")
        final_model_before = regressor.fit(X_train, y_train)
        y_pred = final_model_before.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        print(f"RMSE BEFORE: {round(rmse, 4)} ({name}) ")

        gs_best = GridSearchCV(regressor, params, cv=cv, n_jobs=-1, verbose=False).fit(X, y)
        final_model = regressor.set_params(**gs_best.best_params_).fit(X_train, y_train)

        y_pred = final_model.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))

        print(f"RMSE AFTER: {round(rmse, 4)} ({name}) ")
        print(f"{name} best params: {gs_best.best_params_}", end="\n\n")

    return final_model
