# Import libraries

import pandas as pd
import numpy as np
pd.set_option('max_colwidth', 2000)

import warnings
warnings.filterwarnings("ignore")

import matplotlib.pyplot as plt
import seaborn as sns
from pandas.plotting import lag_plot
import statsmodels.api as sm
sns.set_context('talk')

import os
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_absolute_error as MAE
from sklearn import set_config




####################################
### Function: load_data         ####
####################################

def load_data():
    df = pd.read_csv(os.path.join('data','energy_data.csv'), index_col=0, parse_dates=True)
    df.rename(columns={"load": "electric load", "temp": "temperature"},inplace=True)
    print('Data loaded.')
    print('Data size:', df.shape)
    return df

####################################
### Function: info_data         ####
####################################

def info_data(df):
    print('Frequency of observations: ' ,pd.infer_freq(df.index))
    print('---------------------------------------')
    print('Number of days: ' ,df.index.dayofyear.unique().max())
    print('Number of month: ' ,df.index.month.unique().max())
    print('---------------------------------------')
    print('First month: ' ,df.index.month_name().unique()[0])
    print('Last month: ' ,df.index.month_name().unique()[-1])
    print('---------------------------------------')
    print('Available variables: ' ,list(df.columns))
    print('variable types:',list(df.dtypes.values))
    print('---------------------------------------')
    print('Number of missing hours: ' ,list(df.isna().sum()))
    
    
    
    
####################################
### Function: plot_data         ####
####################################

def plot_data():
    


    # load data and rename columns
    df = pd.read_csv(os.path.join('data','energy_data.csv'), index_col=0, parse_dates=True)
    df.rename(columns={"load": "electric load", "temp": "temperature"},inplace=True)
    ###################################
    # plotting distributions
    fig, ax = plt.subplots(1, 2, figsize=(15, 4), gridspec_kw={"wspace": 0.2})
    
    ax[0].hist(df["electric load"],bins=30,color='skyblue')
    ax[0].set_title("distribution of electric load",fontsize=14)
    ax[0].tick_params(labelsize=12)
    
    ax[1].hist(df["temperature"],bins=30,color='skyblue')
    ax[1].set_title("distribution of temperature",fontsize=14)
    ax[1].tick_params(labelsize=12)

    plt.show()
    ####################################    
    # plotting trend and seasonality
    fig, ax = plt.subplots(2, 1, figsize=(15, 7), gridspec_kw={"hspace": 0.6})

    ax[0].plot(df["electric load"],color='skyblue')
    ax[0].set_title("Hourly electric load",fontsize=14)
    ax[0].set_ylabel("electric load (MW)",fontsize=14)
    ax[0].tick_params(labelsize=12)
    
    ax[1].plot(df["temperature"],color='skyblue')
    ax[1].set_title("Hourly temperature",fontsize=14)
    ax[1].set_ylabel("temperature",fontsize=14)
    ax[1].tick_params(labelsize=12)

    plt.show()
    ####################################    
    # create time features
    df["month"] = df.index.month - 1
    df["day"] = df.index.day_of_week
    df["hour"] = df.index.hour
    
    # plot aggregate trends    
    fig, ax = plt.subplots(
    nrows=1, ncols=3, figsize=(15, 4), sharey=True, gridspec_kw={"wspace": 0.1}
    )

    ax[0].plot(df.groupby("month").mean()["electric load"],color='skyblue')
    ax[1].plot(df.groupby("day").mean()["electric load"],color='skyblue')
    ax[2].plot(df.groupby("hour").mean()["electric load"],color='skyblue')

    ax[0].set_title("average loads over the months",fontsize=14)
    ax[1].set_title("average loads over the week",fontsize=14)
    ax[2].set_title("average loads during the day",fontsize=14)

    ax[0].set_xlabel("January to December",fontsize=14)
    ax[1].set_xlabel("Monday to Sunday",fontsize=14)
    ax[2].set_xlabel("0 to 23 hours",fontsize=14)
    
    ax[0].tick_params(labelsize=12)
    ax[1].tick_params(labelsize=12)
    ax[2].tick_params(labelsize=12)

    ax[0].set_ylabel("electric loads (MW)",fontsize=14)
    plt.show()
    
    ####################################    
    # heatmap
    df1 = df.groupby(["month", "hour"], sort=False).agg(["mean"])
    df1.columns = df1.columns.droplevel(1)
    df1.reset_index(inplace=True)

    fig, axes = plt.subplots(figsize=(15, 5))

    heatmap = pd.pivot_table(df1, values="electric load", index="month", columns="hour")
    heatmap.index = df.index.month_name().unique().tolist()
    
    sns.heatmap(heatmap, cmap="YlGnBu")
    axes.set_title("electric load per month and hour",fontsize=14)
    axes.tick_params(labelsize=11)
    axes.set_xlabel("hours",fontsize=14)
    plt.show()
    
    
####################################
### Function: plot_corr         ####
####################################

def plot_corr(n_lags):    

    # load data and rename columns
    df = pd.read_csv(os.path.join('data','energy_data.csv'), index_col=0, parse_dates=True)
    df.rename(columns={"load": "electric load", "temp": "temperature"},inplace=True)
    
    ###################################

    # create 1 hour lags
    df["load_lag"] = df["electric load"].shift(n_lags)
    df["temp_lag"] = df["temperature"].shift(n_lags)

    #  dropping the rows with na
    df.dropna(how="any", inplace=True)

    # we don't need the temp anymore
    df.drop(["temperature"], axis=1, inplace=True)
    ####################################   
    # correlation analysis
    fig, axes = plt.subplots(figsize=(12, 3.5))
    
   #  plot autocorrelations
    sm.graphics.tsa.plot_acf(df["electric load"], lags=24, alpha=0.05, zero=False, ax=axes)

    axes.set_xlabel("1 to 24 hours lags",fontsize=14)
    axes.set_ylabel("autocorrelations",fontsize=14)
    axes.set_title("electric loads autocorrelations, shaded area shows 95% confidence bands",fontsize=14)

    # red marker for the n_lags
    axes.scatter(n_lags, df["electric load"].squeeze().autocorr(n_lags), marker="x", c="red", s=50 * 2 ** 2)
    ####################################    
   #  scatter plots
    fig, axes = plt.subplots(
        nrows=1,
        ncols=2,
        figsize=(12, 3.5),
        sharey=True,
        gridspec_kw={"wspace": 0.10, "width_ratios": [1, 1]},
    )

    # scatter plots of the values with their first lags
    lag_plot(df["electric load"], lag=n_lags, ax=axes[0], alpha=0.02)
    axes[1].scatter(df["temp_lag"], df["electric load"], alpha=0.02)

    axes[0].set_xlabel(r'electric load {} hours ago'.format(n_lags),fontsize=14)
    axes[0].set_ylabel("current load",fontsize=14)
    axes[0].set_title(r"auto-correlation: {:.3f}".format(df["electric load"].squeeze().autocorr(n_lags)),fontsize=14)

    axes[1].set_xlabel(r'temperature {} hours ago'.format(n_lags),fontsize=14)
    axes[1].set_title(r"correlation: {:.3f}".format(df["electric load"].corr(df["temp_lag"])),fontsize=14)

    axes[0].tick_params(labelsize=12)
    axes[1].tick_params(labelsize=12)
 
    plt.show()
    
    
###########################################
### Function: sample_split             ####
###########################################

def sample_split():
    
    # load data and rename columns
    df = pd.read_csv(os.path.join('data','energy_data.csv'), index_col=0, parse_dates=True)
    df.rename(columns={"load": "electric load", "temp": "temperature"},inplace=True)
   
    # create time features
    df["month"] = df.index.month - 1
    df["day"] = df.index.day_of_week
    df["hour"] = df.index.hour
    
    # create 1 hour lags
    df["load_lag"] = df["electric load"].shift(1)
    df["temp_lag"] = df["temperature"].shift(1)
    
    #  dropping the rows with na
    df.dropna(how="any", inplace=True)

    df.drop(["temperature"], axis=1, inplace=True)
    
    # Split data into train/test sets
    train = df.loc["2014-01-01":"2014-11-30"]  # 11 months
    test = df.loc["2014-12-01":"2014-12-31"]  # 1 month

    print("Train data from 2014-Jan to 2014-Nov", train.shape)
    print("Test data 2014-Dec", test.shape)
    
    
    # Plot training and test sets
    fig, ax = plt.subplots(figsize=(12, 3.5))
    ax.plot(train['electric load'], label="Train set", color="skyblue")
    ax.plot(test['electric load'], label="Test set", color="lightsalmon")

    ax.set_title("train and test sets",fontsize=14)
    ax.set_ylabel("electric loads (MW)",fontsize=14)
    ax.tick_params(labelsize=12)
    ax.legend(fontsize=14)
    plt.show()
    
    return train, test

###########################################
### Function: train_model                ####
###########################################

def train_model(train, select_model):

    # specifying features and the target for train/test sets
    X_tr = train.drop(["electric load"], axis=1)
    y_tr = train["electric load"]

    # One-hot encoding for categorical columns
    cat_transformer = Pipeline([("onehot", OneHotEncoder(handle_unknown="ignore"))])

    # Create cross-validation object
    n_split = 5
    cv = TimeSeriesSplit(n_splits=n_split, test_size=24 * 30 * 1)  #  1 month of validation set

    
    if select_model=='regression':

        # Applying the transformers
        preprocessor = ColumnTransformer(
            [("cat", cat_transformer, ["month", "day", "hour"])], remainder=StandardScaler()
        )
       
        # Create the pipeline
        pipe_ridge = Pipeline([("preprocessor", preprocessor), ("ridge", Ridge())])

        print('Preprocessing data is done!')
        print('Cross-validation is setup!')

        # Create grid for alpha
        grid = {"ridge__alpha": np.logspace(-4, 4, num=20)}

        # Create the grid search object
        fit_model = GridSearchCV(
            pipe_ridge,
            grid,
            cv=cv,
            return_train_score=True,
            scoring="neg_mean_absolute_error",
            verbose=0)

        print('Grid search is setup!')
        
        # Fit on the training set
        fit_model.fit(X_tr, y_tr)
        
        print('Training with grid search and cross-validation is done!')
        print('A total of {} models are fitted'.format(n_split * len(grid['ridge__alpha'])))  
        
        print('\n')
        print('Here is the results of grid search:')
        print('\n')

        # Collect results in a DataFrame
        cv_results = pd.DataFrame(fit_model.cv_results_)

        # Plot train and validation curves

        fig, ax = plt.subplots(figsize=(12, 3.5))
        ax.semilogx(
            cv_results["param_ridge__alpha"],
            -cv_results["mean_train_score"],
            label="training error",
        )
        ax.semilogx(
            cv_results["param_ridge__alpha"],
            -cv_results["mean_test_score"],
            label="validation error",
        )

        # Add marker for best score
        ax.scatter(
            fit_model.best_params_.values(),
            -1 * fit_model.best_score_,
            marker="x",
            c="red",
            zorder=10,
        )
        ax.set_xlabel("$alpha$",fontsize=14)
        ax.set_ylabel("MAE",fontsize=14)
        ax.set_title("Evolution of training/validation errors with $alpha$",fontsize=14)

        ax.legend(fontsize=14)
        plt.show()
        
    else:
        print('------This can take a few minutes------')
        # Applying the transformers without scaling
        preprocessor = ColumnTransformer(
            [("cat", cat_transformer, ["month", "day", "hour"])], remainder="passthrough"
        )

        # Create pipeline
        pipe_rf = Pipeline(
            [
                ("preprocessor", preprocessor),
                ("rf", RandomForestRegressor(min_samples_split=20)),
            ]
        )
        print('Preprocessing data is done!')
        print('Cross-validation is setup!')

        # Create grid for alpha
        grid = {"rf__n_estimators": np.arange(1, 1001, 100)}

        # Create the grid search object
        fit_model = GridSearchCV(
            pipe_rf,
            grid,
            cv=cv,
            return_train_score=True,
            scoring="neg_mean_absolute_error",
            verbose=0)
        
        print('Grid search is setup!')
        
        # Fit on the training set
        fit_model.fit(X_tr, y_tr)
        
        print('Training with grid search and cross-validation is done!')
        print('A total of {} models are fitted'.format(n_split * len(grid['rf__n_estimators'])))  

        print('Here is the results of grid search:')
        
        # Collect results in a DataFrame
        cv_results = pd.DataFrame(fit_model.cv_results_)

        # Plot test curve
        fig, ax = plt.subplots(figsize=(12, 3.5))
        ax.plot(
            cv_results["param_rf__n_estimators"],
            -cv_results["mean_train_score"],
            label="training error",
        )
        ax.plot(
            cv_results["param_rf__n_estimators"],
            -cv_results["mean_test_score"],
            label="validation error",
        )

        ax.scatter(
            fit_model.best_params_.values(),
            -1 * fit_model.best_score_,
            marker="x",
            c="red",
            zorder=10,
        )
        ax.set_xlabel("number of trees",fontsize=14)
        ax.set_ylabel("MAE",fontsize=14)
        ax.set_title("Evolution of training/validation errors with number of trees",fontsize=14)
        ax.legend(fontsize=14)
        plt.show()

    return fit_model

###########################################
### Function: evaluate_model           ####
###########################################

def evaluate_model(model, train, test, n_days):

    # specifying features and the target for train/test sets
    X_tr = train.drop(["electric load"], axis=1)
    y_tr = train["electric load"]
    X_te = test.drop(["electric load"], axis=1)
    y_te = test["electric load"]
 
    # baseline
    baseline = MAE(y_te, np.median(y_tr) * np.ones(len(y_te)))
    print("baseline MAE: {:.2f}".format(baseline))

    baseline_lag = MAE(y_te, X_te["load_lag"])
    print("smart baseline MAE: {:.2f}".format(baseline_lag))

    if list(model.get_params().keys())[6]=='estimator__ridge':

        # Evaluate on the test set
        y_pred = model.predict(X_te)

        model_mae = MAE(y_te, y_pred)
        print("linear regression MAE: {:.2f}".format(model_mae))
        
    else:
        
        # Evaluate on the test set
        y_pred = model.predict(X_te)

        model_mae = MAE(y_te, y_pred)
        print("random forest regression MAE: {:.2f}".format(model_mae))
    
    print('\n')
    
    # plot the prediction
    fig, axes = plt.subplots(
        2, 1, figsize=(12, 7), gridspec_kw={"hspace": 0.75}
    )

    axes[0].barh(np.arange(3), [baseline, baseline_lag, model_mae],color='skyblue', fill=False)
    axes[0].set_yticks(np.arange(3))
    axes[0].set_yticklabels(("baseline", "smart baseline", "model"))
#    axes[0].set_xlabel("MAE")
    axes[0].set_title("MAE")    
    axes[0].spines['top'].set_visible(False)
    axes[0].spines['right'].set_visible(False)
    axes[0].spines['bottom'].set_visible(False)

    axes[1].plot(y_te.values[-n_days * 24 :], lw=2, label="actual loads")
    axes[1].plot(y_pred[-n_days * 24 :], ls="-.", lw=2, label="model predicted loads")
    axes[1].plot(X_te["load_lag"].values[-n_days * 24 :],ls="--", lw=1.5,label="loads from the previous hour")

    axes[1].set_title('Actual vs predicted loads during the last {} days of the test set'.format(n_days))
    axes[1].set_xlabel('hours')
    axes[1].set_ylabel("loads")
    axes[1].legend(bbox_to_anchor=(1,1),frameon=False)

    plt.show()    
    


###########################################
### Function: xx_model           ####
###########################################


