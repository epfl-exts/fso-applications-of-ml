# Import libraries

import pandas as pd
import numpy as np

pd.set_option("max_colwidth", 2000)

import warnings

warnings.filterwarnings("ignore")

import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Patch


sns.set_context("talk")

import os
import email
import re
import string
import nltk
import textwrap
from bs4 import BeautifulSoup
from collections import Counter
from random import randrange
from IPython.display import display

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV
from sklearn.compose import ColumnTransformer


####################################
### font size.                 ####
####################################

label_specs = {"fontsize": 15, "fontweight": "bold"}
title_specs = {"fontsize": 20, "fontweight": "bold"}


###########################################
###                                    ####
###    Chapter: Modelling              ####
###                                    ####
###########################################


###########################################
### Function: train_test_split_        ####
###########################################


def train_test_split_(df):
    """
    This function performs train/test splitting
    Note: The label column was named "spam_label" to avoid conflict with the word column
    "label" in the text features.
    """

    return train_test_split(
        df, test_size=0.3, stratify=df["spam_label"], random_state=0
    )


###########################################
### Function: fit_model   ####
###########################################


def fit_model(df_train, C=1):
    """
    This function fits a LogReg model with specified C
    It is meant for a fast refit on the best cv-results

    Note the label column was named "spam_label" to avoid conflict with the word column
    "label" in the text features.
    """

    # Train set: features
    X_train = df_train.drop(columns=["spam_label"]).values

    # Train set: Labels
    y_train = df_train["spam_label"].values

    # Define model
    model = LogisticRegression(
        solver="liblinear", class_weight="balanced", random_state=None, C=C
    )

    # Fit model
    model.fit(X_train, y_train)

    # feature_names = df_train.drop(columns=["spam_label"]).columns.tolist()

    return model  # , feature_names


###########################################
### Function: fit_log_reg_model        ####
###########################################


def fit_log_reg_model(df_train, scoring=None, scaler=True):
    """
    This function performs the following:
    - identify which num_features are part of the dataframe
    - fits a pipeline consisting of the following steps
        - apply standardscaler only to the num_features via CustomTransformer
        - use logistic regression as model
        - apply gridsearchcv to find the best parameters for the logistic regression model
        - store the cv_results in a dataframe
        - plot the training and validation curves based on the cv_results
    - return the fitted model
    """

    # Train set: features
    X_train = df_train.drop(columns=["spam_label"])
    # Train set: Labels
    y_train = df_train["spam_label"]

    ### COMMENT: For text data could work directly with `scipy.sparse.csr_matrix`-format
    # Currently used to stored TfIdF data
    
    # Create a pipeline with StandardScaler and LogisticRegression
    if scaler:
        pipeline = Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "log_reg",
                    LogisticRegression(
                        solver="liblinear", class_weight="balanced", random_state=24
                    ),
                ),
            ]
        )
    else: 
        pipeline = Pipeline(
            [
                (
                    "log_reg",
                    LogisticRegression(
                        solver="liblinear", class_weight="balanced", random_state=24
                    ),
                ),
            ]
        )

    # Define grid for the C parameter with 20 values
    C_values = np.logspace(-5, 5, 20)
    param_grid = {
        "log_reg__C": C_values,
    }

    # Create a grid search with cross-validation
    grid_search = GridSearchCV(
        pipeline,
        param_grid,
        cv=5,
        return_train_score=True,
        scoring=scoring,
        n_jobs=-1,
        verbose=1,
    )

    # Fit the grid search to the training data
    grid_search.fit(X_train, y_train)
    model = grid_search.best_estimator_

    # Get the cv_results as a DataFrame
    cv_results = pd.DataFrame(grid_search.cv_results_)
    cv_results = cv_results[[col for col in cv_results.columns if "split" not in col]]
    return model, cv_results


###########################################
### Function: viz_cv_results           ####
###########################################


def viz_cv_results(
    cv_results,
    param="param_log_reg__C",
    xlabel="C (Regularization Parameter)",
    show_table=False,
    plot_confidence=False,
    plot_fit_time=False,
    best_results=False,
):
    """
    This function visualizes the cv_results
    It displays the top 10 results in a table
    and plots the training and validation curves with the best parameter choice marked
    """

    cv_results_sorted = cv_results.sort_values(by="mean_test_score", ascending=False)

    if show_table:
        # Select the columns to display
        cv_results_table_cols = [
            "rank_test_score",
            param,
            "mean_test_score",
            "std_test_score",
            "mean_train_score",
            "std_train_score",
        ]

        display(cv_results_sorted[cv_results_table_cols].head(10))

    # mark best validation score
    best_row = cv_results.loc[cv_results["mean_test_score"].idxmax()]
    best_param = best_row[param]
    best_score = best_row["mean_test_score"]
    # extract row of cv_results with best_param
    if best_results:
        print("Best results")
        display(best_row)

    # Plot the training and validation curves
    plt.figure(figsize=(15, 10))
    plt.plot(
        cv_results[param],
        cv_results["mean_train_score"],
        label="Train Score",
        color="blue",
    )
    plt.plot(
        cv_results[param],
        cv_results["mean_test_score"],
        label="Validation Score",
        color="orange",
    )
    if plot_confidence:
        plt.fill_between(
            np.logspace(-5, 5, 20),
            cv_results["mean_train_score"] - cv_results["std_train_score"],
            cv_results["mean_train_score"] + cv_results["std_train_score"],
            color="blue",
            alpha=0.2,
        )
        plt.fill_between(
            np.logspace(-5, 5, 20),
            cv_results["mean_test_score"] - cv_results["std_test_score"],
            cv_results["mean_test_score"] + cv_results["std_test_score"],
            color="orange",
            alpha=0.2,
        )

    plt.scatter(
        best_param,
        best_score,
        color="red",
        marker="x",
        s=50,
        label="Best Validation Score",
    )
    plt.axvline(
        best_param,
        color="grey",
        linestyle="--",
        label=f"Best parameter: {best_param:.4f}",
    )
    # Add labels and title
    plt.xscale("log")
    plt.xlabel(xlabel, **label_specs)
    plt.ylabel("Score", **label_specs)
    plt.title("Training and Validation Curves", **title_specs)
    plt.legend()
    plt.grid()
    plt.show()

    ######
    if plot_fit_time:
        # Plot mean_fit_time and std_fit_time against param
        plt.figure(figsize=(15, 10))
        plt.plot(
            np.logspace(-5, 5, 20),
            cv_results["mean_fit_time"],
            label="Mean Fit Time",
            color="blue",
        )
        plt.fill_between(
            np.logspace(-5, 5, 20),
            cv_results["mean_fit_time"] - cv_results["std_fit_time"],
            cv_results["mean_fit_time"] + cv_results["std_fit_time"],
            label="Stabilitiy of Fit Time",
            color="orange",
            alpha=0.3,
        )
        # Add labels and title
        plt.xscale("log")
        plt.xlabel(xlabel, **label_specs)
        plt.ylabel("Fit Time", **label_specs)
        plt.title("Fit time vs. parameter", **title_specs)
        plt.legend()
        plt.grid()
        plt.show()


###########################################
### Function: plot_confusion_matrix   ####
###########################################


def plot_confusion_matrix(df_test, model):
    """
    This function plots the confusion matrix.
    We need to check whether "spam_level" is used as target
    or whether original df with "label" as target is being used
    """

    if "spam_label" in df_test.columns:
        # Test set: features
        X_test = df_test.drop(columns=["spam_label"])
        y_true = df_test["spam_label"]

        # Compute predictions on test set
        y_pred = model.predict(X_test)

    else:
        # Test set: features
        X_test = df_test["text_cleaned"].values

        # True labels
        y_true = df_test["spam_label"].values

        # Compute predictions on test set
        y_pred = model.predict(X_test)

    # Class labels
    classes = ["Non-spam", "Spam"]

    cmap = plt.cm.Blues
    title = None
    normalize = False

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    classes = classes
    if normalize:
        cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

    fig, ax = plt.subplots(figsize=(7, 7))
    im = ax.imshow(cm, interpolation="nearest", cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        # ... and label them with the respective list entries
        xticklabels=classes,
        yticklabels=classes,
        title="Confusion matrix",
        ylabel="True label",
        xlabel="Predicted label",
    )

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=0, ha="center", rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = ".2f" if normalize else "d"
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                format(cm[i, j], fmt),
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
            )
    fig.tight_layout()
    plt.show()
    # return ax


###########################################
### Function: classification_report_    ####
###########################################


def classification_report_(df_test, model):
    """
    This function computes the classification report
    """

    # Test set: features
    X_test = df_test.drop(columns=["spam_label"])
    y_true = df_test["spam_label"]

    # Compute predictions on test set
    y_pred = model.predict(X_test)

    # Print classification report
    print(classification_report(y_true, y_pred, target_names=["Non-spam", "Spam"]))


###########################################
### Function: visualize_coefficients   ####
###########################################

# Inspiration: "Introduction to Machine Learning with Python", A. Muller
# https://github.com/amueller/introduction_to_ml_with_python


def visualize_coefficients(model, df_train, n_top_features=25):

    # ensure all feature_names are strings and drop last column (spam_label)
    feature_names = [str(c) for c in df_train.columns[:-1]]
    coefficients = (model.named_steps["log_reg"].coef_).squeeze()

    """Visualize coefficients of a linear model.

    Parameters
    ----------
    coefficients : nd-array, shape (n_features,)
        Model coefficients.

    feature_names : list or nd-array of strings, shape (n_features,)
        Feature names for labeling the coefficients.

    n_top_features : int
        How many features to show. The function will show the largest (most
        positive) and smallest (most negative)  n_top_features coefficients,
        for a total of 2 * n_top_features coefficients.
    """

    # coefficients = coefficients.ravel()

    # Get coefficients with largest absolute values
    coef = coefficients.ravel()
    positive_coefficients = np.argsort(coef)[-n_top_features:]
    negative_coefficients = np.argsort(coef)[:n_top_features]
    interesting_coefficients = np.hstack([negative_coefficients, positive_coefficients])

    common_features = np.array(feature_names)[interesting_coefficients]

    counts_0 = [
        np.array(coefficients)[c] if c in positive_coefficients else 0
        for c in interesting_coefficients
    ]
    counts_1 = [
        np.array(coefficients)[c] if c in negative_coefficients else 0
        for c in interesting_coefficients
    ]

    # Store results in DataFrame and sort values
    df_coeffs = pd.DataFrame(
        list(zip(common_features, counts_0, counts_1)),
        columns=["Feature", "Non_spam", "Spam"],
    )

    # Plot top features
    # -----------------

    plt.figure(figsize=(8, 10))
    plt.barh(
        y=df_coeffs.Feature,
        width=df_coeffs.Spam,
        edgecolor="black",
        label="Non-spam",
        alpha=0.3,
    )
    plt.barh(
        y=df_coeffs.Feature,
        width=df_coeffs.Non_spam,
        edgecolor="black",
        label="Spam",
        alpha=0.3,
    )
    plt.xlabel("Coefficient magnitude", **label_specs)
    plt.title(
        "Top " + str(n_top_features) + " most important features in spam and non-spam",
        **title_specs,
    )
    plt.legend()  # (**label_specs)
    plt.xticks(**label_specs)
    plt.yticks(**label_specs)
    plt.show()


###################################################
### Function: plot_prediction_certainties      ####
###################################################


def plot_prediction_certainties(df_test, model, log_scale=True):
    """
    Input is df_test, model
    df_test: test data frame containing extract features or embeddings
    model: trained LogReg model, contains the coefficients for the features
    Output:
    - plot histogram of the predicted probabilities colored by y_test
    """

    df_test_extended = df_test.copy()
    # Setup test feature and test labels
    X_test = df_test_extended.drop(columns=["spam_label"])
    y_test = df_test_extended["spam_label"]
    # Get predictions
    y_pred = model.predict(X_test)
    df_test_extended["prediction"] = y_pred  # Uncommenting this line
    df_test_extended["misclf"] = y_test != y_pred
    # Compute probabilities for test set
    y_test_probs = model.predict_proba(X_test)
    # Add new columns for probabilities
    df_test_extended["proba_0"] = y_test_probs[:, 0]
    df_test_extended["proba_1"] = y_test_probs[:, 1]

    # Plot histogram of probabilities of samples predicted true and false
    # Use log_scale for count
    plt.figure(figsize=(10, 6))
    plt.hist(
        df_test_extended[df_test_extended["spam_label"] == 0]["proba_1"],
        bins=50,
        alpha=0.5,
        label="Non-spam",
        log=log_scale,
    )
    plt.hist(
        df_test_extended[df_test_extended["spam_label"] == 1]["proba_1"],
        bins=50,
        alpha=0.5,
        label="Spam",
        log=log_scale,
    )
    # Add a vertical line for the threshold of 0.5
    plt.axvline(x=0.5, color="red", linestyle="--", label="Threshold (0.5)")

    plt.xlabel("Predicted Probability of Spam", **label_specs)
    plt.ylabel("Count", **label_specs)
    plt.title("Predicted Probabilities of Spam", **title_specs)
    plt.legend()
    plt.show()

    # Define data frame of msiclf samples
    df_test_extended_misclf = df_test_extended[df_test_extended["misclf"] == 1]

    # Plot histogram of probabilities of samples with "misclf" is True
    plt.figure(figsize=(10, 6))
    plt.hist(
        df_test_extended_misclf[df_test_extended_misclf["spam_label"] == 0]["proba_1"],
        bins=50,
        alpha=0.5,
        label="Non-spam",
    )
    plt.hist(
        df_test_extended_misclf[df_test_extended_misclf["spam_label"] == 1]["proba_1"],
        bins=50,
        alpha=0.5,
        label="Spam",
    )
    # Add a vertical line for the threshold of 0.5
    plt.axvline(x=0.5, color="red", linestyle="--", label="Threshold (0.5)")

    plt.xlabel("Predicted Probability of Spam", **label_specs)
    plt.ylabel("Count", **label_specs)
    plt.title(
        "Predicted Probabilities of Spam \n for Misclassified Samples only",
        **title_specs,
    )
    plt.legend()
    plt.show()


###########################################
### Function: Print sample overview    ####
###########################################


def print_sample_overview(sample_extended):
    """
    Input:
    - sample_extended containing: text, text_cleaned, spam_label, prediction, proba_0, proba_1
    Output:
    - Overview
    """

    # Colab formating: wrap text with restriction to 2000 characters
    orig_text = "\n".join(textwrap.wrap(sample_extended["text"][:2000], 100))
    # Colab formating: wrap text with restriction to 2500 characters
    cleaned_text = "\n".join(textwrap.wrap(sample_extended["text_cleaned"][:2500], 100))
    print("Original Text\n=============")
    print(orig_text, "\n")
    print("Cleaned Text\n============")
    print(cleaned_text, "\n")
    print()
    print("Predictions\n============")
    print()
    # convert numerical target to class labels in spam_label and prediction
    sample_extended[["spam_label", "prediction"]] = sample_extended[
        ["spam_label", "prediction"]
    ].replace({0: "Non-spam", 1: "Spam"})

    print(f"Actual class:   {sample_extended['spam_label']}")
    print(f"Predicted class:   {sample_extended['prediction']}\n")
    print("Predicted probabilities\n========================")
    print(f"Non-spam: {sample_extended['proba_0']:.4f}")
    print(f"Spam: {sample_extended['proba_1']:.4f}\n")
    # print(f"Text: {sample_extended['text']}\n")
    # print(f"Cleaned Text: {sample_extended['text_cleaned']}\n")


###########################################
### Function: Plot top features        ####
###########################################


def plot_top_features(features, title, xlabel, ylabel, color_by_coeff_sign=False):
    """
    Plots the top features with their contributions.

    Args:
        features (pd.DataFrame): DataFrame containing feature names, contributions, and signs.
        title (str): Title of the plot.
        xlabel (str): Label for the x-axis.
        ylabel (str): Label for the y-axis.
        color_by_sign (bool): Whether to color bars based on the sign of the coefficients.
    """
    # Set colors for the bars
    colors = (
        features["sign"].map({1: "red", -1: "green"})
        if color_by_coeff_sign
        else "steelblue"
    )

    # Create the plot
    plt.figure(figsize=(6, 10))
    plt.barh(
        y=features["feature"],
        width=features["contribution"],
        color=colors,
        edgecolor="black",
        height=0.6,
    )
    plt.title(title, fontdict={"fontname": "Arial", "size": 14})
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    # Add legend if coloring by sign
    if color_by_coeff_sign:
        plt.legend(
            handles=[
                Patch(facecolor="green", label="positive"),
                Patch(facecolor="red", label="negative"),
            ],
            title="coefficient",
        )

    plt.show()


###########################################
### Function: error_analysis           ####
###########################################


def error_analysis(
    df_test, model, doc_nbr=1, n_top_coeff=20, color_by_coeff_sign=False
):
    """
    Input is df_test, model, doc_nbr, n_top_coeff
    df_test: test data frame containing extract features or embeddings
    model: trained LogReg model, contains the coefficients for the features
    doc_nbr: number to select a misclassified sample
    n_top_coeff: how many contributions should be selected
    Prints:
    - text and probabilities using print_sample_overview
    Plot:
    - top contributions using plot_top_features
    for a selected misclassified sample
    """

    # Load cleaned data with orignal "text" and with "text cleaned"
    df_cleaned = pd.read_csv("data/df_cleaned", index_col=0)
    df_cleaned_test = df_cleaned.loc[df_test.index]

    # Setup test feature and test labels
    X_test = df_test.drop(columns=["spam_label"])
    y_test = df_test["spam_label"]

    # Expand df_test with text, text cleaned, predictions, probabilities, misclassified flag
    df_test_extended = pd.concat(
        [df_cleaned_test.drop(columns=["spam_label"]), df_test], axis=1
    )
    # Add predictions
    y_pred = model.predict(X_test)
    df_test_extended["prediction"] = y_pred
    # Compute probabilities for test set
    y_test_probs = model.predict_proba(X_test)
    # Add new columns for probabilities
    df_test_extended["proba_0"] = y_test_probs[:, 0]
    df_test_extended["proba_1"] = y_test_probs[:, 1]
    df_test_extended["misclf"] = y_test != y_pred

    # extract coefficents from model
    coefficients = (model.named_steps["log_reg"].coef_).squeeze()
    print("Coefficients:", coefficients.shape[0])
    feature_names = [str(c) for c in df_test.columns[:-1]]
    feature_contribution = X_test * coefficients

    # display(X_test)
    # display(coefficients)
    # display(feature_contribution)

    # Extract sample
    df_test_extended_misclf = df_test_extended[df_test_extended["misclf"] == 1]
    df_test_extended_misclf.reset_index(drop=True)
    misclf_number = df_test_extended_misclf.shape[0]
    print("Number of misclassified samples:", misclf_number)

    # reduce doc_nbr by  to act as an index
    doc_nbr = doc_nbr - 1
    if misclf_number < doc_nbr:
        doc_nbr = misclf_number - 1
        print(
            "The parameter doc_nbr was too large so the last misclassified sample is being analysed."
        )
    # sample = X_test.iloc[doc_nbr, :]
    sample_extended = df_test_extended_misclf.iloc[doc_nbr, :]
    sample_contrib = feature_contribution.iloc[doc_nbr, :]
    sample_feat_val = X_test.iloc[doc_nbr, :]

    # Find top absolute coefficients
    # Store index of sorted coefficients
    # Store for top coefficients
    biggest_contributions_index = list(
        np.argsort(np.abs(sample_contrib))[-n_top_coeff:]
    )
    top_feature_names = [feature_names[i] for i in biggest_contributions_index]
    top_feat_val = sample_feat_val[biggest_contributions_index]
    top_contrib = sample_contrib[biggest_contributions_index]
    top_coeffs = coefficients[biggest_contributions_index]
    # Store sign for plotting
    top_coeff_signs = [
        1 if s >= 0 else -1 for s in coefficients[biggest_contributions_index]
    ]
    top_features = (
        pd.DataFrame(
            zip(
                top_feature_names,
                top_feat_val,
                top_contrib,
                top_coeff_signs,
            ),
            columns=["feature", "feature_values", "contribution", "sign"],
        )
        # .set_index("feature")
    )

    ## Output: Display text information ##
    print()
    print(f"Overview for document index: {doc_nbr + 1}\n")
    print_sample_overview(sample_extended)

    # Plotting top features
    plot_top_features(
        features=top_features,
        title=f"Top {n_top_coeff} contributions for the misclassified sample",
        xlabel="Feature Contribution",
        ylabel="Top Features",
        color_by_coeff_sign=color_by_coeff_sign,
    )


###########################################
### Function: simple_func              ####
###########################################


def simple_func(x):
    if x > 0:
        return 1
    elif x == 0:
        return 0


###########################################
### Function: run_pca              ####
###########################################


def run_pca(df, N="all", with_labels=False):
    """
    This function
    - performs PCA on the data frame
    - plots the explained variance ratio as barplot and the cumulative explained variance as a step plot in the same figure
    - extracts the number of components need for each multiple of 10% explained variance
    - creates a scatterplot of the first two components
    Output:
    - returns the transformed data frame
    """

    # Remove labels from the data frame
    X = df.drop(columns=["spam_label"])

    spam_labels = df["spam_label"] == 1

    # Standardize the data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Perform PCA
    pca = PCA()
    X_pca = pca.fit_transform(X_scaled)

    # Explained variance ratio
    explained_variance = pca.explained_variance_ratio_

    # Plot explained variance ratio and cumulative explained variance
    plt.figure(figsize=(10, 6))
    plt.bar(range(1, len(explained_variance) + 1), explained_variance, alpha=0.5)
    plt.step(
        range(1, len(explained_variance) + 1),
        np.cumsum(explained_variance),
        where="mid",
        label="Cumulative explained variance",
        color="red",
    )
    plt.xlabel("Number of components")
    plt.ylabel("Explained variance ratio")
    plt.title("PCA: Explained Variance Ratio and Cumulative Explained Variance")
    plt.legend()
    plt.grid()
    plt.show()

    # Extract number of components for each multiple of 10% explained variance
    if N == "all":
        N = np.arange(0.1, 1.1, 0.1)
        for n in N:
            num_components = np.argmax(np.cumsum(explained_variance) >= n) + 1
            print(
                f"Number of components for {n:.0%} explained variance: {num_components}"
            )

    # Scatterplot of the first two components
    if with_labels:
        plt.figure(figsize=(8, 8))
        plt.scatter(
            x=X_pca[spam_labels, 0],
            y=X_pca[spam_labels, 1],
            alpha=0.5,
            s=10,
            label="Spam",
            color="red",
        )
        plt.scatter(
            x=X_pca[~spam_labels, 0],
            y=X_pca[~spam_labels, 1],
            alpha=0.5,
            s=10,
            label="Non-spam",
            color="lightgreen",
        )
        plt.xlabel("Principal Component 1")
        plt.ylabel("Principal Component 2")
        plt.legend()
        plt.title("PCA: Scatterplot of the First Two Components")
        plt.show()
    else:
        plt.figure(figsize=(8, 8))
        plt.scatter(x=X_pca[:, 0], y=X_pca[:, 1], alpha=0.5, s=10, label="all data")
        plt.xlabel("Principal Component 1")
        plt.ylabel("Principal Component 2")
        plt.legend()
        plt.title("PCA: Scatterplot of the First Two Components")
        plt.show()

    return pd.DataFrame(X_pca)


###########################################
