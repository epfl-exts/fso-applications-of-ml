# Import libraries

import pandas as pd
import numpy as np

pd.set_option("max_colwidth", 2000)

import warnings

warnings.filterwarnings("ignore")

import matplotlib.pyplot as plt
import seaborn as sns

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


####################################
### font size.                 ####
####################################

label_specs = {"fontsize": 9}
title_specs = {"fontsize": 12, "fontweight": "bold"}


###########################################
###                                    ####
###    Chapter: EDA                    ####
###                                    ####
###########################################


###################################################
### Function:   plot_class_frequency           ####
###################################################


def plot_class_frequency(df):
    """
    This function plots the number of samples per class in the data.
    """

    class_counts = np.round(pd.value_counts(df["spam_label"], normalize=True), 3)

    ################
    # Spam dataset #
    ################

    class_counts.index = ["non-spam", "spam"]

    print("Samples per class (%):")
    print(class_counts * 100)
    print("\n")

    sns.barplot(x=class_counts.index, 
                y=pd.value_counts(df["spam_label"]),
                hue=class_counts.index, 
                legend=False,
               )
    plt.ylabel("Counts")
    plt.title("Sample frequency per class")


#############################################
### Function:   plot_numeric_features    ####
#############################################


def plot_numeric_features(df, with_labels=False):
    """
    This function uses get_numeric_features() to extract:
    - counts characters, words, unique words, punctuation marks, uppercase & lowercase words,
      digits and alphabetic chars
    - removes 'URL:' and 'mailto:' strings from text
    - counts the number of HTML tags, e-mail addresses, URLs and twitter usernames.

    Outputs:
    - plots the distribution of features per class if with_labels=True (default).
    - no return value.
    """

    num_features = [
        "email_counts",
        "html tag_counts",
        "url_counts",
        "Twitter username_counts",
        "hashtag_counts",
        "character_counts",
        "word_counts",
        "unique word_counts",
        "punctuation mark_counts",
        "uppercase word_counts",
        "lowercase word_counts",
        "digit_counts",
        "alphabetic char_counts",
        "spam_label",
    ]

    # Get numeric features
    num_features_df = df.loc[:, num_features].copy()

    # Plot results
    plot_cols = num_features[:-1]

    fig, axes = plt.subplots(nrows=5, ncols=3, figsize=(16, 4 * 5))

    for col, ax in zip(plot_cols, axes.ravel()):
        if with_labels:
            ax.hist(
                np.log1p(num_features_df[num_features_df["spam_label"] == 0][col]),
                bins=20,
                density=True,
                label="Non-spam",
                alpha=0.3,
                edgecolor="grey",
            )
            ax.hist(
                np.log1p(num_features_df[num_features_df["spam_label"] == 1][col]),
                bins=20,
                density=True,
                label="Spam",
                alpha=0.3,
                edgecolor="grey",
            )
        else:
            ax.hist(
                np.log1p(num_features_df[col]),
                bins=20,
                density=True,
                label="All",
                # alpha=0.3,
                edgecolor="grey",
            )

        ax.legend(fontsize=12)
        ax.set_ylabel("Normalized Frequency", fontsize=14)
        ax.set_xlabel("Number of " + col.lower()[:-7] + "s (log scale)", fontsize=14)

    axes[4, 1].axis("off")
    axes[4, 2].axis("off")
    plt.tight_layout()
    plt.show()


####################################
### Function:   Convert         ####
####################################


def Convert(tup, di):
    """
    This function converts tuples
    into dictionaries.
    """
    di = dict(tup)
    return di


###########################################
### Function: plot_most_common_words   ####
###########################################


def plot_most_common_words(df, N, per_1000=False, with_labels=True):
    """
    This function computes the N most common words in
    hams and spams and plots the results in a common
    histogram.
    """

    if with_labels == True:
        # Join documents in single strings for each corpus
        corpus_0 = " ".join(df[df["spam_label"] == 0]["text_cleaned"])
        corpus_1 = " ".join(df[df["spam_label"] == 1]["text_cleaned"])
        tokenizer = nltk.WordPunctTokenizer()

        tokens_0 = tokenizer.tokenize(corpus_0)
        tokens_1 = tokenizer.tokenize(corpus_1)

        freq_0 = Counter(tokens_0)
        freq_1 = Counter(tokens_1)

        # Get top N words from each class
        top_words_0 = [word for word, _ in freq_0.most_common(N)]
        top_words_1 = [word for word, _ in freq_1.most_common(N)]
        print(f"Top {N} words in class 0:")
        print(top_words_0)
        print(f"Top {N} words in class 1:")
        print(top_words_1)

        # Combine top words from both classes
        common_list = list(set(top_words_0 + top_words_1))

        # Get frequencies for all common words in both classes
        counts_0 = [freq_0.get(word, 0) for word in common_list]
        counts_1 = [freq_1.get(word, 0) for word in common_list]

        if per_1000:
            # Normalize counts to per 1000 emails
            counts_0 = 1000 * (counts_0 / (df["spam_label"] == 0).sum())
            counts_1 = 1000 * (counts_1 / (df["spam_label"] == 1).sum())

        df_plot = pd.DataFrame(
            list(zip(common_list, counts_0, counts_1)),
            columns=["Word", "ham_counts", "spam_counts"],
        ).sort_values(by="ham_counts", ascending=True)

        # Plot
        plt.figure(figsize=(8, 8))
        plt.barh(
            y=df_plot.Word,
            width=df_plot.ham_counts,
            edgecolor="black",
            label="Non-spam",
            alpha=0.3,
        )
        plt.barh(
            y=df_plot.Word,
            width=df_plot.spam_counts,
            edgecolor="black",
            label="Spam",
            alpha=0.3,
        )
        if per_1000:
            plt.xlabel("Word counts per 1000 emails", **label_specs)
        else:
            plt.xlabel("Word counts across corpus", **label_specs)
        plt.title(
            f"{N} most frequent words in spam and non-spam combined", **title_specs
        )
        plt.legend(**label_specs)
        plt.xticks(**label_specs)
        plt.yticks(**label_specs)
    else:
        # Join documents in single strings
        corpus = " ".join(df["text_cleaned"])
        tokenizer = nltk.WordPunctTokenizer()

        tokens = tokenizer.tokenize(corpus)
        freq = Counter(tokens)

        # Get top N words from each class
        top_words = [word for word, _ in freq.most_common(N)]
        print(f"Top {N} words in all classes:")
        print(top_words)

        # Get the counts of the top words
        counts = [freq.get(word, 0) for word in top_words]
        # Normalize counts to per 1000 emails
        if per_1000:
            counts = 1000 * (counts / df.shape[0])
        # Create a DataFrame for plotting
        df_plot = pd.DataFrame(
            list(zip(top_words, counts)), columns=["Word", "Counts"]
        ).sort_values(by="Counts", ascending=True)
        # Plot
        plt.figure(figsize=(8, 8))
        plt.barh(
            y=df_plot.Word,
            width=df_plot.Counts,
            edgecolor="black",
            label="All emails",
            alpha=0.3,
        )
        if per_1000:
            plt.xlabel("Word counts per 1000 emails", **label_specs)
        else:
            plt.xlabel("Word counts across corpus", **label_specs)
            plt.legend(**label_specs)
            plt.xticks(**label_specs)
            plt.yticks(**label_specs)
        plt.title(f"{N} most frequent words", **title_specs)


###########################################
### Function: plot_most_common_tokens   ####
###########################################


def plot_most_common_tokens(df, N, per_1000=False):
    """
    This function computes the N most common words in the text_cleaned column.
    It computes them by class and plots the results in a common histogram.
    For better comparison, the counts can normalized to per 1000 emails.
    """

    df = df.copy()

    tokenizer = nltk.WordPunctTokenizer()

    corpus_0 = " ".join(df[df["spam_label"] == 0]["text_cleaned"])
    corpus_1 = " ".join(df[df["spam_label"] == 1]["text_cleaned"])

    tokens_0 = tokenizer.tokenize(corpus_0)
    tokens_1 = tokenizer.tokenize(corpus_1)

    freq_0 = Counter(tokens_0)
    freq_1 = Counter(tokens_1)

    # Get top N words from each class
    top_words_0 = [word for word, _ in freq_0.most_common(N)]
    top_words_1 = [word for word, _ in freq_1.most_common(N)]
    print(f"Top {N} words in class 0:")
    print(top_words_0)
    print(f"Top {N} words in class 1:")
    print(top_words_1)

    # Combine top words from both classes
    common_list = list(set(top_words_0 + top_words_1))

    # Get frequencies for all common words in both classes
    counts_0 = [freq_0.get(word, 0) for word in common_list]
    counts_1 = [freq_1.get(word, 0) for word in common_list]

    if per_1000:
        # Normalize counts to per 1000 emails
        counts_0 = 1000 * (counts_0 / (df["spam_label"] == 0).sum())
        counts_1 = 1000 * (counts_1 / (df["spam_label"] == 1).sum())

    df_plot = pd.DataFrame(
        list(zip(common_list, counts_0, counts_1)),
        columns=["Word", "ham_counts", "spam_counts"],
    ).sort_values(by="ham_counts", ascending=False)

    # Plot
    plt.figure(figsize=(8, 8))
    plt.barh(
        y=df_plot.Word,
        width=df_plot.ham_counts,
        edgecolor="black",
        label="Non-spam",
        alpha=0.3,
    )
    plt.barh(
        y=df_plot.Word,
        width=df_plot.spam_counts,
        edgecolor="black",
        label="Spam",
        alpha=0.3,
    )
    if per_1000:
        plt.xlabel("Word counts per 1000 emails", **label_specs)
    else:
        plt.xlabel("Word counts", **label_specs)
    plt.title(f"{N} most frequent words in spam and non-spam combined", **title_specs)
    plt.legend(**label_specs)
    plt.xticks(**label_specs)
    plt.yticks(**label_specs)


###########################################
### Function: corpus_vocabulary        ####
###########################################


def corpus_vocabulary(df):

    # Join documents in single strings
    corpus = " ".join([text for text in df["text_cleaned"]])
    corpus_0 = " ".join([text for text in df[df["spam_label"] == 0]["text_cleaned"]])
    corpus_1 = " ".join([text for text in df[df["spam_label"] == 1]["text_cleaned"]])

    # nltk.Text() expects tokenized text
    # Create an instance of WordPunctTokenizer
    tokenizer = nltk.WordPunctTokenizer()

    corpusText = nltk.Text(tokenizer.tokenize(corpus))
    corpusText_0 = nltk.Text(tokenizer.tokenize(corpus_0))
    corpusText_1 = nltk.Text(tokenizer.tokenize(corpus_1))

    print("Vocabulary size")
    print("---------------")
    print()

    print("Non-spam mails : {} unique words ".format(len(set(corpusText_0))))
    print("Spam mails     : {} unique words ".format(len(set(corpusText_1))))
    print("All mails      : {} unique words ".format(len(set(corpusText))))


###########################################
### Function: show_bag_of_words_vector ####
###########################################


def show_bag_of_words_vector():
    """
    This functio extracts BoW features for a toy corpus.
    """

    # Toy corpus
    corpus = [
        "I enjoy paragliding.",
        "I do like NLP.",
        "I like deep learning.",
        "O Captain! my Captain!",
    ]

    vectorizer = CountVectorizer(
        analyzer="word", ngram_range=(1, 1), token_pattern="(?u)\\b\\w+\\b", min_df=1
    )

    # Transform corpus
    corpus_bow = vectorizer.fit_transform(corpus)

    # Get the vocabulary
    vocab = vectorizer.get_feature_names_out()

    corpus_df = pd.DataFrame(
        corpus_bow.toarray(), columns=vectorizer.get_feature_names_out()
    )
    corpus_df["Text"] = corpus
    corpus_df.set_index("Text", inplace=True)
    return corpus_df


###########################################
### Function: show_tfidf_vector ####
###########################################


def show_tfidf_vector():
    """
    This function extracts TF-IDF features for a toy corpus.
    """

    # Toy corpus
    corpus = [
        "I enjoy paragliding.",
        "I do like NLP.",
        "I like deep learning.",
        "O Captain! my Captain!",
    ]

    vectorizer = TfidfVectorizer(
        analyzer="word", ngram_range=(1, 1), token_pattern="(?u)\\b\\w+\\b", min_df=1
    )

    # Transform corpus
    corpus_tfidf = vectorizer.fit_transform(corpus)

    # Get the vocabulary
    vocab = vectorizer.get_feature_names_out()

    corpus_df = pd.DataFrame(
        corpus_tfidf.toarray(), columns=vectorizer.get_feature_names_out()
    )
    corpus_df["Text"] = corpus
    corpus_df.set_index("Text", inplace=True)
    return corpus_df
