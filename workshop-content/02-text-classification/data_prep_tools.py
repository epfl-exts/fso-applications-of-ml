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
import scipy.sparse as sp


####################################
### font size.                 ####
####################################

label_specs = {"fontsize": 9}
title_specs = {"fontsize": 12, "fontweight": "bold"}


###########################################
###                                    ####
###    Chapter: Data Preparation       ####
###                                    ####
###########################################


####################################
### Function: get_email_content ####
####################################


def get_email_content(email_path):
    """
    This function uses the email library to extract text from mails
    Used in the load_source_data function to extract text from emails
    """

    file = open(email_path, encoding="ISO-8859-1")
    try:
        msg = email.message_from_file(file)
        for part in msg.walk():
            if (part.get_content_type() == "text/plain") | (
                part.get_content_type() == "text/html"
            ):
                return part.get_payload()
    except Exception as e:
        print(e)


####################################
### Function: remove_excesspace ####
####################################


def remove_excesspace(doc):
    """
    This function replaces multiple whitespace, new lines and tab characters
    by single whitespace and strips leading and trailing whitespace from strings.
    It is used in the load_source_data function to clean the text.
    """

    doc = doc.str.replace(r"[\s]+", " ", regex=True)
    doc = doc.str.strip()
    return doc


#########################################
### Function: display_column_names     ####
#########################################


def display_column_names(df, max_columns=20):
    """
    This function takes a pd.DataFrame as input and prints the column names in individual lines
    If we have more than max_columns columns, it prints the total number of columns the first 5 and last 5 columns

    Parameters:
    - df: pd.DataFrame
    - max_columns: maximum number of columns to print, otherwise print the first 5 and last 5 columns
    """
    print(f"Number of columns: {df.shape[1]}")

    if df.shape[1] > max_columns:
        # Print first 5 columns as a single line
        print("First 5 names:")
        print(", ".join(df.columns[:5]))
        # Print last 5 columns as a single line
        print("Last 5 columns:")
        print(", ".join(df.columns[-5:]))
    else:
        print("Columns names:")
        # print 5 column names at a time
        for i in range(0, df.shape[1], 5):
            # Print 5 column names
            print(", ".join(df.columns[i : i + 5]))


############################################
### Function: load_source_data          ####
############################################


def load_source_data(files=False, verbose=True):
    """
    This function performs the following steps:
    - Loads the data.
    - Stores email text, labels and filenames in a DataFrame
    - Note: the label column is named "spam_label" to avoid conflict with "label" word columns
    - Removes duplicated entries and rows with missing values
    - Removes multiple whitespace, new lines and tab characters from text.
    - Resets DataFrame index.

    Parameters:
    - files: if True, the filenames are included in the DataFrame.
    """

    ################################
    # Composition of spam dataset  #
    ################################

    # Combine data from the following folders:
    folders = ["spam", "hard_ham", "spam_2", "easy_ham", "easy_ham 2", "easy_ham_2"]
    # Fix labels for each data set
    # 1 = spam, 0 = ham
    labels_dict = {
        "spam": 1,
        "hard_ham": 0,
        "spam_2": 1,
        "easy_ham": 0,
        "easy_ham 2": 0,
        "easy_ham_2": 0,
    }

    # Extract filenames and labels and store them in lists
    filenames = []
    labels = []

    for folder in folders:
        for file in os.listdir("data/spam_assasin/" + folder):
            if file != "cmds":
                fullpath = "data/spam_assasin/" + folder + "/" + file
                filenames.append(fullpath)
                labels.append(labels_dict[folder])

    # Extract text from emails
    docs = [get_email_content(fn) for fn in filenames]

    # Store text, label and filename in DataFrame
    # file names are optional
    if files:
        df = pd.DataFrame.from_dict(
            {"spam_label": labels, "text": docs, "filename": filenames}
        )
    else:
        df = pd.DataFrame.from_dict({"spam_label": labels, "text": docs})

    # Check duplicates in text column - to be removed
    duplicates = df[df.duplicated(subset="text", keep="first")]
    # Remove duplicated rows from DataFrame
    df.drop_duplicates(subset="text", keep="first", inplace=True)

    # Remove excess whitespace, new lines and tabs
    df["text"] = remove_excesspace(df["text"])

    # Remove rows with missing values and empty rows
    df["text"].replace("", np.nan, inplace=True)
    df["text"].replace(" ", np.nan, inplace=True)
    empty_emails = df["text"].isnull().sum()
    df.dropna(inplace=True)

    # Reset index
    df.reset_index(inplace=True)
    df.drop("index", axis=1, inplace=True)

    if verbose:
        print(len(docs), "emails loaded")
        print("Cleaning data set:")
        print(duplicates.shape[0], "duplicate emails found and removed")
        print(empty_emails, "empty emails found and removed")
        print()
        print(df.shape[0], "emails remaining")
        print()
        display_column_names(df)

    return df


#############################################
### Function:   store_labels               ####
#############################################


def store_labels(df):
    """
    This function is used incase later unlabelled data is generated but labels are needed
    The function
    - extracts the labels from the data frame
    - stores them in a csv file
    Note: the label column is named "spam_label" to avoid conflict with "label" word columns
    Outputs:
    - DataFrame of labels in the column "spam_label"
    """
    # Create data frame with labels in a column named "spam_label"
    labels_df = df[["spam_label"]]

    # Save to csv
    labels_df.to_csv("data/labels.csv", index=False)


#############################################
### Function:   load_labels               ####
#############################################


def load_labels():
    """
    Load labels from "data/labels.csv"
    """

    # Check if file exists
    if not os.path.exists("data/labels.csv"):
        print("Labels file not found. Please run store_labels() first.")
        return None
    else:
        # Load labels from csv
        labels_df = pd.read_csv("data/labels.csv")
        print(f"{labels_df.shape[0]} labels loaded")
        print("Labels found:", labels_df["spam_label"].unique())
        return labels_df


#########################################
### Function: extract_numeric_features   ####
#########################################


def extract_numeric_features(df, with_labels=True, store=True):
    """
    This function takes a pd.DataFrame as input and performs the following tasks:
    - counts characters, words, unique words, punctuation marks, uppercase & lowercase words,
      digits and alphabetic chars
    - removes 'URL:' and 'mailto:' strings from text
    - counts the number of HTML tags, e-mail addresses, URLs and twitter usernames.

    Outputs:
    - a pd.DataFrame with all counts.
    - with labels
    """

    docs = df["text"]
    spam_labels = df["spam_label"]
    print("Number of samples and columns of input:", df.shape)
    display_column_names(df)
    print()

    # Create empty lists for storing counts
    digit_counts = []
    alpha_counts = []
    chars_counts = []
    word_counts = []
    unique_word_counts = []
    punctuation_counts = []
    uppercase_word_counts = []
    lowercase_word_counts = []

    mail_counts = []
    url_counts = []
    mention_counts = []
    tag_counts = []
    hashtag_counts = []

    # Compile Regex patterns
    mail_pattern = re.compile(r"[<\[(]?[\w][\w.-]+@[\w.]+[>\])]?[:=]?[0-9]{0,}")
    url_regex = r"[<]?https?:\/\/(www\.)?[-a-zA-Z0-9@:,%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b([-a-zA-Z0-9()@,:%_\+.~#?&//=]*)[>]?"
    url_pattern = re.compile(url_regex)
    mention_pattern = re.compile(r"@[\w.-]+")
    hashtag_pattern = re.compile(r"#[\w]+")

    # Count characters
    chars_counts = docs.apply(len)

    # Count words
    word_counts = docs.apply(lambda x: len(x.split()))

    # Count unique words
    unique_word_counts = docs.apply(lambda x: len(set(x.split())))

    # Count punctuation marks
    punctuation_counts = docs.apply(
        lambda x: len([x for x in x if x in string.punctuation])
    )

    # Count uppercase words
    uppercase_word_counts = docs.apply(
        lambda x: len([wrd for wrd in x.split() if wrd.isupper()])
    )

    # Count lowercase words
    lowercase_word_counts = docs.apply(
        lambda x: len([wrd for wrd in x.split() if wrd.islower()])
    )

    # Count digits
    digit_counts = docs.apply(lambda x: len([x for x in x if x.isdigit()]))

    # Count alphabetic chars
    alpha_counts = docs.apply(lambda x: len([x for x in x if x.isalpha()]))

    for doc in docs:

        # Remove 'URL:' and 'mailto:' strings from text
        doc = re.sub("URL:", "", doc)
        doc = re.sub("mailto:", "", doc)

        # Count HTML Tags
        soup = BeautifulSoup(doc, "html.parser")
        tag_counts.append(len(soup.findAll()))

        # Count e-mail addresses (e.g. ilug@linux.ie)
        doc = mail_pattern.sub("EMAILHERE", doc)
        mail_counts.append(doc.count("EMAILHERE"))

        # Count URLs (e.g. https://lists.sourceforge.net/lists/listinfo/razor-user)
        doc = url_pattern.sub("URLHERE", doc)
        url_counts.append(doc.count("URLHERE"))

        # Count Twitter usernames (e.g. @username)
        doc = mention_pattern.sub("MENTIONHERE", doc)
        mention_counts.append(doc.count("MENTIONHERE"))

        # Count hashtags (##weddingdress)
        doc = hashtag_pattern.sub("HASHTAGHERE", doc)
        hashtag_counts.append(doc.count("HASHTAGHERE"))

    # Store features in a DataFrame
    num_features_df = pd.DataFrame(
        list(
            zip(
                mail_counts,
                tag_counts,
                url_counts,
                mention_counts,
                hashtag_counts,
                chars_counts,
                word_counts,
                unique_word_counts,
                punctuation_counts,
                uppercase_word_counts,
                lowercase_word_counts,
                digit_counts,
                alpha_counts,
            )
        ),
        columns=[
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
        ],
    )

    if with_labels:
        # Add labels to DataFrame
        num_features_df = pd.concat([num_features_df, spam_labels], axis=1)

    print("Numeric features extracted")
    print("Data size:", num_features_df.shape)
    display_column_names(num_features_df)
    # print("Columns names:", ", ".join(num_features_df.columns))

    # Save to csv
    if store:
        num_features_df.to_csv("data/num_features.csv", index=False)
        print("Numeric features saved to data/num_features.csv")

    return num_features_df


#########################################
### Section:   Text features         ####
#########################################

#########################################
### Function:   clean_corpus         ####
#########################################


def clean_corpus(df, remove_duplicates_and_empty=False, verbose=True):

    df = df.copy()
    docs = df["text"]
    clean_corpus = []

    """
    This function takes a pd.DataFrame as input and performs the following tasks:
    - removes 'URL:' and 'mailto:' strings from text
    - removes HTML tags, e-mail addresses, urls, twitter usernames and hashtags
    - removes multiple whitespace and strips leading and trailing whitespace
    - removes punctuation marks
    - removes ENGLISH_STOP_WORDS and words smaller than 3 characters
    Returns the preprocessed text.
    Parameters:
    - df: pd.DataFrame
    - remove_duplicates_and_empty: if True, removes duplicated and empty rows
      # by default deactivated so original email are treated as distinct
    """

    # Compile Regex patterns
    mail_pattern = re.compile(r"[<\[(]?[\w][\w.-]+@[\w.]+[>\])]?[:=]?[0-9]{0,}")
    url_regex = r"[<]?https?:\/\/(www\.)?[-a-zA-Z0-9@:,%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b([-a-zA-Z0-9()@,:%_\+.~#?&//=]*)[>]?"
    url_pattern = re.compile(url_regex)
    mention_pattern = re.compile(r"@[\w.-]+")
    hashtag_pattern = re.compile(r"#[\w]+")
    regex_chars = r"[^a-zA-Z\s]*"
    special_char_pattern = re.compile(r"([{.(-)!:\-,\=\/\<\>?}])")

    for doc in docs:

        # Remove 'URL:' and 'mailto:' strings from text
        doc = re.sub("URL:", "", doc)
        doc = re.sub("mailto:", "", doc)
        # doc = irish_pattern.sub('',doc)

        # Remove HTML Tags
        soup = BeautifulSoup(doc, "html.parser")
        doc = soup.get_text()
        doc = doc.replace(r"[\s]+", " ")
        doc = doc.strip()

        # Remove e-mail addresses (e.g. ilug@linux.ie)
        doc = mail_pattern.sub("", doc)

        # Remove URLs (e.g. https://lists.sourceforge.net/lists/listinfo/razor-user)
        doc = url_pattern.sub("", doc)

        # Remove Twitter usernames (e.g. @username)
        doc = mention_pattern.sub("", doc)

        # Remove hashtags (#weddingdress)
        doc = hashtag_pattern.sub("", doc)

        # Remove excess whitespace
        doc = doc.replace(r"[\s]+", " ")
        doc = doc.strip()

        # Convert to lowercase
        doc = doc.lower()

        # Remove special characters
        doc = special_char_pattern.sub(" \\1 ", doc)
        doc = re.sub(regex_chars, "", doc)

        # Remove excess whitespace
        doc = doc.replace(r"[\s]+", " ")
        doc = doc.strip()

        # Remove small words (smaller than 3 characters)
        doc = " ".join(re.findall(r"[\w]{4,}", doc))

        # Removes very long words (longer than 40 characters)
        doc = doc.replace(r"[a-zA-Z]{40,}", "")
        longp = re.compile(r"[a-zA-Z]{40,}")
        doc = longp.sub("", doc)

        # Remove ENGLISH_STOP_WORDS
        doc = " ".join([w for w in doc.split() if w not in ENGLISH_STOP_WORDS])

        clean_corpus.append(doc)

    # Store clean corpus in a new column
    df["text_cleaned"] = clean_corpus

    if verbose:
        print("Number of samples:", df.shape[0])
    display_column_names(df)
    print()

    # Check for duplicated cleaned text
    duplicates = df[df.duplicated(subset="text_cleaned", keep="first")]
    if verbose:
        print("Number of duplicate cleaned texts found:", duplicates.shape[0])
    # Remove rows with duplicate cleaned texts
    if remove_duplicates_and_empty:
        df.drop_duplicates(subset="text_cleaned", keep="first", inplace=True)
        if verbose:
            print("Number of samples without duplicates:", df.shape[0])

    # Check whether a string is an empty string or just whitespaces
    empty_text_mask = df["text_cleaned"].str.strip().eq("")
    empty_rows_df = df[empty_text_mask]
    # Save rows with empty cleaned text to csv for later instpection
    empty_rows_df.to_csv("data/empty_cleared_text_rows.csv", index=False)
    if verbose:
        print("Number of empty texts found:", empty_text_mask.sum())
    # Remove rows with empty cleaned text
    if remove_duplicates_and_empty:
        # Remove rows with empty cleaned text
        df = df[~empty_text_mask]
        if verbose:
            print("Number of samples without empty text:", df.shape[0])

    # Reset index
    df.reset_index(inplace=True)
    df.drop("index", axis=1, inplace=True)

    if verbose:
        print()
        print("Email texts cleaned")
        if remove_duplicates_and_empty:
            print("Duplicates and empty texts were removed")
        print("Number of samples:", df.shape[0])

    df.to_csv("data/df_cleaned", index=True)

    return df


#########################################
### Function:  load_cleaned_text     ####
#########################################


### This function is currently creating bugs with  plot-functions in EDA tools
def load_cleaned_text():
    """
    This function loads the cleaned text from the data frame
    and returns a DataFrame with the cleaned text.
    """
    # Check if file exists
    if not os.path.exists("data/df_cleaned"):
        print("Cleaned text file not found. Please run clean_corpus() first.")
        return None
    else:
        # Load cleaned text from csv
        df = pd.read_csv("data/df_cleaned", index_col=0)
        print(f"{df.shape[0]} cleaned emails loaded")
        print("Data includes labels in the column 'spam_label'")
        print(f"The data set has {df.shape[0]} rows, {df.shape[1]} columns")
        return df


#########################################
### Function:  show_clean_text       ####
#########################################


def show_clean_text(df):
    """
    This function takes a random document number (doc_nbr) and
    outputs:
    - the original document
    - the cleaned document

    For very long texts, only the first
    2'000 characters are printed on the screen.
    """

    df = df.copy()
    doc_nbr = randrange(len(df))

    # Document to print
    doc = df.iloc[doc_nbr : doc_nbr + 1, :].copy()
    doc_length = len(doc["text"].values[0])

    # Print only 2'000 chars
    value = 2000

    orig_text = doc["text"].values[0][0:value]
    # Colab formating: wrap text
    orig_text = "\n".join(textwrap.wrap(orig_text, 100))
    print("\nOriginal document:\n\n{}\n".format(orig_text))

    # Clean text if not already cleaned
    if "text_cleaned" not in df.columns:
        # Clean text
        clean_text = clean_corpus(
            doc, remove_duplicates_and_empty=False, verbose=False
        )["text_cleaned"][0][0:value]
    else:
        clean_text = doc["text_cleaned"].values[0][0:value]
    # Colab formating: wrap text
    clean_text = "\n".join(textwrap.wrap(clean_text, 100))
    print("Cleaned document:\n\n{}".format(clean_text))


#############################################
### Function:   extract_text_features    ####
#############################################


def extract_text_features(df, vectorizer="count", with_labels=True, store=True, compress=True):
    """
    This function takes a pd.DataFrame as input and performs the following tasks:
    - check if cleaned_text column already exists
    - otherwise clean text column (check default settigns of clean_corpus)
    - apply count-vectorizer or otherwise tfidf-vectorizer
    Outputs:
    - DataFrame with all count above fixed limit
    - includes labels if specified
    """

    df = df.copy()
    spam_labels = df["spam_label"]

    # Clean text if not already cleaned
    if "text_cleaned" not in df.columns:
        # Clean text
        df = clean_corpus(df)

    if vectorizer == "count":
        print("Count Vectorizer selected")
        file_add = "_count"
    elif vectorizer == "tfidf":
        print("TF-IDF Vectorizer selected")
        file_add = "_tfidf"
    else:
        print("You can choose between 'count' and 'tfidf' vectorizer.")
        print("No valid vectorizer name selected. Defaulting to Count Vectorizer.")
        vectorizer = "count"
        file_add = "_count"

    # Extract features
    ## Note: Params below were optimized for maximizing f1-score for validation splits (cv = 5)
    if vectorizer == "tfidf":
        # TfidfVectorizer
        vectorizer = TfidfVectorizer(
            analyzer="word",  # default
            ngram_range=(1, 1),  # default
            token_pattern="(?u)\\b\\w+\\b",  # default
            dtype=np.float32, 
            min_df=8,
            max_features=10000,
            stop_words=list(ENGLISH_STOP_WORDS),
        )
    else:
        # CountVectorizer
        vectorizer = CountVectorizer(
            analyzer="word",  # default
            ngram_range=(1, 1),  # default
            token_pattern="(?u)\\b\\w+\\b",  # default
            dtype=np.float32, 
            min_df=5,
            max_features=10000,
            stop_words=list(ENGLISH_STOP_WORDS),
        )

    # Transform corpus
    X_corpus = vectorizer.fit_transform(df["text_cleaned"])
    print("Corpus data type: ", type(X_corpus))
    perc=X_corpus.nnz / (X_corpus.shape[0] * X_corpus.shape[1])
    print(100*perc, "% are null values")
    
    # Get the vocabulary
    vocab = vectorizer.get_feature_names_out()

    text_features_df = pd.DataFrame(X_corpus.toarray(), columns=vocab)

    # Add labels to DataFrame
    if with_labels:
        text_features_df["spam_label"] = spam_labels

    display_column_names(text_features_df)

    # Save to csv
    if store:
        if compress:
            file_name = "data/sparse_text_features" + file_add + ".npz"
            sp.save_npz(file_name, X_corpus, compressed=True)
            print("Text features saved to ", file_name)
        else:
            file_name = "data/text_features" + file_add + ".csv"
            text_features_df.to_csv(file_name, index=False)
            print("Text features saved to", file_name)

    return text_features_df


###################################################
### Function:   load_feature_space             ####
###################################################


def load_feature_space(features="text", no_labels=False):
    """
    This function loads the previously generated features spaces
    Parameters:
    - features: "num", "text", "num_text" or "embedding"
        - "num": load numeric features
        - "text": load text features
        - "num_text": load numeric and text features
        - "embedding": load email embeddings
    - no_labels: if True, the "spam_label" column is dropped
    If no_labels=True then drop "spam_label" column
    Output:
    - data frame with specified features
    - with labels unless no_labels=True
    """

    labels_df = pd.read_csv("data/labels.csv")
    
    if features == "num":
        # Load numeric features
        df = pd.read_csv("data/num_features.csv")
        print("Numeric features loaded")
    elif features == "text":
        # Load text features
        X_text = sp.load_npz("data/sparse_text_features_tfidf.npz")
        text_features_df = pd.DataFrame.sparse.from_spmatrix(X_text)
        text_features_df = text_features_df.sparse.to_dense().astype(np.float32)
        # Add labels 
        # - later consider use of `scipy.sparse.hstack` instead but beware some prediction analyses failed in supervised learning
        df= pd.concat(
            [text_features_df, labels_df], axis=1)
        
        print("Text features loaded")
    elif features == "num_text":
        # Load numeric features
        num_features_df = pd.read_csv("data/num_features.csv")
        # Load text features
        X_text = sp.load_npz("data/sparse_text_features_tfidf.npz")
        text_features_df = pd.DataFrame.sparse.from_spmatrix(X_text)
        text_features_df = text_features_df.sparse.to_dense().astype(np.float32)
        # Merge the two frames
        # - later consider use of `scipy.sparse.hstack` instead
        df = pd.concat(
            [num_features_df, text_features_df], axis=1
        )
        print("Numeric and text features loaded")
    if features == "embedding":
        # Load email embeddings
        X = np.load("data/embeddings.npz")["X"]
        X_df = pd.DataFrame(X)
        df=pd.concat(
            [X_df, labels_df], axis=1)
        print("Email embeddings loaded")

    if features == "embedding_full":
        # Load email embeddings
        X = np.load("data/embeddings_full.npz")["X"]
        X_df = pd.DataFrame(X)
        df=pd.concat(
            [X_df, labels_df], axis=1)
        print("Email embeddings loaded with full 64bit precision")

    if no_labels:
        df.drop("spam_label", axis=1, inplace=True)
        print("Data loaded without labels")
    else:
        print("Data includes labels in the column 'spam_label'")
    print(f"The data set has {df.shape[0]} rows, {df.shape[1]} columns")
    return df
