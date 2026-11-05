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


####################################
### Function: get_email_content ####
####################################


def get_email_content(email_path):
    """
    This function uses the email library to extract text from mails
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
    """

    doc = doc.str.replace("[\s]+", " ")
    doc = doc.str.strip()
    return doc


####################################
### Function: load_data         ####
####################################


def load_data(files=False):
    """
    This function performs the following steps:
    - Loads the data.
    - Stores email text, labels and filenames in a DataFrame.
    - Removes duplicated entries and rows with missing values
    - Removes multiple whitespace, new lines and tab characters from text.
    - Resets DataFrame index.
    """

    ################
    # Spam dataset #
    ################

    folders = ["spam", "hard_ham", "spam_2", "easy_ham", "easy_ham 2", "easy_ham_2"]
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
    if files:
        df = pd.DataFrame.from_dict(
            {"spam_label": labels, "text": docs, "filename": filenames}
        )
    else:
        df = pd.DataFrame.from_dict({"spam_label": labels, "text": docs})

    # Check duplicates in text column - to be removed
    duplicates = df[df.duplicated(subset="text", keep="first")]
    print("Cleaning duplicates")
    print("Number of duplicates found", duplicates.shape[0])

    # Remove duplicated rows from DataFrame
    print("Data with duplicates", df.shape[0])
    df.drop_duplicates(subset="text", keep="first", inplace=True)
    print("After duplicates are removed", df.shape[0])

    # Remove excess whitespace, new lines and tabs
    df["text"] = remove_excesspace(df["text"])

    # Remove rows with missing values and empty rows
    df["text"].replace("", np.nan, inplace=True)
    df["text"].replace(" ", np.nan, inplace=True)
    df.dropna(inplace=True)
    print("After removing empty text", df.shape[0])

    # Reset index
    df.reset_index(inplace=True)
    df.drop("index", axis=1, inplace=True)

    print("Data loaded.")
    print("Data size:", df.shape)
    print("Columns names:", ", ".join(df.columns))

    return df


#########################################
### Function: get_numeric_features   ####
#########################################


def get_numeric_features(df, with_labels=False):
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
    print("Columns names:", ", ".join(num_features_df.columns))

    return num_features_df


#########################################
### Function:   clean_corpus         ####
#########################################


def clean_corpus(df):

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
    """

    # Compile Regex patterns
    mail_pattern = re.compile(r"[<\[(]?[\w][\w.-]+@[\w.]+[>\])]?[:=]?[0-9]{0,}")
    url_regex = r"[<]?https?:\/\/(www\.)?[-a-zA-Z0-9@:,%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b([-a-zA-Z0-9()@,:%_\+.~#?&//=]*)[>]?"
    url_pattern = re.compile(url_regex)
    mention_pattern = re.compile(r"@[\w.-]+")
    hashtag_pattern = re.compile(r"#[\w]+")
    regex_chars = "[^a-zA-Z\s]*"
    special_char_pattern = re.compile(r"([{.(-)!:\-,\=\/\<\>?}])")

    for doc in docs:

        # Remove 'URL:' and 'mailto:' strings from text
        doc = re.sub("URL:", "", doc)
        doc = re.sub("mailto:", "", doc)
        # doc = irish_pattern.sub('',doc)

        # Remove HTML Tags
        soup = BeautifulSoup(doc, "html.parser")
        doc = soup.get_text()
        doc = doc.replace("[\s]+", " ")
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
        doc = doc.replace("[\s]+", " ")
        doc = doc.strip()

        # Convert to lowercase
        doc = doc.lower()

        # Remove special characters
        doc = special_char_pattern.sub(" \\1 ", doc)
        doc = re.sub(regex_chars, "", doc)

        # Remove excess whitespace
        doc = doc.replace("[\s]+", " ")
        doc = doc.strip()

        # Remove small words (smaller than 3 characters)
        doc = " ".join(re.findall("[\w]{4,}", doc))

        # Removes very long words (longer than 40 characters)
        doc = doc.replace(r"[a-zA-Z]{40,}", "")
        longp = re.compile(r"[a-zA-Z]{40,}")
        doc = longp.sub("", doc)

        # Remove ENGLISH_STOP_WORDS
        doc = " ".join([w for w in doc.split() if w not in ENGLISH_STOP_WORDS])

        clean_corpus.append(doc)

    # Store clean corpus in a new column
    df["text_cleaned"] = clean_corpus

    print("Number of samples:", df.shape[0])

    # Remove duplicated rows
    df.drop_duplicates(subset="text_cleaned", keep="first", inplace=True)
    print("Number of samples without duplicates:", df.shape[0])

    # Remove empty rows
    df["text_cleaned"].replace("", np.nan, inplace=True)
    df.dropna(inplace=True)
    print("Number of samples without empty text:", df.shape[0])

    # Reset index
    df.reset_index(inplace=True)
    df.drop("index", axis=1, inplace=True)

    print("Data cleaned")
    return df


#########################################
### Function:  show_clean_text       ####
#########################################


def show_clean_text(df):
    """
    This function takes a document number (doc_nbr) and
    outputs:
    - the original document
    - the cleaned document

    For very long texts, only the first
    2'000 characters are printed on the screen.
    """

    doc_nbr = randrange(len(df))

    # Document to print
    doc = df.iloc[doc_nbr : doc_nbr + 1, :]

    doc_length = len(doc["text"].values[0])

    # Print only 2'000 chars
    value = 2000

    orig_text = doc["text"].values[0][0:value]
    # Colab formating: wrap text
    orig_text = "\n".join(textwrap.wrap(orig_text, 100))
    print("\nOriginal document:\n\n{}\n".format(orig_text))

    # Colab formating: wrap text
    clean_text = clean_corpus(doc)["text_cleaned"][0][0:value]
    clean_text = "\n".join(textwrap.wrap(clean_text, 100))
    print("Cleaned document:\n\n{}".format(clean_text))


#############################################
### Function:   store_labels               ####
#############################################


def store_labels(df):
    """
    This function
    - extracts the labels from the data frame
    - stores them in a csv file
    - returns them
    To avoid conflict with "label" word column the column has been named "spam_label"
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


#############################################
### Function:   get_text_features        ####
#############################################


def get_text_features(df, vectorizer="count", with_labels=False):
    """
    This function takes a pd.DataFrame as input and performs the following tasks:
    - clean text column
    - apply count-vectorizer or otherwise tfidf-vectorizer
    Outputs:
    - DataFrame with all count above fixed limit
    - without labels (labels are stored separately in a csv file)
    """

    # Clean text
    df = clean_corpus(df)

    # Extract features
    ## Note: Params below were optimized for maximizing f1-score for validation splits (cv = 5)
    if vectorizer == "count":
        # CountVectorizer
        vectorizer = CountVectorizer(
            analyzer="word",  # default
            ngram_range=(1, 1),  # default
            token_pattern="(?u)\\b\\w+\\b",  # default
            min_df=5,
            max_features=20000,
            stop_words=list(ENGLISH_STOP_WORDS),
        )
    else:
        # TfidfVectorizer
        vectorizer = TfidfVectorizer(
            analyzer="word",  # default
            ngram_range=(1, 1),  # default
            token_pattern="(?u)\\b\\w+\\b",  # default
            min_df=5,
            max_features=20000,
            stop_words=list(ENGLISH_STOP_WORDS),
        )

    # Transform corpus
    corpus_bow = vectorizer.fit_transform(df["text_cleaned"])

    # Get the vocabulary
    vocab = vectorizer.get_feature_names_out()

    text_features_df = pd.DataFrame(
        corpus_bow.toarray(), columns=vectorizer.get_feature_names_out()
    )

    return text_features_df


#############################################
### Function:   get_all_features        ####
#############################################


def get_all_features(df, vectorizer="count", with_labels=False):
    """
    This function takes a pd.DataFrame as input and performs the following tasks:
    - get numeric features
    - get text features
    - merge the two frames
    Outputs:
    - DataFrame with all features
    - with labels
    """

    # Get numeric features
    num_features_df = get_numeric_features(df, with_labels=False)

    # Get text features
    text_features_df = get_text_features(df, vectorizer=vectorizer, with_labels=False)

    # Merge the two frames
    if with_labels:
        all_features_df = pd.merge(
            num_features_df.drop("spam_label", axis=1),
            text_features_df,
            left_index=True,
            right_index=True,
        )
    else:
        # Merge the two frames
        # without labels
        all_features_df = pd.merge(
            num_features_df, text_features_df, left_index=True, right_index=True
        )

    return all_features_df


###################################################
### Function:   create_all_feature_spaces      ####
###################################################


def create_all_feature_spaces(df, vectorizer="count"):
    """
    This function takes a pd.DataFrame as input and performs the following tasks:
    - get numeric features
    - get text features
    - merge the two frames
    Outputs:
    - DataFrame with all features
    - with labels
    """

    # Get numeric features
    num_features_df = get_numeric_features(df)
    # Save to csv
    num_features_df.to_csv("data/num_features.csv", index=False)
    print("Numeric features saved to data/num_features.csv")

    # Get text features
    text_features_df = get_text_features(df, vectorizer=vectorizer)
    # Save to csv
    text_features_df.to_csv("data/text_features.csv", index=False)
    print("Text features saved to data/text_features.csv")

    # Merge the two frames
    all_features_df = pd.merge(
        num_features_df, text_features_df, left_index=True, right_index=True
    )
    # Save to csv
    all_features_df.to_csv("data/all_features.csv", index=False)
    print("All features saved to data/all_features.csv")


###################################################
### Function:   load_all_feature_spaces      ####
###################################################


def load_all_feature_spaces(with_labels=False):
    """
    This function loads the previously generated features spaces:
    If with_labels=False then drop "spam_label" column
    Outputs:
    - num_features
    - text_features
    - all_features
    - embeddings
    """

    # Load numeric features
    num_features_df = pd.read_csv("data/num_features.csv")
    print("Numeric features loaded")
    print(f"{num_features_df.shape[0]} rows, {num_features_df.shape[1]} columns")
    # Load text features
    text_features_df = pd.read_csv("data/text_features.csv")
    print("Text features loaded")
    print(f"{text_features_df.shape[0]} rows, {text_features_df.shape[1]} columns")
    # Load all features
    all_features_df = pd.read_csv("data/all_features.csv")
    print("All features loaded")
    print(f"{all_features_df.shape[0]} rows, {all_features_df.shape[1]} columns")
    # Load email embeddings
    embeddings_df = pd.read_csv("data/email_embeddings.csv")
    print("Email embeddings loaded")
    print(f"{embeddings_df.shape[0]} rows, {embeddings_df.shape[1]} columns")

    if with_labels == False:
        num_features_df.drop("spam_label", axis=1, inplace=True)
        text_features_df.drop("spam_label", axis=1, inplace=True)
        all_features_df.drop("spam_label", axis=1, inplace=True)
        embeddings_df.drop("spam_label", axis=1, inplace=True)
        print("Data loaded without labels")
    else:
        print("Data includes labels in the column 'spam_label'")

    return num_features_df, text_features_df, all_features_df, embeddings_df


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

    sns.barplot(x=class_counts.index, y=pd.value_counts(df["spam_label"]))
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

    num_features_df = get_numeric_features(df, with_labels=with_labels)
    ### old setting
    ### num_features_df = pd.concat([num_features, df["spam_label"]], axis=1)

    # Plot results
    # ============
    cols = num_features_df.columns[:-1]

    fig, axes = plt.subplots(nrows=5, ncols=3, figsize=(16, 4 * 5))

    for col, ax in zip(cols, axes.ravel()):
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

    # return num_features_df


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


###############################################
### Function: plot_most_common_words_old   ####
###############################################


def plot_most_common_words_old(df, N):
    """
    This function computes the N most common words in
    hams and spams and plots the results in a common
    histogram.
    """

    # Text data needs to be converted into a nltk.Text() object to be able to use it with nltk tools

    # Join "cleaned text" in single strings
    corpus_0 = " ".join([text for text in df[df["spam_label"] == 0]["text_cleaned"]])
    corpus_1 = " ".join([text for text in df[df["spam_label"] == 1]["text_cleaned"]])

    # nltk.Text() expects tokenized text
    # Create an instance of WordPunctTokenizer
    tokenizer = nltk.WordPunctTokenizer()

    corpusText_0 = nltk.Text(tokenizer.tokenize(corpus_0))
    corpusText_1 = nltk.Text(tokenizer.tokenize(corpus_1))

    # Get N most common words in class 0
    tups = Counter(corpusText_0).most_common(N)
    dictionary = {}
    dict_0 = Convert(tups, dictionary)

    # Get N most common words in class 1
    tups = Counter(corpusText_1).most_common(N)
    dictionary = {}
    dict_1 = Convert(tups, dictionary)

    # Create lists of most common words in both classes
    words_0 = list(dict_0.keys())
    words_1 = list(dict_1.keys())

    # Create list of common words in hams and spams
    common_list = set(words_0 + words_1)

    # Count common words in hams and spams
    counts_0 = [dict_0[w] if w in words_0 else 0 for w in common_list]
    counts_1 = [dict_1[w] if w in words_1 else 0 for w in common_list]

    # Store results in DataFrame and sort values
    df = pd.DataFrame(
        list(zip(common_list, counts_0, counts_1)),
        columns=["Word", "ham_counts", "spam_counts"],
    )
    df = df.sort_values(by="ham_counts", ascending=False)

    # Plot most common words in hams and spams
    # -----------------------------------------
    plt.figure(figsize=(8, 8))

    plt.barh(
        y=df.Word, width=df.ham_counts, edgecolor="black", label="Non-spam", alpha=0.3
    )
    plt.barh(
        y=df.Word, width=df.spam_counts, edgecolor="black", label="Spam", alpha=0.3
    )
    plt.xlabel("Word counts", **label_specs)
    plt.title(
        "Top " + str(N) + " most frequent words in spam and non-spam", **title_specs
    )
    # change font size
    plt.legend(**label_specs)
    plt.xticks(**label_specs)
    plt.yticks(**label_specs)


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
        # Join documents in single strings
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
        ).sort_values(by="Counts", ascending=False)
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
            plt.xlabel("Word counts", **label_specs)
            plt.xticks(**label_specs)
            plt.yticks(**label_specs)
        plt.title(f"{N} most frequent words", **title_specs)


###########################################
### Function: plot_most_common_tokens   ####
###########################################


def plot_most_common_tokens(df, N, per_1000=False):
    """
    This function computes the N most common words in
    hams and spams and plots the results in a common
    histogram.
    """

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
    This functio extracts BoW features for a
    toy corpus.
    """

    # Toy corpus
    corpus = [
        "I enjoy paragliding.",
        "I like NLP.",
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
### Function: train_test_split_old     ####
###########################################


def train_test_split_old(df):
    """
    This function performs train/test splitting
    """

    return train_test_split(
        df, test_size=0.3, stratify=df["spam_label"], random_state=0
    )


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
### Function: fit_model - old version  ####
###########################################


def fit_model_old(df_train):
    """
    This function performs the following:
    - fits a LogReg model.

    **Note**: The model parameters were optimized with GridSearchCV
              in a previous calculation to maximize f1-score on the
              validation splits (cv=5).
    """

    # Train set: features
    X_train = df_train["text_cleaned"].values

    # Train set: Labels
    y_train = df_train["spam_label"].values

    # Define pipeline
    model = Pipeline(
        [
            ("vectorizer", CountVectorizer()),
            (
                "lr",
                LogisticRegression(
                    solver="liblinear", class_weight="balanced", random_state=None
                ),
            ),
        ]
    )

    ## Optimized parameters ##
    ## -------------------- ##
    ## Note: Params optimized for maximizing f1-score for validation splits (cv = 5)
    optimized_params = {
        "lr__C": 0.1,
        "vectorizer__max_features": 20000,
        "vectorizer__min_df": 5,  # was 2, but changed for statistical stability
        "vectorizer__ngram_range": (1, 1),
    }

    model.set_params(
        lr__C=optimized_params["lr__C"],
        vectorizer__max_features=optimized_params["vectorizer__max_features"],
        vectorizer__min_df=optimized_params["vectorizer__min_df"],
        vectorizer__ngram_range=optimized_params["vectorizer__ngram_range"],
    )

    # Fit model
    model.fit(X_train, y_train)

    return model


###########################################
### Function: fit_model - new version  ####
###########################################


def fit_model(df_train, C=1):
    """
    This function performs the following:
    - fits a LogReg model.

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

    return model


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
        X_test = df_test.drop(columns=["spam_label"]).values

        # True labels
        y_true = df_test["spam_label"].values

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
### Function: visualize_coefficients   ####
###########################################

# Inspiration: "Introduction to Machine Learning with Python", A. Muller
# https://github.com/amueller/introduction_to_ml_with_python


def visualize_coefficients(model, n_top_features=25):

    feature_names = model.named_steps["vectorizer"].get_feature_names_out()
    coefficients = model.named_steps["lr"].coef_

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

    coefficients = coefficients.squeeze()
    coefficients = coefficients.ravel()

    # Get coefficients with large absolute values
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
    plt.legend(**label_specs)
    plt.xticks(**label_specs)
    plt.yticks(**label_specs)


###########################################
### Function: error_analysis           ####
###########################################


def error_analysis(df_test, model, doc_nbr, n_top=25):
    """
    Prints :
    - text,
    - probabilities
    - top features
    for a selected missclassified sample
    """

    # Add a new column to df_test for predictions
    df_test["prediction"] = model.predict(df_test["text_cleaned"].values)

    # Compute probabilities for test set
    y_test_probs = model.predict_proba(df_test["text_cleaned"].values)

    # Add new columns for probabilities
    df_test["proba_0"] = y_test_probs[:, 0]
    df_test["proba_1"] = y_test_probs[:, 1]

    # Misclassified samples
    idx = df_test["spam_label"] != df_test["prediction"]

    # New dataframe for misclassified samples
    df_missed = df_test.loc[idx, :]

    # Reset index
    temp = df_missed.copy()
    temp.reset_index(inplace=True)

    doc_nbr = doc_nbr

    # Create class dictionary
    class_dict = dict(zip(["0", "1"], ["Non-spam", "Spam"]))

    # Get vocabulary and model coefficients
    feature_names = model.named_steps["vectorizer"].get_feature_names_out()
    coefficients = model.named_steps["lr"].coef_
    coefficients = coefficients.squeeze()

    # Create vocabulary vs. model coefficients dictionary
    dict1 = dict(zip(feature_names, coefficients))

    # Tokenize text
    list_tokens = temp.loc[doc_nbr, "text_cleaned"].split()

    # Remove tokens if not in vocabulary
    list_tokens = [tok for tok in list_tokens if tok in feature_names]

    # Get model coefficients for text tokens
    doc_coeffs = []
    for token in list_tokens:
        doc_coeffs.append(dict1[token])

    # Select top coefficients
    n_top = n_top

    # Store indexes for top coefficients
    interesting_coefficients = np.argsort(np.abs(doc_coeffs))[-n_top:]

    # Convert to numpy array
    coef = np.array(doc_coeffs)
    toks = np.array(list_tokens)

    # Store sign for plotting
    coef_sign = [1 if s >= 0 else 0 for s in coef[interesting_coefficients]]

    top_features = (
        pd.DataFrame(
            zip(
                toks[interesting_coefficients],
                coef[interesting_coefficients],
                coef_sign,
            ),
            columns=["feature", "coef", "class"],
        )
        .groupby("feature")
        .sum()
    )
    top_features.sort_values("coef", ascending=True, inplace=True)
    top_features["class"] = top_features["class"].apply(simple_func)
    top_features.reset_index(inplace=True)

    ## Outputs ##
    print("Document index:", doc_nbr, "\n")
    print("\nOriginal Text")
    print("=============")
    orig_text = temp.loc[doc_nbr, "text"][0:2000]
    # Colab formating: wrap text
    orig_text = "\n".join(textwrap.wrap(orig_text, 100))
    print(orig_text, "\n")

    print("\nCleaned text")
    print("============")
    clean_text = temp.loc[doc_nbr, "text_cleaned"][0:2500]
    # Colab formating: wrap text
    clean_text = "\n".join(textwrap.wrap(clean_text, 100))
    print(clean_text, "\n\n")

    print("Actual class:    ", class_dict[str(temp.loc[doc_nbr, "spam_label"])])
    print("Predicted class: ", class_dict[str(temp.loc[doc_nbr, "prediction"])], "\n\n")

    print("Predicted probabilities")
    print("========================")
    print(
        "\nNon-spam:   {}\nSpam:       {}\n".format(
            np.round(temp.loc[doc_nbr, "proba_0"], 4),
            np.round(temp.loc[doc_nbr, "proba_1"], 4),
        )
    )

    # Plot top features
    font_ticks = {"fontname": "Arial", "size": "12"}
    plt.figure(figsize=(6, 10))

    if top_features[top_features["class"] == 1].shape[0] != 0:
        plt.barh(
            y=top_features[top_features["class"] == 1]["feature"],
            width=top_features[top_features["class"] == 1]["coef"],
            color="black",
            height=0.6,
            label="Spam",
        )
    if top_features[top_features["class"] == 0].shape[0] != 0:
        plt.barh(
            y=top_features[top_features["class"] == 0]["feature"],
            width=top_features[top_features["class"] == 0]["coef"],
            color="white",
            edgecolor="grey",
            height=0.6,
            label="Non-spam",
        )
    plt.legend()
    plt.xlabel("Feature Importance")
    plt.ylabel("Top Features")


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


def run_pca(df, N="all"):
    """
    This function
    - performs PCA on the data frame
    - plots the explained variance ratio as barplot and the cumulative explained variance as a step plot in the same figure
    - extracts the number of components need for each multiple of 10% explained variance
    - creates a scatterplot of the first two components
    Output:
    - returns the transformed data frame
    """

    # Standardize the data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df)

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

    # Extract number of components for each multiple of 10% explained variance
    if N == "all":
        N = np.arange(0.1, 1.1, 0.1)
        for n in N:
            num_components = np.argmax(np.cumsum(explained_variance) >= n) + 1
            print(
                f"Number of components for {n:.0%} explained variance: {num_components}"
            )

    # Scatterplot of the first two components
    plt.figure(figsize=(8, 8))
    plt.scatter(X_pca[:, 0], X_pca[:, 1], alpha=0.5, s=10)
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.title("PCA: Scatterplot of the First Two Components")

    return pd.DataFrame(X_pca)
