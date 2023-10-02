import re
import urllib.parse
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer


def tokenizer_pakhare(url):
    """Separates feature words from the raw data
    Keyword arguments:
      url ---- The full URL

    :Returns -- The tokenized words; returned as a list
    """

    # Split by slash (/) and dash (-)
    tokens = re.split('[/-]', url)

    for i in tokens:
        # Include the splits extensions and subdomains
        if i.find(".") >= 0:
            dot_split = i.split('.')

            # Remove .com and www. since they're too common
            if "com" in dot_split:
                dot_split.remove("com")
            if "www" in dot_split:
                dot_split.remove("www")

            tokens += dot_split

    return tokens

def tokenizer(url):
    """Separates feature words from the raw data
    Keyword arguments:
      url ---- The full URL

    :Returns -- The tokenized words; returned as a list
    """
    # decode UTF - URL
    url = urllib.parse.unquote(url)

    # Split by slash (/), dash (-), full stop (.), question mark (?), hashtag (#),
    # (:), (=), (&), (_)
    tokens = re.split('[/\-.?#:=& _]', url)
    tokens = [item for item in tokens if item]
    return tokens



def apply_TFIDF(X_train, X_test):
    # print("- Training TF-IDF Vectorizer -")
    # tVec = TfidfVectorizer(tokenizer=tokenizer, stop_words='english', dtype=np.float32, ngram_range=(1, 1))
    tVec = TfidfVectorizer(tokenizer=tokenizer_pakhare, dtype=np.float32, ngram_range=(1, 1))
    tfidf_X_train = tVec.fit_transform(X_train['URLs'])
    # use transform method, because the vectorizers should be trained on the training set
    tfidf_X_test = tVec.transform(X_test['URLs'])
    return tfidf_X_train, tfidf_X_test
