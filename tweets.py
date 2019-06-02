import os
import json
import re, string, unicodedata
import nltk
from nltk import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import LancasterStemmer, WordNetLemmatizer
import numpy as np
from nltk.tokenize import RegexpTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from scipy.cluster import  hierarchy
import matplotlib.pyplot as plt
from operator import itemgetter

def mirror(arg):
    return arg

nltk.download('wordnet')
nltk.download('stopwords')


def removeURL(str):
    return re.sub(r"http\S+", "", str)

def lemmatization(tokens):
    lemmatizer = WordNetLemmatizer()
    newTokens = []
    for token in tokens:
        newToken = lemmatizer.lemmatize(token, pos='v')
        newTokens.append(newToken)

    return newTokens

def removeStopWords(tokens):
    newTokens = []
    for token in tokens:
        if token not in stopwords.words('english'):
            newTokens.append(token)
    return newTokens


def main():
    os.chdir("/home/meg/Downloads")
    tokenizer = RegexpTokenizer(r'\w+')
    with open('tweets.json') as json_file:
        data = json.load(json_file)
    tweets = data['tweets']
    for tweet in tweets:
        text = tweet['text']
        text = removeURL(text)
        text = text.lower()
        tokens = tokenizer.tokenize(text)
        tokens = lemmatization(tokens)
        tokens = removeStopWords(tokens)
        tweet['tokens'] = tokens
    tfidf = TfidfVectorizer(tokenizer=mirror, stop_words='english', lowercase=False)
    listOfTokens = []
    for tweet in tweets:
        listOfTokens.append(tweet['tokens'])

    tfidf_matrix = tfidf.fit_transform(listOfTokens)
    tfidf_matrix = tfidf_matrix.todense()
    threshold = 0.03
    Z = hierarchy.linkage(tfidf_matrix, "average", metric="cosine")
    C = hierarchy.fcluster(Z, threshold, criterion="distance")
    clusters = C.tolist()
    dict = {}
    for c, tweet in zip(clusters, tweets):
        if c not in dict:
            dict[c] = {}
            dict[c]['reach'] = 0
            dict[c]['tweets'] = []
        dict[c]['tweets'].append(tweet['text'])
        dict[c]['reach'] = dict[c]['reach'] + tweet['author']['followers_count']
    for c in dict:
        dict[c]['retweetcount'] = len(dict[c]['tweets'])
        dict[c]['retweetability'] = float(dict[c]['retweetcount'] * 100000 / dict[c]['reach'])

    with open('output_tweets.json', 'w') as file:
        file.write(json.dumps(dict))

    resultList = []

    for cluster in dict.values():
        resultList.append(cluster)

    with open('results.txt', 'w') as file:
        file.write("The total number of clusters are %d \n\n\n" % len(dict))
        file.write(" ///////////////////////////////////////////////////////////////////////////////////////////////////////////// \n\n ")
        file.write("The ranking according to Total Outreach is \n\n\n")
        resultList = sorted(resultList, key=itemgetter('reach'), reverse=True)
        file.write(json.dumps(resultList))
        file.write(" ///////////////////////////////////////////////////////////////////////////////////////////////////////////// \n\n ")
        file.write("The ranking according to Number of Retweets is \n\n\n")
        resultList = sorted(resultList, key=itemgetter('retweetcount'), reverse=True)
        file.write(json.dumps(resultList))
        file.write(" ///////////////////////////////////////////////////////////////////////////////////////////////////////////// \n\n ")
        file.write("The ranking according to Number of Retweetability is \n\n\n")
        resultList = sorted(resultList, key=itemgetter('retweetability'), reverse=True)
        file.write(json.dumps(resultList))























if __name__ == '__main__':
        main()

