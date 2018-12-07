import pandas as pd
import numpy as np

import glob
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from sklearn.metrics.pairwise import cosine_similarity

import json
Reviews = []
with open("/Users/ashikshafi/Downloads/goodreads_books_poetry.json", 'r') as f:
    for line in f:
        Reviews.append(json.loads(line))

Reviews= pd.DataFrame(Reviews)

Reviewssmall=Reviews.sample(n=1000)


Reviews.columns.index
Reviews.shape
Reviews.head(2)
Reviews.tail(2)
Reviews.iloc[4, 5]


#Define a TF-IDF Vectorizer Object. Remove all english stop words such as 'the', 'a'
tfidf = TfidfVectorizer(stop_words='english')

#Replace NaN with an empty string
Reviewssmall['description'] = Reviewssmall['description'].fillna('')

#Construct the required TF-IDF matrix by fitting and transforming the data
tfidf_matrix = tfidf.fit_transform(Reviewssmall['description'])

#Output the shape of tfidf_matrix
tfidf_matrix.shape

cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)


def SimilarBooks(enterisbn):
    enterisbn=str(enterisbn)
    isbnindex= Reviewssmall[Reviewssmall["isbn"]==enterisbn].index.values.astype(int)[0]
    similarityindex = np.argsort(-cosine_sim[indexx]).tolist()[:10]
    print(isbnindex)
    print(similarityindex)
    print([Reviewssmall["isbn"].loc[similarityindex]])
    print([Reviewssmall["description"].iloc[isbnindex]])
    print([Reviewssmall["description"].iloc[similarityindex]])

SimilarBooks(enterisbn)
