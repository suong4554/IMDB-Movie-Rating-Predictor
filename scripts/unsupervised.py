import os
import pandas as pd
import numpy as np
import pickle
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split




def getDirectory(dirName):
    tempWhole = []
    for fileName in os.listdir(dirName):
        temp = []
        fileNameS = fileName.split("_")
        movid = fileNameS[0]
        rating = fileNameS[1].replace(".txt", "")

        with open(dirName + "\\" + fileName, "r", encoding ="utf-8") as myfile:
            text = myfile.read().replace("<br />", "").replace(")", "").replace("(", "")

        #text = cleanTokens(text)
        temp.append(movid)
        temp.append(rating)
        temp.append(text)

        tempWhole.append(temp)
    return tempWhole



def retrieveData(getData, home_dir):
    if getData:
        #####################Gets Data#####################
        unsup = home_dir + "data\\aclImdb\\train\\unsup"
        data = []
        data = getDirectory(unsup)


        with open(home_dir + "\\data\\extractedUnsupData.arr", "wb") as myfile:
            pickle.dump(data, myfile)
    else:
        with open(home_dir + "\\data\\extractedUnsupData.arr", "rb") as myfile:
            data = pickle.load(myfile)

    return data

getData = False
home_dir = os.path.dirname(os.path.realpath(__file__)).replace("scripts", "")

#gets data
data = retrieveData(getData, home_dir)

#Creates DataFrame:
movie_train = pd.DataFrame(data, columns=["id", "rating", "data"])






# initialize movie_vector object, and then turn movie train data into a vector
vectorizer = TfidfVectorizer(stop_words="english")
movie_tfidf = vectorizer.fit_transform(movie_train.data)


#Creates test/training data
train_y = movie_train.rating
train_x = movie_tfidf
#train_x, test_x, train_y, test_y = train_test_split(movie_tfidf, train_y, test_size=0.3, random_state=5)


from sklearn.cluster import KMeans
#Number of Clusters
true_k = 20
model = KMeans(n_clusters=true_k, init='k-means++', max_iter=1000, n_init=1)
model.fit(train_x)

order_centroids = model.cluster_centers_.argsort()[:, ::-1]
terms = vectorizer.get_feature_names()
for i in range(true_k):
    print("Cluster " + str(i)),
    for x in order_centroids[i, :10]:
        print(terms[x])
    print("")


"""
from sklearn.cluster import FeatureAgglomeration
true_k = 2
model = FeatureAgglomeration(n_clusters=true_k)
train_x.toarray(dtype=object)
model.fit(train_x)

order_centroids = model.cluster_centers_.argsort()[:, ::-1]
terms = vectorizer.get_feature_names()
for i in range(true_k):
    print("Cluster " + str(i)),
    for x in order_centroids[i, :10]:
        print(terms[x])
    print("")
"""
