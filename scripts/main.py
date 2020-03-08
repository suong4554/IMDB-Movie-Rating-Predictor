import sklearn
import nltk
import pandas as pd
import matplotlib.pyplot as plt
import os
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
import pickle

import preprocess as pp



#Polarizes data and gets accuracy
def getAccuracy(pred_y, test_y):
    #Polarize categorizes values to positive or negative
    pred_y = polarize(pred_y)
    #Converts series to list so that polarize function can work
    test_y = test_y.tolist()
    test_y = polarize(test_y)
    print(sklearn.metrics.accuracy_score(test_y, pred_y))



def visualize(test_y, pred_y, title):
    plt.scatter(test_y, pred_y)
    plt.xlabel("Rating: $Y_i$")
    plt.ylabel("Predicted Rating: $\hat{Y}_i$")
    plt.title(title)
    plt.show()


import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
import re
import string


def cleanTokens(input_str):
    #Removes numbers
    input_str = re.sub(r'\d+', '', input_str).lower().replace("<br /><br />", "")
    #Removes punctuation
    input_str = input_str.translate(str.maketrans('','', string.punctuation))
    stop_words = set(stopwords.words("english"))

    tokens = word_tokenize(input_str)
    #Removes stop words and gets word stem
    stemmer = SnowballStemmer("english")
    result = []
    for i in tokens:
        if i not in stop_words:
            #print(i, stemmer.stem(i))
            #result.append(stemmer.stem(i))
            result.append(i)
        #result.append(stemmer.stem(i))
    resultStr = " ".join(result)
    return resultStr



def polarize(predict):
    for x, data in enumerate(predict):
        #1 = positive review, 0 = negative review
        data = int(data)
        if data > 5:
            predict[x] = "1"
        else:
            predict[x] = "0"
    return predict

def getDirectory(dirName):
    tempWhole = []
    for fileName in os.listdir(dirName):
        temp = []
        fileNameS = fileName.split("_")
        movid = fileNameS[0]
        rating = fileNameS[1].replace(".txt", "")

        with open(dirName + "\\" + fileName, "r", encoding ="utf-8") as myfile:
            text = myfile.read().replace("<br />", "").replace(")", "").replace("(", "")

        text = cleanTokens(text)
        temp.append(movid)
        temp.append(rating)
        temp.append(text)

        tempWhole.append(temp)
    return tempWhole




def retrieveData(getData, home_dir):
    if getData:
        #####################Gets Data#####################
        negTrain = home_dir + "\\data\\aclImdb\\train\\neg"
        posTrain = home_dir + "\\data\\aclImdb\\train\\pos"
        data = []
        data = getDirectory(negTrain)
        data.extend(getDirectory(posTrain))

        with open(home_dir + "\\data\\extractedData.arr", "wb") as myfile:
            pickle.dump(data, myfile)
    else:
        with open(home_dir + "\\data\\extractedData.arr", "rb") as myfile:
            data = pickle.load(myfile)

    return data

def retrieveTestData(getData, home_dir):
    if getData:
        #####################Gets Data#####################
        negTrain = home_dir + "\\data\\aclImdb\\test\\neg"
        posTrain = home_dir + "\\data\\aclImdb\\test\\pos"
        data = []
        data = getDirectory(negTrain)
        data.extend(getDirectory(posTrain))

        with open(home_dir + "\\data\\extractedTestData.arr", "wb") as myfile:
            pickle.dump(data, myfile)
    else:
        with open(home_dir + "\\data\\extractedTestData.arr", "rb") as myfile:
            data = pickle.load(myfile)

    return data



getData = False
home_dir = os.path.dirname(os.path.realpath(__file__)).replace("scripts", "")

#gets data
data = retrieveData(getData, home_dir)
dataTest = retrieveTestData(getData, home_dir)
#Creates DataFrame:
movie_train = pd.DataFrame(data, columns=["id", "rating", "data"])
movie_test = pd.DataFrame(dataTest, columns=["id", "rating", "data"])

trainLength = len(data)
movie_whole = movie_train.data.append(movie_test.data)
movie_whole = movie_whole.reset_index(drop=True)
#print(movie_train.data)
#print(movie_whole)


# initialize movie_vector object, and then turn movie train data into a vector
#movie_vec = CountVectorizer(min_df=2, tokenizer=nltk.word_tokenize)# use all 25K words.
movie_vec = CountVectorizer(min_df=2, tokenizer=nltk.word_tokenize, max_features = 3000) # use top 3000 words only.
movie_counts = movie_vec.fit_transform(movie_whole)
#movie_counts = movie_train.data

# Convert raw frequency counts into TF-IDF values
tfidf_transformer = TfidfTransformer()
movie_tfidf = tfidf_transformer.fit_transform(movie_counts)

tfidf_train = movie_tfidf[:trainLength]
tfidf_test = movie_tfidf[trainLength:]

#Creates training data
train_y = movie_train.rating
train_x = tfidf_train
#Creates test data
test_y = movie_test.rating
test_x = tfidf_test



# initialize movie_vector object, and then turn movie train data into a vector
#movie_vec_test = CountVectorizer(min_df=2, tokenizer=nltk.word_tokenize)# use all 25K words.
movie_vec = CountVectorizer(min_df=2, tokenizer=nltk.word_tokenize, max_features = 3000) # use top 3000 words only.
movie_counts_test = movie_vec.fit_transform(movie_test.data)
#movie_counts = movie_train.data

# Convert raw frequency counts into TF-IDF values
tfidf_transformer = TfidfTransformer()
movie_tfidf_test = tfidf_transformer.fit_transform(movie_counts_test)





#Classifier Model
import models

#naive_bayes MultinomialNB
pred_y = models.nbMultiNB(train_x, train_y, test_x)
print("MultinomialNB")
getAccuracy(pred_y, test_y)
print("")

pred_y = models.logreg(train_x, train_y, test_x)
print("Logistic Regression")
getAccuracy(pred_y, test_y)
print("")

pred_y = models.sgd(train_x, train_y, test_x)
print("Stochastic Gradient Descent")
getAccuracy(pred_y, test_y)
print("")


pred_y = models.passiveAgressive(train_x, train_y, test_x)
print("Passive Agressive Classifier")
getAccuracy(pred_y, test_y)
print("")

pred_y = models.perceptron(train_x, train_y, test_x)
print("Linear Perceptron")
getAccuracy(pred_y, test_y)
print("")
