from rake_nltk import Rake
import nltk
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from sklearn import preprocessing

def getKeyWords(text):
    r = Rake(min_length=1,
    max_length=2,
    language="english",
    punctuations="´<>,.?':;{}[]-_=+`~/|\\")
    r.extract_keywords_from_text(text)

    temp = r.get_ranked_phrases()

    if len(temp) > 2:
        temp = temp[:3]
    elif len(temp) > 1:
        temp = temp[:2]
        temp.append(" ")
    elif len(temp) > 0:
        temp = temp[:1]
        temp.append(" ")
        temp.append(" ")
    
    if "<" in temp or None in temp:
        print(temp)
    return temp
    

def getDirectory(dirName):
    tempWhole = []
    columns = [
    "ID",
    "rating",
    "keyword1",
    "keyword2",
    "keyword3"
    ]
    for fileName in os.listdir(dirName):
        fileNameS = fileName.split("_")
        movid = fileNameS[0]
        rating = fileNameS[1].replace(".txt", "")
        
        with open(dirName + "\\" + fileName, "r", encoding ="utf-8") as myfile:
            text = myfile.read().replace("< br />", "").replace(")", "").replace("(", "")
            
            
        keyWords = getKeyWords(text)
        keyWords.insert(0, rating)
        keyWords.insert(0, movid)
        
        #print(keyWords)

        
        tempWhole.append(keyWords)
        #movieDF = pd.DataFrame(keyWords, index = columns)
        #mDF.append(movieDF, ignore_index = True)
    return tempWhole

        
def train(trainDF):

    targetY = trainDF["rating"]
    trainX = trainDF.drop("rating",  axis = 1).drop("ID",  axis = 1)
    print(trainX)
    le = preprocessing.LabelEncoder()
    for column in trainX:
        le.fit(trainX[column])
        trainX[column] = le.transform(trainX[column])
    #print("training X: ", trainX)
    #print("training Y: ", targetY)
    #print("Train DF: ", trainDF)
    

        
    lm = LinearRegression()
    lm.fit(trainX, targetY)
    return lm
        
def main():
    columns = [
        "ID",
        "rating",
        "keyword1",
        "keyword2",
        "keyword3"
    ]

    #mDF = pd.DataFrame(columns = columns)
    home_dir = os.path.dirname(os.path.realpath(__file__)).replace("scripts", "")
    negTrain = home_dir + "\\data\\aclImdb\\train\\neg"
    posTrain = home_dir + "\\data\\aclImdb\\train\\pos"
    data = []
    data = getDirectory(negTrain)
    data.extend(getDirectory(posTrain))
    mDF = pd.DataFrame(data)
    #print(mDF)
    #mDF = mDF.transpose()
    mDF.columns = columns
    
    #print("mDF: ", mDF)
    lm = train(mDF)
    
    
    #tDF = pd.DataFrame(columns = columns)
    negTest = home_dir + "\\data\\aclImdb\\test\\neg"
    posTest = home_dir + "\\data\\aclImdb\\test\\pos"
    data = getDirectory(negTest)
    data.extend(getDirectory(posTest))
    
    tDF = pd.DataFrame(data)
    #mDF = mDF.transpose()
    tDF.columns = columns
    le = preprocessing.LabelEncoder()
    for column in tDF:
        le.fit(tDF[column])
        tDF[column] = le.transform(tDF[column])
    Y_actual = tDF["rating"]
    X_test = tDF.drop("rating",  axis = 1).drop("ID",  axis = 1)
    

    Y_pred = lm.predict(X_test)
    #print(mDF)
    plt.scatter(Y_actual, Y_pred)
    plt.show()
    print("finished")
main()   
        
        