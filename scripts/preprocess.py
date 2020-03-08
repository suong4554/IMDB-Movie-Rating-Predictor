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
            result.append(stemmer.stem(i))
    print(result)
    return result
