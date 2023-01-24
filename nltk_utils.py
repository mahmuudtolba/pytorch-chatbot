import numpy as np
import nltk

from nltk.stem.porter import PorterStemmer
stemmer = PorterStemmer()


def tokenize(sentence):
    """ 
    split sentence into array of words/tokens
    """
    return nltk.word_tokenize(sentence)

def stem(word):
    """
    returing the root of the given word 
    """
    return stemmer.stem(word.lower())


def bag_of_words(tokenized_sentence , words):
    """
    return bag of words array of each giving sentence
    """
    sentence_words = [stem(word) for word in tokenized_sentence]
    bag = np.zeros(len(words) , dtype = np.float32)
    for idx , w in enumerate(words):
        if w in sentence_words:
            bag[idx] = 1
    return bag