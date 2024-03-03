import nltk
nltk.download('punkt')
import pandas as pd

from nltk.stem.snowball import SnowballStemmer
from nltk.tokenize import word_tokenize

def stemSentence(sentence, language="english"):
    print(sentence)
    print("stemSentence started...")
    
    if pd.isna(sentence):
        print("Skipping NaN value")
        return ""
    
    stemmer = SnowballStemmer(language)
    token_words = word_tokenize(sentence)
    stem_sentence = []
    for word in token_words:
        if len(word) > 3 and not word.isdigit():
            stem_sentence.append(stemmer.stem(word))
            stem_sentence.append(" ")
    return "".join(stem_sentence).strip()
