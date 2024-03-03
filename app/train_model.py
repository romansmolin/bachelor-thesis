import time
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import nltk

#Data payload
data = pd.read_csv('data/normalized.csv', encoding='utf-8')

data.dropna(subset=['text'], inplace=True) #Remove empty row

texts = data['text']
categories = data['cat']

#Text vectorization
tfidf = TfidfVectorizer(stop_words='english')
X = tfidf.fit_transform(texts)
y = np.array(categories)

#Distribution for learning and test 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

#Moder learning
start = time.time()

#Naive Bayes
nb_classifier = MultinomialNB()
nb_classifier.fit(X_train, y_train)

# KNN
knn_classifier = KNeighborsClassifier()
knn_classifier.fit(X_train, y_train)

# Logistic Regression
logistic_classifier = LogisticRegression()
logistic_classifier.fit(X_train, y_train)

end = time.time()
print(f"Time to train: {end - start} seconds")

#Accuracy estimation
nb_accuracy = accuracy_score(y_test, nb_classifier.predict(X_test))
knn_accuracy = accuracy_score(y_test, knn_classifier.predict(X_test))
logistic_accuracy = accuracy_score(y_test, logistic_classifier.predict(X_test))

print(f"Naive Bayes Accuracy: {nb_accuracy}")
print(f"KNN Accuracy: {knn_accuracy}")
print(f"Logistic Regression Accuracy: {logistic_accuracy}")

pickle.dump(nb_classifier, open('models/nb_model.pickle', 'wb'))
pickle.dump(knn_classifier, open('models/knn_model.pickle', 'wb'))
pickle.dump(logistic_classifier, open('models/logistic_model.pickle', 'wb'))


# Test data
new_texts = [
    "Interested in a smartwatch that tracks fitness activities like running and swimming. It should have a heart rate monitor and be compatible with my Android phone.",
    "In search of summer sandals that are comfortable for walking long distances. They should have good arch support and be stylish enough to wear with dresses or shorts.",
    "Looking for a science fiction book with a strong female lead. Preferably set in space with a focus on exploration and alien cultures. Should have an engaging plot with lots of action.",
    "I need a new vacuum cleaner that's effective on pet hair and works well on hardwood floors. Ideally, it should be lightweight and easy to maneuver around furniture."
]

#Test data vectorization
new_X = tfidf.transform(new_texts)

# Predictions
nb_predictions = nb_classifier.predict(new_X)
knn_predictions = knn_classifier.predict(new_X)
logistic_predictions = logistic_classifier.predict(new_X)

print("Naive Bayes Predictions:", nb_predictions)
print("KNN Predictions:", knn_predictions)
print("Logistic Regression Predictions:", logistic_predictions)