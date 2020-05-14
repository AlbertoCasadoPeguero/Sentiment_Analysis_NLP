#Importing the libraries
import re
import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer

#Importing the dataset
dataset = pd.read_csv('Restaurant_Reviews.tsv',delimiter = '\t',quoting = 3)

#Cleaning the text
corpus = []
for i in range(0,1000):
    review = re.sub('[^a-zA-Z]',' ',dataset['Review'][i])
    review = review.lower()
    review = review.split()
    porterStemmer = PorterStemmer()
    review = [porterStemmer.stem(word) for word in review if not word in set(stopwords.words('english'))]
    review = ' '.join(review)
    corpus.append(review)

#Creating the bag of words model and my y
countVectorizer = CountVectorizer(max_features = 1500)
X = countVectorizer.fit_transform(corpus).toarray()
y = dataset.iloc[:,-1]

#Dataset split
X_train,X_test, y_train, y_test = train_test_split(X,y,test_size = 0.20)

#Model training
classifier = GaussianNB()
classifier.fit(X_train,y_train)

#Predictions
y_pred = classifier.predict(X_test)

#Making the confusion matrix
confusion_matrix = confusion_matrix(y_test,y_pred)

#Accuracy 60%
report = classification_report(y_test, y_pred)