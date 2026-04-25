import pandas as pd
import numpy as np
import nltk
import os
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import movie_reviews

nltk.download('movie_reviews')

# Check corpora
print(os.listdir(nltk.data.find("corpora")))
print(movie_reviews.categories())

# Get file ids
neg_rev = movie_reviews.fileids('neg')
pos_rev = movie_reviews.fileids('pos')

print(len(neg_rev))
print(len(pos_rev))

# Convert reviews to strings
rev_list = []

# Negative reviews
for rev in neg_rev:
    rev_text = movie_reviews.words(rev)
    review_one_string = " ".join(rev_text)
    rev_list.append(review_one_string)

# Positive reviews
for rev in pos_rev:
    rev_text = movie_reviews.words(rev)
    review_one_string = " ".join(rev_text)
    rev_list.append(review_one_string)

print(len(rev_list))  # should be 2000

# Create target labels
neg_targets = np.zeros((1000,), dtype=int)
pos_targets = np.ones((1000,), dtype=int)

target_list = []

for val in neg_targets:
    target_list.append(val)

for val in pos_targets:
    target_list.append(val)

print(len(target_list))  # should be 2000

# Convert to pandas Series
y = pd.Series(target_list)

# Vectorization
count_vect = CountVectorizer(lowercase=True, stop_words='english', min_df=2)
x_count_vect = count_vect.fit_transform(rev_list)

print(x_count_vect.shape)

# Feature names
x_names = count_vect.get_feature_names_out()
print(x_names)
print(x_names.shape)
x_count_vect=pd.DataFrame(x_count_vect.toarray(),columns=x_names)
print(x_count_vect.head())
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import confusion_matrix
x_train,x_test,y_train,y_test=train_test_split(x_count_vect,y,test_size=0.5,random_state=5)
from sklearn.naive_bayes import MultinomialNB
clf=MultinomialNB()
clf.fit(x_train,y_train)
y_pred_cv=clf.predict(x_test)
print(metrics.accuracy_score(y_test,y_pred_cv))
score_clf_cv=confusion_matrix(y_test,y_pred_cv)
print(score_clf_cv)



