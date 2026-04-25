# Movie_review_Classifier
Determining if the movie is positive or negative from the reviews
# Sentiment Analysis using Naive Bayes

## Overview
This project classifies movie reviews as **positive or negative** using **Multinomial Naive Bayes**.

---

## Dataset
- Source: NLTK `movie_reviews`
- 1000 positive reviews
- 1000 negative reviews

---

## Steps Followed

1. Load dataset from NLTK
2. Extract review text (tokens → strings)
3. Create labels:
   - 0 → Negative
   - 1 → Positive
4. Convert text to numerical form using **CountVectorizer (Bag of Words)**
5. Split data into train and test
6. Train model using **Multinomial Naive Bayes**
7. Predict on test data
8. Evaluate using accuracy and confusion matrix

---

## Key Concepts

- Natural Language Processing (NLP)
- Bag of Words (BoW)
- Vectorization
- Multinomial Naive Bayes
- Confusion Matrix

---

## Result

- Accuracy: ~79%
- Model performs reasonably well but has some misclassifications
