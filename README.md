# Email Classifier

A machine learning project that classifies emails as spam or legitimate using Naive Bayes and TF-IDF vectorization.

# Live App:- https://email-classifier-p5uv.onrender.com/

## Overview

This project trains a `MultinomialNB` classifier on email data to detect spam messages. It uses scikit-learn for model training and includes both a Python training script and Jupyter notebook for experimentation.

## Features

- Text preprocessing with regex and stopword removal
- TF-IDF vectorization with 3000 features
- Naive Bayes classification model
- Model persistence using pickle
- Prediction function for new emails
- Dataset Link:- https://www.kaggle.com/code/azzayahia/email-spam?select=spam.csv

## Project Structure

```
email_classifier/
├── data/
│   └── spam.csv
├── model/
│   ├── model.pkl
│   └── vectorizer.pkl
├── notebooks/
│   └── train.ipynb
├── train.py
├── app.py
├── requirements.txt
└── README.md
```

## RUN

```
streamlit run app.py
```

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Training

Run the training script:
```bash
python train.py
```

### Prediction

Load the saved model and vectorizer to classify new emails:
```python
import pickle

model = pickle.load(open("model/model.pkl", "rb"))
vectorizer = pickle.load(open("model/vectorizer.pkl", "rb"))

cleaned_text = vectorizer.transform([email_text])
prediction = model.predict(cleaned_text)
```

## Model Performance

The classifier achieves high accuracy on the test set with balanced precision and recall for both spam and legitimate emails.
Metric	      Value
Accuracy  ->  97.40%
Precision ->  100.00%
Recall	  ->  80.67%
F1 Score	->  89.30%

## Dependencies

- scikit-learn
- pandas
- numpy
- streamlit (optional, for UI)
