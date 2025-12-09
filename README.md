# CodeAlpha_Sentiment-Analysis
Sentiment Analysis on Twitter Sentiment140 Dataset
CodeAlpha – Data Analytics Internship (Task 4)

This project applies advanced Sentiment Analysis techniques to the Sentiment140 Twitter Dataset, consisting of 1.6M tweets labeled as positive or negative. The objective is to analyze real-world social media text, apply sentiment classification techniques, and extract patterns in user emotions using both rule-based NLP methods and machine learning models.

📂 Dataset

Sentiment140 Twitter Dataset
🔗 https://www.kaggle.com/datasets/kazanova/sentiment140

1,600,000 tweets

Labels:

0 → Negative

4 → Positive

🎯 Objectives

Clean and preprocess large-scale tweet text

Apply VADER & TextBlob sentiment scoring

Build a machine learning model using TF-IDF + Logistic Regression

Evaluate model performance with accuracy, classification report, and confusion matrix

Visualize sentiment patterns and common words using word clouds

Compare dataset labels with predicted sentiment

🧹 Data Cleaning

Several text preprocessing steps were applied:

Lowercasing

Removing URLs, mentions, hashtags

Removing numbers & punctuation

Tokenization

Stopword removal

Normalizing whitespace

Resulting cleaned tweets are suitable for both lexicon-based and ML-based sentiment models.

🧪 Techniques Used
1️⃣ Rule-Based Sentiment (Baseline)

VADER Compound Score

TextBlob Polarity Score

Label mapping (Positive / Negative / Neutral)

2️⃣ Machine Learning Model

TF-IDF Vectorization (1–2 grams, 50,000 features)

Logistic Regression Classifier

3️⃣ Evaluation Metrics

Accuracy

Precision, Recall, F1

Confusion Matrix

Top positive/negative features

Error analysis (FP/FN samples)

📊 Key Visualizations

Original label distribution

VADER sentiment distribution

Confusion Matrix

Word clouds for predicted Positive & Negative tweets

Top n-grams influencing predictions

🔑 Key Insights

The dataset contains slightly more negative tweets than positive.

VADER detects an additional Neutral class not included in the dataset labels.

Common positive words include expressions of excitement, joy, and appreciation.

Negative tweets often express frustration, disappointment, or complaints.

TF-IDF + Logistic Regression achieves strong performance on the test set, showing effective separation between positive and negative sentiment.

🛠 Tech Stack

Python

Pandas, NumPy

Matplotlib, Seaborn, WordCloud

Scikit-learn

VADER Sentiment Analyzer

TextBlob

Google Colab

📁 Project Notebook

sentiment140_task4_full.ipynb — Complete analysis, model training, visualizations & outputs.

👤 Author

Sanket Shakya
AI & Data Science | Data Analytics | Machine Learning
