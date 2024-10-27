# Sentiment Analysis of COVID-19 Tweets Using Machine Learning Classifiers

## Project Overview
This project performs sentiment analysis on COVID-19-related tweets using machine learning classifiers, specifically Random Forest, Support Vector Machine (SVM), and Naive Bayes. The model classifies tweet sentiments into categories such as negative, neutral, positive, extremely negative, and extremely positive.

## Dataset
The dataset contains COVID-19 tweets stored in `Coronavirus Tweets.csv`. It includes tweets in multiple sentiments, with pre-labeled sentiment classifications, enabling model training and validation.

## Project Structure
- **Data Preprocessing**: Clean and preprocess tweet text data using regular expressions and NLP techniques.
- **Feature Extraction**: Convert text data to numerical format using TF-IDF vectorization.
- **Model Training**: Train and evaluate multiple classifiers (Random Forest, SVM, Naive Bayes).
- **Performance Metrics**: Measure accuracy for each classifier and plot confusion matrices.

## Code Explanation
1. **Data Loading**: Import the dataset and inspect initial rows.
2. **Text Preprocessing**: 
   - Remove special characters, single characters, prefixed characters, and extra spaces.
   - Convert text to lowercase.
3. **Vectorization**: Use TF-IDF to transform tweets into a numerical format for machine learning models.
4. **Classifier Training**:
   - **Random Forest**: Used with 200 estimators.
   - **SVM**: Linear kernel classifier for tweet classification.
   - **Naive Bayes**: A multinomial classifier.
5. **Evaluation**: Generate predictions and measure accuracy using `accuracy_score`.
6. **Confusion Matrix Visualization**: Plot confusion matrices for each classifier to visualize true vs. predicted values.

## Requirements
- Python
- Libraries: `numpy`, `pandas`, `re`, `nltk`, `sklearn`, `matplotlib`, `cv2`

## Instructions
1. Install dependencies using `pip install -r requirements.txt`.
2. Run the main script to train models and view results:
   ```bash
   python sentiment_analysis.py
   ```
3. Visualize classifier performance with accuracy scores and confusion matrix plots.

## Results
The project compares accuracy across classifiers, showing the efficiency of various machine learning models in sentiment classification for social media text data.
