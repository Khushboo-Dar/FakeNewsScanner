# Fake News Detection

## Overview
This project aims to classify news articles as either "Real" or "Fake" using machine learning techniques. The model is trained using a dataset containing labeled news articles, with "REAL" labeled as 0 and "FAKE" labeled as 1. The classification is performed using a Support Vector Machine (SVM) with a TF-IDF vectorizer for feature extraction.

## Dataset
- The dataset used is `fake_or_real_news.csv`.
- The `label` column indicates whether the news is real ("REAL") or fake ("FAKE").
- The `text` column contains the news content.
- The `label` column is transformed into a numerical format where:
  - `0` represents real news.
  - `1` represents fake news.

## Dependencies
The following Python libraries are required:
- `numpy`
- `pandas`
- `sklearn`

## Installation
Ensure that all dependencies are installed using:
```bash
pip install numpy pandas scikit-learn
```
Or using Conda:
```bash
conda install numpy pandas scikit-learn
```

## Implementation Steps
1. **Load Dataset:** Read the CSV file into a Pandas DataFrame.
2. **Preprocessing:** Convert labels into numerical format and drop the original label column.
3. **Train-Test Split:** Divide the dataset into training (80%) and testing (20%) sets.
4. **Feature Extraction:** Use `TfidfVectorizer` to convert text data into numerical features.
5. **Model Training:** Train a `LinearSVC` model using the extracted features.
6. **Model Evaluation:** Compute the accuracy of the model on the test dataset.
7. **Prediction on New Data:**
   - A sample news article is saved to a text file (`mytext.txt`).
   - The file is read, vectorized, and passed to the trained model for prediction.
   - The predicted label is compared with the actual label from the dataset.

## Usage
Run the script to train the model and evaluate its performance. To make a prediction on a new text file:
```python
with open("mytext.txt", "w", encoding="utf-8") as f:
    f.write(x_test.iloc[10])

with open("mytext.txt", "r", encoding="utf-8") as f:
    text = f.read()

vectorized_text = vectorizer.transform([text])
prediction = clf.predict(vectorized_text)
print("Predicted Label:", prediction[0])
```

## Output
- The script prints the model accuracy.
- It also predicts whether a given news article is real or fake.
- The actual label from the dataset is retrieved for comparison.


