import pandas as pd
import numpy as np
import os
from sklearn.model_selection import cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

def main():
    # File paths
    train_path = "project-3-nlp/dataset/training_data.csv"
    test_path = "project-3-nlp/dataset/testing_data.csv"
    output_path = "project-3-nlp/dataset/testing_predictions.csv"
    
    # Load training data
    # Training data is tab-separated, no headers. Column 0 is label, Column 1 is text.
    print("Loading training data...")
    train_df = pd.read_csv(train_path, sep="\t", header=None, names=["label", "text"])
    print(f"Loaded {len(train_df)} training samples.")
    
    # Clean text slightly (handle missing values if any)
    train_df["text"] = train_df["text"].fillna("")
    
    # Define features and labels
    X = train_df["text"]
    y = train_df["label"]
    
    # Build pipeline
    # We use TF-IDF vectorizer (unigrams + bigrams) and Logistic Regression
    print("Building model pipeline...")
    pipeline = make_pipeline(
        TfidfVectorizer(ngram_range=(1, 2), max_features=50000, stop_words='english'),
        LogisticRegression(C=2.0, max_iter=1000, random_state=42)
    )
    
    # Cross-validation score
    print("Evaluating model with 5-fold cross-validation...")
    scores = cross_val_score(pipeline, X, y, cv=5, scoring="accuracy")
    print(f"Cross-validation accuracies: {scores}")
    print(f"Mean CV Accuracy: {scores.mean():.4f} (+/- {scores.std() * 2:.4f})")
    
    # Fit on all training data
    print("Fitting model on all training data...")
    pipeline.fit(X, y)
    
    # Load testing data
    print("Loading testing data...")
    test_df = pd.read_csv(test_path, sep="\t", header=None, names=["label", "text"])
    test_df["text"] = test_df["text"].fillna("")
    
    # Predict labels
    print("Predicting labels for testing data...")
    predictions = pipeline.predict(test_df["text"])
    
    # Replace label column with predictions
    test_df["label"] = predictions
    
    # Save back to CSV in the original format (tab-separated, no header, no index)
    test_df.to_csv(output_path, sep="\t", header=False, index=False)
    print(f"Predictions saved to {output_path}")
    
    # Also save to testing_data.csv if requested, but let's keep testing_predictions.csv as the new file
    # Let's write a copy of testing_predictions.csv to testing_data_predictions.csv just to be safe
    test_df.to_csv("project-3-nlp/dataset/testing_data_predictions.csv", sep="\t", header=False, index=False)
    print("Estimated accuracy is around {:.2f}%".format(scores.mean() * 100))

if __name__ == "__main__":
    main()
