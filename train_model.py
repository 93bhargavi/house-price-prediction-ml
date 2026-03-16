import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import os

def load_data(filepath):
    """Load dataset using pandas."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Dataset not found at {filepath}")
    return pd.read_csv(filepath)

def preprocess_data(df):
    """Perform basic preprocessing and split features and target."""
    X = df[['house_size', 'num_bedrooms', 'location_score']]
    y = df['price']
    
    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    return X_train, X_test, y_train, y_test

def train_model(X_train, y_train):
    """Train a Linear Regression model."""
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model

def save_model(model, filename):
    """Save the trained model as model.pkl."""
    joblib.dump(model, filename)
    print(f"Model saved as {filename}")

def main():
    dataset_path = 'dataset.csv'
    model_path = 'model.pkl'
    
    print("Loading data...")
    df = load_data(dataset_path)
    
    print("Preprocessing data...")
    X_train, X_test, y_train, y_test = preprocess_data(df)
    
    print("Training model...")
    model = train_model(X_train, y_train)
    
    print("Saving model...")
    save_model(model, model_path)
    
    # Also save the test data for evaluation script
    test_data = pd.concat([X_test, y_test], axis=1)
    test_data.to_csv('test_data.csv', index=False)
    print("Test data saved as test_data.csv for evaluation.")

if __name__ == "__main__":
    main()
