# House Price Prediction ML

## Project Overview
This project implements a simple machine learning pipeline to predict house prices based on features such as house size, number of bedrooms, and location score. It uses a Linear Regression model from the scikit-learn library.

## Dataset Explanation
The dataset is synthetically generated and consists of the following features:
- `house_size`: Square footage of the house.
- `num_bedrooms`: Number of bedrooms.
- `location_score`: A score from 1 to 10 representing the desirability of the location.
- `price`: The target variable, representing the house price in dollars.

## Model Explanation
The project uses a **Linear Regression** model, which is a fundamental regression algorithm that models the relationship between the features and the target variable by fitting a linear equation to observed data.

## How to Run the Project
1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
2. **Train the Model**:
   ```bash
   python train_model.py
   ```
   This will load `dataset.csv`, train the model, and save it as `model.pkl`.
3. **Evaluate the Model**:
   ```bash
   python evaluate_model.py
   ```
   This will load `model.pkl`, evaluate its performance on test data, and generate a visualization.

## Example Output
Running `evaluate_model.py` produces metrics like:
- **Mean Absolute Error (MAE)**: ~\$10,000.00
- **R² Score**: ~0.98

It also generates a scatter plot `prediction_plot.png` comparing actual vs. predicted prices.
