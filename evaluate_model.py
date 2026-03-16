import pandas as pd
import joblib
from sklearn.metrics import mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import os

def load_model(filename):
    """Load the saved model."""
    if not os.path.exists(filename):
        raise FileNotFoundError(f"Model file not found at {filename}")
    return joblib.load(filename)

def evaluate_model(model, test_data_path):
    """Evaluate using metrics such as MAE and R² score."""
    if not os.path.exists(test_data_path):
        raise FileNotFoundError(f"Test data not found at {test_data_path}")
    
    df = pd.read_csv(test_data_path)
    X_test = df[['house_size', 'num_bedrooms', 'location_score']]
    y_test = df['price']
    
    # Make predictions
    predictions = model.predict(X_test)
    
    # Calculate metrics
    mae = mean_absolute_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)
    
    print("\n--- Evaluation Metrics ---")
    print(f"Mean Absolute Error (MAE): ${mae:,.2f}")
    print(f"R² Score: {r2:.4f}")
    
    return y_test, predictions

def show_results(y_test, predictions):
    """Show prediction results."""
    print("\n--- Prediction Samples ---")
    results = pd.DataFrame({'Actual': y_test, 'Predicted': predictions})
    print(results.head(10))
    
    # Visualize
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, predictions, alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.xlabel('Actual Prices')
    plt.ylabel('Predicted Prices')
    plt.title('Actual vs Predicted House Prices')
    plt.grid(True)
    plt.savefig('prediction_plot.png')
    print("\nVisualization saved as prediction_plot.png")

def main():
    model_path = 'model.pkl'
    test_data_path = 'test_data.csv'
    
    try:
        print("Loading model...")
        model = load_model(model_path)
        
        print("Evaluating model...")
        y_test, predictions = evaluate_model(model, test_data_path)
        
        print("Showing results...")
        show_results(y_test, predictions)
        
    except Exception as e:
        print(f"Error during evaluation: {e}")

if __name__ == "__main__":
    main()
