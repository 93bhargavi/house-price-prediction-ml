import pandas as pd
import numpy as np

def generate_dataset(num_samples=200):
    np.random.seed(42)
    
    # Features
    house_size = np.random.normal(1500, 500, num_samples).astype(int)
    num_bedrooms = np.random.randint(1, 6, num_samples)
    location_score = np.random.uniform(1, 10, num_samples).round(1)
    
    # target: Price (some logic: base 50k + 150/sqft + 20k/bedroom + 10k*location) + noise
    price = (50000 + (house_size * 150) + (num_bedrooms * 20000) + (location_score * 10000) + np.random.normal(0, 10000, num_samples)).astype(int)
    
    df = pd.DataFrame({
        'house_size': house_size,
        'num_bedrooms': num_bedrooms,
        'location_score': location_score,
        'price': price
    })
    
    # Save to CSV
    df.to_csv('C:\\Users\\abc\\.gemini\\antigravity\\scratch\\house-price-prediction-ml\\dataset.csv', index=False)
    print("Dataset generated successfully at dataset.csv")

if __name__ == "__main__":
    generate_dataset()
