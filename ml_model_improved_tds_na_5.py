import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import PolynomialFeatures
import matplotlib.pyplot as plt

def improve_tds_sodium_prediction(df):
    # Prepare data with multiple features
    X = df[[
        'Total Dissolved Solids (mg/L)',
        'Electrical Conductivity (µS/cm) at 25°C)',
        'Chloride (mg/L)'  # Adding Chloride due to strong Na-Cl relationship
    ]].values
    y = df['Sodium (mg/L)'].values
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Scale features
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    
    X_train_scaled = scaler_X.fit_transform(X_train)
    X_test_scaled = scaler_X.transform(X_test)
    
    y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1)).ravel()
    y_test_scaled = scaler_y.transform(y_test.reshape(-1, 1)).ravel()
    
    # Add polynomial features
    poly = PolynomialFeatures(degree=2, include_bias=False)
    X_train_poly = poly.fit_transform(X_train_scaled)
    X_test_poly = poly.transform(X_test_scaled)
    
    # Train model
    model = LinearRegression()
    model.fit(X_train_poly, y_train_scaled)
    
    # Make predictions
    y_train_pred = scaler_y.inverse_transform(
        model.predict(X_train_poly).reshape(-1, 1)
    ).ravel()
    y_test_pred = scaler_y.inverse_transform(
        model.predict(X_test_poly).reshape(-1, 1)
    ).ravel()
    
    # Calculate metrics
    results = {
        'train': {
            'r2': r2_score(y_train, y_train_pred),
            'rmse': np.sqrt(mean_squared_error(y_train, y_train_pred))
        },
        'test': {
            'r2': r2_score(y_test, y_test_pred),
            'rmse': np.sqrt(mean_squared_error(y_test, y_test_pred))
        }
    }
    
    # Plot results
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_test_pred, alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    plt.xlabel('Actual Sodium (mg/L)')
    plt.ylabel('Predicted Sodium (mg/L)')
    plt.title('Improved TDS → Sodium Prediction\nUsing TDS, EC, & Chloride with Polynomial Features')
    plt.savefig('ml_results/TDS_Na_improved_predictions.png')
    plt.close()
    
    return results, model, poly, scaler_X, scaler_y

if __name__ == "__main__":
    # Load data
    df = pd.read_csv('cleaned_water_quality_data.csv')
    
    # Create results directory if needed
    if not os.path.exists('ml_results'):
        os.makedirs('ml_results')
    
    # Train improved model
    results, model, poly, scaler_X, scaler_y = improve_tds_sodium_prediction(df)
    
    # Print results
    print("\nImproved TDS → Sodium Model Performance:")
    print(f"Training R² Score: {results['train']['r2']:.4f}")
    print(f"Training RMSE: {results['train']['rmse']:.4f}")
    print(f"Testing R² Score: {results['test']['r2']:.4f}")
    print(f"Testing RMSE: {results['test']['rmse']:.4f}")
    
    # Compare with original
    print("\nImprovement over original model:")
    print(f"Original Test R²: 0.7798")
    print(f"New Test R²: {results['test']['r2']:.4f}")
    print(f"R² Improvement: {(results['test']['r2'] - 0.7798)*100:.2f}%")
