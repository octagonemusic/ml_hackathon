# Enhanced model for TDS to Sodium prediction using multiple features
# Implements polynomial regression with TDS, EC, and Chloride as predictors

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt

def improve_tds_sodium_prediction(df):
    """
    Enhanced TDS to Sodium prediction using multiple features and polynomial transformation
    
    Args:
        df (DataFrame): Input water quality dataset containing TDS, EC, and Chloride measurements
    
    Returns:
        tuple: (results dictionary, trained model, polynomial transformer, X scaler, y scaler)
    """
    # Prepare data with multiple features for better prediction accuracy
    # Using TDS, EC, and Chloride as predictors due to their strong relationships with Sodium
    X = df[[
        'Total Dissolved Solids (mg/L)',
        'Electrical Conductivity (µS/cm) at 25°C)',
        'Chloride (mg/L)'  # Adding Chloride due to strong Na-Cl relationship
    ]].values
    y = df['Sodium (mg/L)'].values
    
    # Split data into training (80%) and testing (20%) sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Initialize and fit scalers for feature normalization
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    
    # Transform features to standard normal distribution
    X_train_scaled = scaler_X.fit_transform(X_train)
    X_test_scaled = scaler_X.transform(X_test)
    
    # Transform target variable
    y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1)).ravel()
    y_test_scaled = scaler_y.transform(y_test.reshape(-1, 1)).ravel()
    
    # Generate polynomial features (degree=2) to capture non-linear relationships
    poly = PolynomialFeatures(degree=2, include_bias=False)
    X_train_poly = poly.fit_transform(X_train_scaled)
    X_test_poly = poly.transform(X_test_scaled)
    
    # Train linear regression model on polynomial features
    model = LinearRegression()
    model.fit(X_train_poly, y_train_scaled)
    
    # Generate predictions and inverse transform to original scale
    y_train_pred = scaler_y.inverse_transform(
        model.predict(X_train_poly).reshape(-1, 1)
    ).ravel()
    y_test_pred = scaler_y.inverse_transform(
        model.predict(X_test_poly).reshape(-1, 1)
    ).ravel()
    
    # Calculate performance metrics
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
    
    # Create and save prediction plot
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
    # Load cleaned water quality dataset
    df = pd.read_csv('cleaned_water_quality_data.csv')
    
    # Create results directory if it doesn't exist
    if not os.path.exists('ml_results'):
        os.makedirs('ml_results')
    
    # Train improved model and get results
    results, model, poly, scaler_X, scaler_y = improve_tds_sodium_prediction(df)
    
    # Print model performance metrics
    print("\nImproved TDS → Sodium Model Performance:")
    print(f"Training R² Score: {results['train']['r2']:.4f}")
    print(f"Training RMSE: {results['train']['rmse']:.4f}")
    print(f"Testing R² Score: {results['test']['r2']:.4f}")
    print(f"Testing RMSE: {results['test']['rmse']:.4f}")
    
    # Compare with baseline model performance
    print("\nImprovement over original model:")
    print(f"Original Test R²: 0.7798")
    print(f"New Test R²: {results['test']['r2']:.4f}")
    print(f"R² Improvement: {(results['test']['r2'] - 0.7798)*100:.2f}%")
