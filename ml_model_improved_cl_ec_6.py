# Enhanced model for Chloride to Electrical Conductivity prediction
# Implements polynomial regression with multiple water quality parameters
# Uses advanced feature engineering to capture complex relationships

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt

def improve_chloride_ec_prediction(df):
    """
    Enhanced Chloride to EC prediction using multiple features and polynomial transformation
    
    Args:
        df (DataFrame): Input water quality dataset with multiple parameters
    
    Returns:
        tuple: (results dictionary, trained model, polynomial transformer, X scaler, y scaler)
    """
    # Prepare data with multiple features for comprehensive EC prediction
    X = df[[
        'Chloride (mg/L)',                # Primary predictor
        'Total Dissolved Solids (mg/L)',  # TDS strongly related to EC
        'Sodium (mg/L)',                  # Na-Cl relationship affects conductivity
        'pH'                              # pH can influence ionic conductivity
    ]].values
    y = df['Electrical Conductivity (µS/cm) at 25°C)'].values
    
    # Split data into training (80%) and testing (20%) sets with fixed random state
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Initialize scalers for feature normalization
    scaler_X = StandardScaler()  # Standardize features
    scaler_y = StandardScaler()  # Standardize target variable
    
    # Transform features to standard normal distribution
    X_train_scaled = scaler_X.fit_transform(X_train)
    X_test_scaled = scaler_X.transform(X_test)
    
    # Transform target variable (EC values)
    y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1)).ravel()
    y_test_scaled = scaler_y.transform(y_test.reshape(-1, 1)).ravel()
    
    # Generate polynomial features to capture non-linear relationships
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
    
    # Calculate comprehensive performance metrics
    results = {
        'train': {
            'r2': r2_score(y_train, y_train_pred),    # Coefficient of determination
            'rmse': np.sqrt(mean_squared_error(y_train, y_train_pred))  # Root mean squared error
        },
        'test': {
            'r2': r2_score(y_test, y_test_pred),      # Test set R² score
            'rmse': np.sqrt(mean_squared_error(y_test, y_test_pred))    # Test set RMSE
        }
    }
    
    # Create and save prediction visualization plot
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_test_pred, alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    plt.xlabel('Actual EC (µS/cm)')
    plt.ylabel('Predicted EC (µS/cm)')
    plt.title('Improved Chloride → EC Prediction\nUsing Multiple Parameters with Polynomial Features')
    plt.savefig('ml_results/Cl_EC_improved_predictions.png')
    plt.close()
    
    return results, model, poly, scaler_X, scaler_y

if __name__ == "__main__":
    # Load cleaned water quality dataset
    df = pd.read_csv('cleaned_water_quality_data.csv')
    
    # Ensure results directory exists
    if not os.path.exists('ml_results'):
        os.makedirs('ml_results')
    
    # Train improved model and get evaluation results
    results, model, poly, scaler_X, scaler_y = improve_chloride_ec_prediction(df)
    
    # Print detailed model performance metrics
    print("\nImproved Chloride → EC Model Performance:")
    print(f"Training R² Score: {results['train']['r2']:.4f}")
    print(f"Training RMSE: {results['train']['rmse']:.4f}")
    print(f"Testing R² Score: {results['test']['r2']:.4f}")
    print(f"Testing RMSE: {results['test']['rmse']:.4f}")
    
    # Compare with baseline model performance
    print("\nImprovement over original model:")
    print(f"Original Test R²: 0.7856")
    print(f"New Test R²: {results['test']['r2']:.4f}")
    print(f"R² Improvement: {(results['test']['r2'] - 0.7856)*100:.2f}%")
