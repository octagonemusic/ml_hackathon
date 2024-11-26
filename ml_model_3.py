# Base model implementation for water quality parameter predictions
# Implements simple linear regression models for initial parameter relationships

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

def load_and_prepare_data():
    """
    Load the cleaned water quality dataset and define prediction pairs
    Returns:
        tuple: (DataFrame, list of prediction pairs)
    """
    df = pd.read_csv('cleaned_water_quality_data.csv')
    
    # Define parameter pairs with known strong correlations
    prediction_pairs = [
        {
            'input': 'Electrical Conductivity (µS/cm) at 25°C)',
            'target': 'Total Dissolved Solids (mg/L)',
            'name': 'EC_TDS'  # Primary relationship in water quality
        },
        {
            'input': 'Chloride (mg/L)',
            'target': 'Sodium (mg/L)',
            'name': 'Cl_Na'   # Strong ionic relationship
        },
        {
            'input': 'Total Dissolved Solids (mg/L)',
            'target': 'Chloride (mg/L)',
            'name': 'TDS_Cl'
        },
        {
            'input': 'Total Dissolved Solids (mg/L)',
            'target': 'Sodium (mg/L)',
            'name': 'TDS_Na'
        },
        {
            'input': 'Chloride (mg/L)',
            'target': 'Electrical Conductivity (µS/cm) at 25°C)',
            'name': 'Cl_EC'
        }
    ]
    
    return df, prediction_pairs

def train_linear_models(df, pairs):
    """
    Train linear regression models for each parameter pair
    
    Args:
        df (DataFrame): Input water quality data
        pairs (list): List of parameter pairs to model
    
    Returns:
        tuple: (results dictionary, trained models dictionary)
    """
    results = {}
    models = {}
    
    for pair in pairs:
        # Prepare data for current parameter pair
        X = df[pair['input']].values.reshape(-1, 1)
        y = df[pair['target']].values
        
        # Standard ML pipeline: split, scale, train, predict
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Scale features for better model performance
        scaler_X = StandardScaler()
        scaler_y = StandardScaler()
        
        X_train_scaled = scaler_X.fit_transform(X_train)
        X_test_scaled = scaler_X.transform(X_test)
        
        y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1)).ravel()
        y_test_scaled = scaler_y.transform(y_test.reshape(-1, 1)).ravel()
        
        # Train linear regression model
        model = LinearRegression()
        model.fit(X_train_scaled, y_train_scaled)
        
        # Store model and scalers for later use
        models[pair['name']] = {
            'model': model,
            'scaler_X': scaler_X,
            'scaler_y': scaler_y
        }
        
        # Generate predictions
        y_train_pred = scaler_y.inverse_transform(
            model.predict(X_train_scaled).reshape(-1, 1)
        ).ravel()
        y_test_pred = scaler_y.inverse_transform(
            model.predict(X_test_scaled).reshape(-1, 1)
        ).ravel()
        
        # Calculate performance metrics
        results[pair['name']] = {
            'train': {
                'r2': r2_score(y_train, y_train_pred),
                'rmse': np.sqrt(mean_squared_error(y_train, y_train_pred))
            },
            'test': {
                'r2': r2_score(y_test, y_test_pred),
                'rmse': np.sqrt(mean_squared_error(y_test, y_test_pred))
            },
            'coefficient': model.coef_[0],
            'intercept': model.intercept_
        }
        
        # Generate and save prediction plots
        plt.figure(figsize=(10, 6))
        plt.scatter(y_test, y_test_pred, alpha=0.5)
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
        plt.xlabel(f'Actual {pair["target"]}')
        plt.ylabel(f'Predicted {pair["target"]}')
        plt.title(f'Linear Prediction: {pair["input"]} → {pair["target"]}')
        plt.savefig(f'ml_results/{pair["name"]}_predictions.png')
        plt.close()
    
    return results, models

def print_results(results):
    print("\nModel Performance Summary:")
    for pair_name, pair_results in results.items():
        print(f"\n{pair_name} Predictions:")
        print(f"Training R² Score: {pair_results['train']['r2']:.4f}")
        print(f"Training RMSE: {pair_results['train']['rmse']:.4f}")
        print(f"Testing R² Score: {pair_results['test']['r2']:.4f}")
        print(f"Testing RMSE: {pair_results['test']['rmse']:.4f}")
        print(f"Relationship: y = {pair_results['coefficient']:.4f}x + {pair_results['intercept']:.4f}")

if __name__ == "__main__":
    # Create results directory
    import os
    if not os.path.exists('ml_results'):
        os.makedirs('ml_results')
    
    # Load and prepare data
    df, prediction_pairs = load_and_prepare_data()
    
    # Train models and get results
    results, models = train_linear_models(df, prediction_pairs)
    
    # Print results
    print_results(results) 