import pandas as pd
import numpy as np
from sklearn.metrics import (
    r2_score, mean_squared_error, mean_absolute_error, 
    explained_variance_score, max_error, mean_absolute_percentage_error
)
import joblib
from tabulate import tabulate
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

def load_models_and_data():
    """Load all trained models and the test dataset"""
    # Load models
    models = {
        'EC→TDS': {
            'model': joblib.load('models/ec_tds_model.pkl'),
            'scaler_X': joblib.load('models/ec_tds_scaler_X.pkl'),
            'scaler_y': joblib.load('models/ec_tds_scaler_y.pkl')
        },
        'TDS→Chloride': {
            'model': joblib.load('models/tds_chloride_model.pkl'),
            'scaler_X': joblib.load('models/tds_chloride_scaler_X.pkl'),
            'scaler_y': joblib.load('models/tds_chloride_scaler_y.pkl'),
            'poly': joblib.load('models/tds_chloride_poly.pkl')
        },
        'TDS→Sodium': {
            'model': joblib.load('models/tds_sodium_model.pkl'),
            'scaler_X': joblib.load('models/tds_sodium_scaler_X.pkl'),
            'scaler_y': joblib.load('models/tds_sodium_scaler_y.pkl'),
            'poly': joblib.load('models/tds_sodium_poly.pkl')
        },
        'Chloride→EC': {
            'model': joblib.load('models/chloride_ec_model.pkl'),
            'scaler_X': joblib.load('models/chloride_ec_scaler_X.pkl'),
            'scaler_y': joblib.load('models/chloride_ec_scaler_y.pkl'),
            'poly': joblib.load('models/chloride_ec_poly.pkl')
        }
    }
    
    # Load test data
    df = pd.read_csv('cleaned_water_quality_data.csv')
    return models, df

def calculate_regression_metrics(y_true, y_pred):
    """Calculate comprehensive regression metrics"""
    metrics = {
        'R² Score': r2_score(y_true, y_pred),
        'Adjusted R²': 1 - (1-r2_score(y_true, y_pred))*(len(y_true)-1)/(len(y_true)-1-1),
        'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
        'MAE': mean_absolute_error(y_true, y_pred),
        'MAPE': mean_absolute_percentage_error(y_true, y_pred),
        'Max Error': max_error(y_true, y_pred),
        'Explained Variance': explained_variance_score(y_true, y_pred)
    }
    
    # Calculate additional statistical metrics
    residuals = y_true - y_pred
    metrics.update({
        'Residual Standard Error': np.std(residuals),
        'Mean Bias Error': np.mean(residuals),
        'Residual Skewness': stats.skew(residuals),
        'Residual Kurtosis': stats.kurtosis(residuals)
    })
    
    return metrics

def evaluate_model_performance(models, df):
    """Evaluate performance for all models"""
    results = {}
    
    # EC → TDS
    X_ec = df['Electrical Conductivity (µS/cm) at 25°C)'].values.reshape(-1, 1)
    y_tds = df['Total Dissolved Solids (mg/L)'].values
    X_ec_scaled = models['EC→TDS']['scaler_X'].transform(X_ec)
    y_tds_pred = models['EC→TDS']['scaler_y'].inverse_transform(
        models['EC→TDS']['model'].predict(X_ec_scaled).reshape(-1, 1)
    ).ravel()
    results['EC→TDS'] = calculate_regression_metrics(y_tds, y_tds_pred)
    
    # TDS → Chloride
    X_tds_cl = df[['Total Dissolved Solids (mg/L)', 'Electrical Conductivity (µS/cm) at 25°C)']].values
    y_cl = df['Chloride (mg/L)'].values
    X_tds_cl_scaled = models['TDS→Chloride']['scaler_X'].transform(X_tds_cl)
    X_tds_cl_poly = models['TDS→Chloride']['poly'].transform(X_tds_cl_scaled)
    y_cl_pred = models['TDS→Chloride']['scaler_y'].inverse_transform(
        models['TDS→Chloride']['model'].predict(X_tds_cl_poly).reshape(-1, 1)
    ).ravel()
    results['TDS→Chloride'] = calculate_regression_metrics(y_cl, y_cl_pred)
    
    # TDS → Sodium
    X_tds_na = df[['Total Dissolved Solids (mg/L)', 'Electrical Conductivity (µS/cm) at 25°C)', 'Chloride (mg/L)']].values
    y_na = df['Sodium (mg/L)'].values
    X_tds_na_scaled = models['TDS→Sodium']['scaler_X'].transform(X_tds_na)
    X_tds_na_poly = models['TDS→Sodium']['poly'].transform(X_tds_na_scaled)
    y_na_pred = models['TDS→Sodium']['scaler_y'].inverse_transform(
        models['TDS→Sodium']['model'].predict(X_tds_na_poly).reshape(-1, 1)
    ).ravel()
    results['TDS→Sodium'] = calculate_regression_metrics(y_na, y_na_pred)
    
    # Chloride → EC
    X_cl_ec = df[['Chloride (mg/L)', 'Total Dissolved Solids (mg/L)', 'Sodium (mg/L)', 'pH']].values
    y_ec = df['Electrical Conductivity (µS/cm) at 25°C)'].values
    X_cl_ec_scaled = models['Chloride→EC']['scaler_X'].transform(X_cl_ec)
    X_cl_ec_poly = models['Chloride→EC']['poly'].transform(X_cl_ec_scaled)
    y_ec_pred = models['Chloride→EC']['scaler_y'].inverse_transform(
        models['Chloride→EC']['model'].predict(X_cl_ec_poly).reshape(-1, 1)
    ).ravel()
    results['Chloride→EC'] = calculate_regression_metrics(y_ec, y_ec_pred)
    
    return results

def generate_latex_table(results):
    """Generate LaTeX formatted table"""
    metrics = list(next(iter(results.values())).keys())
    models = list(results.keys())
    
    latex_table = "\\begin{table}[h]\n\\centering\n\\begin{tabular}{l" + "r"*len(models) + "}\n"
    latex_table += "\\hline\nMetric & " + " & ".join(models) + " \\\\\n\\hline\n"
    
    for metric in metrics:
        row = f"{metric} & " + " & ".join([f"{results[model][metric]:.4f}" for model in models]) + " \\\\\n"
        latex_table += row
    
    latex_table += "\\hline\n\\end{tabular}\n"
    latex_table += "\\caption{Comprehensive Regression Metrics for Water Quality Prediction Models}\n"
    latex_table += "\\label{tab:regression_metrics}\n\\end{table}"
    
    return latex_table

def plot_residual_analysis(models, df):
    """Generate residual analysis plots"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 15))
    fig.suptitle('Residual Analysis for Water Quality Prediction Models', fontsize=16)
    
    # Plot settings
    plot_positions = {
        'EC→TDS': (0, 0),
        'TDS→Chloride': (0, 1),
        'TDS→Sodium': (1, 0),
        'Chloride→EC': (1, 1)
    }
    
    for model_name, (i, j) in plot_positions.items():
        ax = axes[i, j]
        
        # Get actual and predicted values
        if model_name == 'EC→TDS':
            X = df['Electrical Conductivity (µS/cm) at 25°C)'].values.reshape(-1, 1)
            y_true = df['Total Dissolved Solids (mg/L)'].values
            X_scaled = models[model_name]['scaler_X'].transform(X)
            y_pred = models[model_name]['scaler_y'].inverse_transform(
                models[model_name]['model'].predict(X_scaled).reshape(-1, 1)
            ).ravel()
        else:
            # Handle polynomial models similarly...
            continue
        
        residuals = y_true - y_pred
        
        # Create residual plot
        ax.scatter(y_pred, residuals, alpha=0.5)
        ax.axhline(y=0, color='r', linestyle='--')
        ax.set_xlabel('Predicted Values')
        ax.set_ylabel('Residuals')
        ax.set_title(f'{model_name} Residual Plot')
        
        # Add trend line
        z = np.polyfit(y_pred, residuals, 1)
        p = np.poly1d(z)
        ax.plot(y_pred, p(y_pred), "r--", alpha=0.8)
    
    plt.tight_layout()
    plt.savefig('model_evaluation_results/residual_analysis.png')
    plt.close()

def main():
    # Create results directory
    import os
    if not os.path.exists('model_evaluation_results'):
        os.makedirs('model_evaluation_results')
    
    # Load models and data
    print("Loading models and data...")
    models, df = load_models_and_data()
    
    # Evaluate models
    print("Evaluating model performance...")
    results = evaluate_model_performance(models, df)
    
    # Generate tables
    print("Generating result tables...")
    
    # Save as markdown
    with open('model_evaluation_results/metrics_table.md', 'w') as f:
        f.write(tabulate(
            [[metric] + [results[model][metric] for model in results.keys()] 
             for metric in next(iter(results.values())).keys()],
            headers=['Metric'] + list(results.keys()),
            tablefmt='pipe',
            floatfmt='.4f'
        ))
    
    # Save as LaTeX
    with open('model_evaluation_results/metrics_table.tex', 'w') as f:
        f.write(generate_latex_table(results))
    
    # Save as CSV
    pd.DataFrame(results).to_csv('model_evaluation_results/metrics.csv')
    
    # Generate residual analysis plots
    print("Generating residual analysis plots...")
    plot_residual_analysis(models, df)
    
    print("Evaluation complete! Results saved in 'model_evaluation_results' directory")

if __name__ == "__main__":
    main() 