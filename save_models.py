import joblib
import os
from ml_model_improved_tds_cl_4 import improve_tds_chloride_prediction
from ml_model_improved_tds_na_5 import improve_tds_sodium_prediction
from ml_model_improved_cl_ec_6 import improve_chloride_ec_prediction
from ml_model_3 import train_linear_models, load_and_prepare_data
import pandas as pd

def save_all_models():
    # Create models directory if it doesn't exist
    if not os.path.exists('models'):
        os.makedirs('models')
    
    # Load data
    print("Loading data...")
    df = pd.read_csv('cleaned_water_quality_data.csv')
    
    # Train and save linear models (EC→TDS and Cl→Na)
    print("\nTraining and saving linear models...")
    df_linear, prediction_pairs = load_and_prepare_data()
    results_linear, models_linear = train_linear_models(df_linear, prediction_pairs)
    
    # Save EC→TDS model
    joblib.dump(models_linear['EC_TDS']['model'], 'models/ec_tds_model.pkl')
    joblib.dump(models_linear['EC_TDS']['scaler_X'], 'models/ec_tds_scaler_X.pkl')
    joblib.dump(models_linear['EC_TDS']['scaler_y'], 'models/ec_tds_scaler_y.pkl')
    
    # Save Cl→Na model
    joblib.dump(models_linear['Cl_Na']['model'], 'models/cl_na_model.pkl')
    joblib.dump(models_linear['Cl_Na']['scaler_X'], 'models/cl_na_scaler_X.pkl')
    joblib.dump(models_linear['Cl_Na']['scaler_y'], 'models/cl_na_scaler_y.pkl')
    
    # Train and save improved models
    print("\nTraining and saving TDS → Chloride model...")
    results_tds_cl, model_tds_cl, poly_tds_cl, scaler_X_tds_cl, scaler_y_tds_cl = improve_tds_chloride_prediction(df)
    
    joblib.dump(model_tds_cl, 'models/tds_chloride_model.pkl')
    joblib.dump(scaler_X_tds_cl, 'models/tds_chloride_scaler_X.pkl')
    joblib.dump(scaler_y_tds_cl, 'models/tds_chloride_scaler_y.pkl')
    joblib.dump(poly_tds_cl, 'models/tds_chloride_poly.pkl')
    
    print("\nTraining and saving TDS → Sodium model...")
    results_tds_na, model_tds_na, poly_tds_na, scaler_X_tds_na, scaler_y_tds_na = improve_tds_sodium_prediction(df)
    
    joblib.dump(model_tds_na, 'models/tds_sodium_model.pkl')
    joblib.dump(scaler_X_tds_na, 'models/tds_sodium_scaler_X.pkl')
    joblib.dump(scaler_y_tds_na, 'models/tds_sodium_scaler_y.pkl')
    joblib.dump(poly_tds_na, 'models/tds_sodium_poly.pkl')
    
    print("\nTraining and saving Chloride → EC model...")
    results_cl_ec, model_cl_ec, poly_cl_ec, scaler_X_cl_ec, scaler_y_cl_ec = improve_chloride_ec_prediction(df)
    
    joblib.dump(model_cl_ec, 'models/chloride_ec_model.pkl')
    joblib.dump(scaler_X_cl_ec, 'models/chloride_ec_scaler_X.pkl')
    joblib.dump(scaler_y_cl_ec, 'models/chloride_ec_scaler_y.pkl')
    joblib.dump(poly_cl_ec, 'models/chloride_ec_poly.pkl')
    
    print("\nAll models saved successfully!")
    
    # Print final performance summary
    print("\nModel Performance Summary:")
    print("\nLinear Models:")
    print("EC → TDS:")
    print(f"Test R²: {results_linear['EC_TDS']['test']['r2']:.4f}")
    print(f"Test RMSE: {results_linear['EC_TDS']['test']['rmse']:.4f}")
    
    print("\nCl → Na:")
    print(f"Test R²: {results_linear['Cl_Na']['test']['r2']:.4f}")
    print(f"Test RMSE: {results_linear['Cl_Na']['test']['rmse']:.4f}")
    
    print("\nImproved Models:")
    print("\nTDS → Chloride:")
    print(f"Test R²: {results_tds_cl['test']['r2']:.4f}")
    print(f"Test RMSE: {results_tds_cl['test']['rmse']:.4f}")
    
    print("\nTDS → Sodium:")
    print(f"Test R²: {results_tds_na['test']['r2']:.4f}")
    print(f"Test RMSE: {results_tds_na['test']['rmse']:.4f}")
    
    print("\nChloride → EC:")
    print(f"Test R²: {results_cl_ec['test']['r2']:.4f}")
    print(f"Test RMSE: {results_cl_ec['test']['rmse']:.4f}")

if __name__ == "__main__":
    save_all_models() 