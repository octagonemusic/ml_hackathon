import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

def clean_outliers(df):
    # Define valid ranges for parameters
    valid_ranges = {
        'pH': (4, 10),  # typical water quality range
        'Total Dissolved Solids (mg/L)': (0, 3000),  # typical freshwater range
        'Temperature': (0, 45),  # reasonable water temperature range
        'Turbidity (NTU)': (0, 1000),
        'Chloride (mg/L)': (0, 1000),
        'Electrical Conductivity (µS/cm) at 25°C)': (0, 5000),
        'Fluoride (mg/L)': (0, 10),
        'Sodium (mg/L)': (0, 1000)  # typical freshwater range
    }
    
    original_size = len(df)
    print("\nCleaning outliers:")
    
    for param, (min_val, max_val) in valid_ranges.items():
        if param in df.columns:
            before = len(df)
            df = df[df[param].between(min_val, max_val)]
            removed = before - len(df)
            print(f"Removed {removed} rows where {param} was outside range [{min_val}, {max_val}]")
    
    print(f"\nTotal rows removed: {original_size - len(df)}")
    return df

def load_and_clean_data():
    # Read the Excel file
    df = pd.read_excel('surfacewater_data.xlsx', header=5)
    
    print("Initial shape:", df.shape)
    
    # Replace '-' with NaN
    df = df.replace('-', np.nan)
    
    # Define parameters to keep (now including Sodium)
    numeric_columns = [
        'pH',
        'Total Dissolved Solids (mg/L)',
        'Temperature',
        'Turbidity (NTU)',
        'Chloride (mg/L)',
        'Electrical Conductivity (µS/cm) at 25°C)',
        'Total Alkalinity (mg/L)',
        'Chemical Oxygen Demand (mg/L)',
        'Nitrate (mg/L)',
        'Fluoride (mg/L)',
        'Sodium (mg/L)'
    ]
    
    # Print initial missing values for Sodium
    print("\nInitial Sodium missing values:")
    print(df['Sodium (mg/L)'].isnull().sum(), "missing values")
    
    # Convert columns to numeric, errors='coerce' will convert invalid values to NaN
    for col in numeric_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Remove duplicate columns
    duplicates = {
        'pH': ['pH_Field', 'ph'],
        'Electrical Conductivity (µS/cm) at 25°C)': ['Electrical Conductivity Field'],
        'Total Alkalinity (mg/L)': ['Alkalinity(Total)'],
    }
    
    # Remove duplicate columns
    for main_col, dup_cols in duplicates.items():
        if main_col in df.columns:
            df = df.drop(columns=[col for col in dup_cols if col in df.columns])
    
    print("\nShape after removing duplicates:", df.shape)
    
    # Keep only the most important water quality parameters
    important_params = [
        'Station Name', 'State Name', 'District Name', 'Basin Name', 
        'Date'
    ] + numeric_columns
    
    # Keep only columns that exist in the dataset
    existing_columns = [col for col in important_params if col in df.columns]
    df = df[existing_columns]
    
    # Print missing value analysis
    print("\nMissing value analysis:")
    for col in df.columns:
        missing = df[col].isnull().sum()
        missing_pct = (missing/len(df))*100
        print(f"{col}: {missing} missing values ({missing_pct:.2f}%)")
    
    # Remove columns with more than 50% missing values
    missing_threshold = 0.5  # 50%
    columns_to_keep = []
    for col in df.columns:
        missing_rate = df[col].isnull().sum() / len(df)
        if missing_rate < missing_threshold:
            columns_to_keep.append(col)
    
    df = df[columns_to_keep]
    print("\nColumns retained after missing value threshold:")
    print(columns_to_keep)
    
    # Convert date column to datetime
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    
    # Remove rows where all numerical parameters are missing
    numeric_columns = [col for col in numeric_columns if col in df.columns]
    before_rows = len(df)
    df = df.dropna(subset=numeric_columns)
    after_rows = len(df)
    print(f"Removed {before_rows - after_rows} rows with any missing numerical values")
    
    # Clean outliers
    df = clean_outliers(df)
    print("Final shape after outlier removal:", df.shape)
    
    return df

def analyze_cleaned_data(df):
    if len(df) == 0:
        print("Error: DataFrame is empty after cleaning!")
        return None
        
    # Display basic statistics for numeric columns
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    print("\nBasic Statistics:")
    print(df[numeric_columns].describe())
    
    # Create correlation matrix for numerical columns
    if len(numeric_columns) > 0:
        print("\nNumerical features used for correlation:")
        for idx, col in enumerate(numeric_columns, 1):
            print(f"{idx}. {col}")
            
        correlation = df[numeric_columns].corr()
        
        # Plot correlation heatmap
        plt.figure(figsize=(12, 8))
        sns.heatmap(correlation, annot=True, cmap='coolwarm', center=0)
        plt.title('Correlation Matrix of Water Quality Parameters')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.show()
        
        # Print strongest correlations
        print("\nStrongest correlations:")
        correlations = []
        for i in range(len(correlation.columns)):
            for j in range(i+1, len(correlation.columns)):
                correlations.append({
                    'feature1': correlation.columns[i],
                    'feature2': correlation.columns[j],
                    'correlation': abs(correlation.iloc[i,j])
                })
        
        correlations_df = pd.DataFrame(correlations)
        print(correlations_df.sort_values('correlation', ascending=False).head(5))
        
        return correlation
    else:
        print("No numerical features available for correlation analysis")
        return None

if __name__ == "__main__":
    cleaned_df = load_and_clean_data()
    
    if len(cleaned_df) > 0:
        correlation_matrix = analyze_cleaned_data(cleaned_df)
        cleaned_df.to_csv('cleaned_water_quality_data.csv', index=False)
        print("\nCleaned data saved to 'cleaned_water_quality_data.csv'")
    else:
        print("Error: No data remaining after cleaning process!")
