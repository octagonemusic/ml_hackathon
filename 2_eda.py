import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
import os

# Create results directory if it doesn't exist
if not os.path.exists('eda_results'):
    os.makedirs('eda_results')
    os.makedirs('eda_results/distributions')
    os.makedirs('eda_results/correlations')
    os.makedirs('eda_results/seasonal')
    os.makedirs('eda_results/stats')

def load_cleaned_data():
    df = pd.read_csv('cleaned_water_quality_data.csv', parse_dates=['Date'])
    print("\nAvailable columns:")
    print(df.columns.tolist())
    return df

def plot_distributions(df):
    numeric_cols = [col for col in df.select_dtypes(include=[np.number]).columns 
                   if col not in ['Year', 'Month']]
    
    distribution_stats = {}
    for col in numeric_cols:
        plt.figure(figsize=(10, 6))
        
        # Distribution plot
        sns.histplot(data=df, x=col, kde=True)
        plt.title(f'Distribution of {col}')
        
        # Add statistical annotations
        mean_val = df[col].mean()
        median_val = df[col].median()
        skew_val = df[col].skew()
        
        plt.axvline(mean_val, color='red', linestyle='--', label=f'Mean: {mean_val:.2f}')
        plt.axvline(median_val, color='green', linestyle='--', label=f'Median: {median_val:.2f}')
        plt.text(0.95, 0.95, f'Skewness: {skew_val:.2f}', 
                transform=plt.gca().transAxes, ha='right')
        
        plt.legend()
        plt.savefig(f'eda_results/distributions/{col.replace("/", "_")}_distribution.png')
        plt.close()
        
        # Save stats
        distribution_stats[col] = {
            'mean': mean_val,
            'median': median_val,
            'skewness': skew_val
        }
    
    # Save distribution stats to CSV
    pd.DataFrame(distribution_stats).to_csv('eda_results/stats/distribution_statistics.csv')

def analyze_by_location(df):
    numeric_cols = [col for col in df.select_dtypes(include=[np.number]).columns 
                   if col not in ['Year', 'Month']]
    
    # District-wise analysis
    if 'District Name' in df.columns:
        district_stats = df.groupby('District Name')[numeric_cols].agg(['mean', 'std'])
        district_stats.to_csv('eda_results/stats/district_statistics.csv')
    
    # Basin-wise analysis
    if 'Basin Name' in df.columns:
        basin_stats = df.groupby('Basin Name')[numeric_cols].agg(['mean', 'std'])
        basin_stats.to_csv('eda_results/stats/basin_statistics.csv')

def plot_parameter_relationships(df):
    high_corr_pairs = [
        ('Total Dissolved Solids (mg/L)', 'Electrical Conductivity (µS/cm) at 25°C)'),
        ('Chloride (mg/L)', 'Sodium (mg/L)'),
        ('Total Dissolved Solids (mg/L)', 'Chloride (mg/L)')
    ]
    
    correlation_stats = {}
    for param1, param2 in high_corr_pairs:
        plt.figure(figsize=(10, 6))
        sns.scatterplot(data=df, x=param1, y=param2, alpha=0.5)
        
        # Add regression line
        sns.regplot(data=df, x=param1, y=param2, scatter=False, color='red')
        
        # Calculate correlation coefficient
        corr = df[param1].corr(df[param2])
        correlation_stats[f"{param1} vs {param2}"] = corr
        
        plt.title(f'Relationship between {param1} and {param2}\nCorrelation: {corr:.3f}')
        plt.savefig(f'eda_results/correlations/{param1[:10]}_{param2[:10]}_correlation.png')
        plt.close()
    
    # Save correlation stats
    pd.DataFrame.from_dict(correlation_stats, orient='index', columns=['correlation'])\
        .to_csv('eda_results/stats/correlation_statistics.csv')

def analyze_seasonal_patterns(df):
    numeric_cols = [col for col in df.select_dtypes(include=[np.number]).columns 
                   if col not in ['Year', 'Month']]
    
    # Monthly patterns
    monthly_means = df.groupby('Month')[numeric_cols].mean()
    monthly_means.to_csv('eda_results/stats/monthly_statistics.csv')
    
    for col in numeric_cols:
        plt.figure(figsize=(10, 6))
        monthly_means[col].plot(kind='line', marker='o')
        plt.title(f'Monthly Pattern of {col}')
        plt.xlabel('Month')
        plt.ylabel(col)
        plt.grid(True)
        plt.savefig(f'eda_results/seasonal/{col.replace("/", "_")}_seasonal.png')
        plt.close()

def perform_eda(df):
    print("Starting EDA analysis...")
    
    print("1. Analyzing distributions...")
    plot_distributions(df)
    
    print("2. Analyzing geographical patterns...")
    analyze_by_location(df)
    
    print("3. Analyzing parameter relationships...")
    plot_parameter_relationships(df)
    
    print("4. Analyzing seasonal patterns...")
    df['Year'] = pd.to_datetime(df['Date']).dt.year
    df['Month'] = pd.to_datetime(df['Date']).dt.month
    analyze_seasonal_patterns(df)
    
    print("\nEDA completed! Results saved in 'eda_results' folder")

if __name__ == "__main__":
    df = load_cleaned_data()
    perform_eda(df)