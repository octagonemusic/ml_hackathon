import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
import os

# Create directory structure for storing EDA results
if not os.path.exists('eda_results'):
    os.makedirs('eda_results')
    os.makedirs('eda_results/distributions')    # For distribution plots
    os.makedirs('eda_results/correlations')     # For correlation analysis
    os.makedirs('eda_results/seasonal')         # For temporal analysis
    os.makedirs('eda_results/stats')            # For statistical summaries

def load_cleaned_data():
    """
    Load the cleaned dataset and display available columns
    Returns:
        pandas.DataFrame: Cleaned water quality dataset
    """
    df = pd.read_csv('cleaned_water_quality_data.csv', parse_dates=['Date'])
    print("\nAvailable columns:")
    print(df.columns.tolist())
    return df

def plot_distributions(df):
    """
    Generate and save distribution plots for each numeric parameter
    - Creates histograms with KDE
    - Adds mean, median lines
    - Calculates skewness
    - Saves distribution statistics
    
    Args:
        df (pandas.DataFrame): Input dataset
    """
    # Select numeric columns, excluding time-related columns
    numeric_cols = [col for col in df.select_dtypes(include=[np.number]).columns 
                   if col not in ['Year', 'Month']]
    
    distribution_stats = {}
    for col in numeric_cols:
        plt.figure(figsize=(10, 6))
        
        # Create distribution plot with kernel density estimation
        sns.histplot(data=df, x=col, kde=True)
        plt.title(f'Distribution of {col}')
        
        # Calculate and add statistical annotations
        mean_val = df[col].mean()
        median_val = df[col].median()
        skew_val = df[col].skew()
        
        # Add reference lines and annotations
        plt.axvline(mean_val, color='red', linestyle='--', label=f'Mean: {mean_val:.2f}')
        plt.axvline(median_val, color='green', linestyle='--', label=f'Median: {median_val:.2f}')
        plt.text(0.95, 0.95, f'Skewness: {skew_val:.2f}', 
                transform=plt.gca().transAxes, ha='right')
        
        plt.legend()
        plt.savefig(f'eda_results/distributions/{col.replace("/", "_")}_distribution.png')
        plt.close()
        
        # Store statistics for later use
        distribution_stats[col] = {
            'mean': mean_val,
            'median': median_val,
            'skewness': skew_val
        }
    
    # Save all distribution statistics to CSV
    pd.DataFrame(distribution_stats).to_csv('eda_results/stats/distribution_statistics.csv')

def analyze_by_location(df):
    """
    Analyze parameter variations across different geographical locations
    - Groups data by district and basin
    - Calculates mean and standard deviation for each parameter
    
    Args:
        df (pandas.DataFrame): Input dataset
    """
    numeric_cols = [col for col in df.select_dtypes(include=[np.number]).columns 
                   if col not in ['Year', 'Month']]
    
    # Analyze district-wise patterns if district information is available
    if 'District Name' in df.columns:
        district_stats = df.groupby('District Name')[numeric_cols].agg(['mean', 'std'])
        district_stats.to_csv('eda_results/stats/district_statistics.csv')
    
    # Analyze basin-wise patterns if basin information is available
    if 'Basin Name' in df.columns:
        basin_stats = df.groupby('Basin Name')[numeric_cols].agg(['mean', 'std'])
        basin_stats.to_csv('eda_results/stats/basin_statistics.csv')

def plot_parameter_relationships(df):
    """
    Analyze and visualize relationships between key parameters
    - Creates scatter plots with regression lines
    - Calculates correlation coefficients
    - Focuses on known important parameter pairs
    
    Args:
        df (pandas.DataFrame): Input dataset
    """
    # Define parameter pairs known to have strong relationships
    high_corr_pairs = [
        ('Total Dissolved Solids (mg/L)', 'Electrical Conductivity (µS/cm) at 25°C)'),
        ('Chloride (mg/L)', 'Sodium (mg/L)'),
        ('Total Dissolved Solids (mg/L)', 'Chloride (mg/L)')
    ]
    
    correlation_stats = {}
    for param1, param2 in high_corr_pairs:
        plt.figure(figsize=(10, 6))
        # Create scatter plot
        sns.scatterplot(data=df, x=param1, y=param2, alpha=0.5)
        
        # Add regression line for trend visualization
        sns.regplot(data=df, x=param1, y=param2, scatter=False, color='red')
        
        # Calculate and store correlation coefficient
        corr = df[param1].corr(df[param2])
        correlation_stats[f"{param1} vs {param2}"] = corr
        
        plt.title(f'Relationship between {param1} and {param2}\nCorrelation: {corr:.3f}')
        plt.savefig(f'eda_results/correlations/{param1[:10]}_{param2[:10]}_correlation.png')
        plt.close()
    
    # Save correlation statistics
    pd.DataFrame.from_dict(correlation_stats, orient='index', columns=['correlation'])\
        .to_csv('eda_results/stats/correlation_statistics.csv')

def analyze_seasonal_patterns(df):
    """
    Analyze temporal patterns in water quality parameters
    - Calculates monthly averages
    - Creates time series plots
    - Identifies seasonal trends
    
    Args:
        df (pandas.DataFrame): Input dataset
    """
    numeric_cols = [col for col in df.select_dtypes(include=[np.number]).columns 
                   if col not in ['Year', 'Month']]
    
    # Calculate and save monthly averages
    monthly_means = df.groupby('Month')[numeric_cols].mean()
    monthly_means.to_csv('eda_results/stats/monthly_statistics.csv')
    
    # Create seasonal pattern plots for each parameter
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
    """
    Main function to perform complete exploratory data analysis
    - Coordinates all analysis steps
    - Provides progress updates
    
    Args:
        df (pandas.DataFrame): Input dataset
    """
    print("Starting EDA analysis...")
    
    print("1. Analyzing distributions...")
    plot_distributions(df)
    
    print("2. Analyzing geographical patterns...")
    analyze_by_location(df)
    
    print("3. Analyzing parameter relationships...")
    plot_parameter_relationships(df)
    
    print("4. Analyzing seasonal patterns...")
    # Extract temporal components
    df['Year'] = pd.to_datetime(df['Date']).dt.year
    df['Month'] = pd.to_datetime(df['Date']).dt.month
    analyze_seasonal_patterns(df)
    
    print("\nEDA completed! Results saved in 'eda_results' folder")

if __name__ == "__main__":
    df = load_cleaned_data()
    perform_eda(df)