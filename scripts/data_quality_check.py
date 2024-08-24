import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy.stats import pearsonr
import statsmodels.api as sm
from pandas.plotting import scatter_matrix

#check missing value
def check_missing_values(df):
    missing_values = df.isnull().sum()
    return missing_values

#check incorrect entry
def check_incorrect_entries(df):
    incorrect_entries = df[(df['GHI'] < 0) | (df['DNI'] < 0) | (df['DHI'] < 0)]
    return incorrect_entries

"""
def handle_outliers(df, z_thresh=3):
    numeric_df = df.select_dtypes(include=[np.number])
    z_scores = np.abs((numeric_df - numeric_df.mean()) / numeric_df.std())
    df_no_outliers = df[(z_scores < z_thresh).all(axis=1)]
    return df_no_outliers

def clean_missing_values(df):
    df_cleaned = df.dropna(axis=1, how='all')
    df_cleaned = df_cleaned.ffill()
    return df_cleaned

def remove_negative_values(df, columns):
    for column in columns:
        df.loc[:, column] = df[column].map(lambda x: max(x, 0))
    return df

def final_cleaning_steps(df):
    df_final = df.drop_duplicates()
    df_final.reset_index(drop=True, inplace=True)
    return df_final
"""

#outlier by z-score
def detect_outliers_zscore(data, threshold=3):
    mean = np.mean(data)
    std_dev = np.std(data)
    z_scores = (data - mean) / std_dev
    outliers = data[np.abs(z_scores) > threshold]
    return outliers

def plot_irradiance_and_temperature(df):
    plt.figure(figsize=(14, 8))

    plt.subplot(2, 2, 1)
    plt.plot(df['Timestamp'], df['GHI'], color='orange', label='GHI')
    plt.title(f'GHI Over Time')
    plt.xlabel('Time')
    plt.ylabel('GHI (W/m²)')
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=1))  # Show ticks monthly
    plt.gca().xaxis.set_minor_locator(mdates.WeekdayLocator())  # Minor ticks weekly
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    plt.legend()

    plt.subplot(2, 2, 2)
    plt.plot(df['Timestamp'], df['DNI'], color='blue', label='DNI')
    plt.title(f'DNI Over Time')
    plt.xlabel('Time')
    plt.ylabel('DNI (W/m²)')
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=1))  
    plt.gca().xaxis.set_minor_locator(mdates.WeekdayLocator())  
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    plt.legend()

    plt.subplot(2, 2, 3)
    plt.plot(df['Timestamp'], df['DHI'], color='green', label='DHI')
    plt.title(f'DHI Over Time')
    plt.xlabel('Time')
    plt.ylabel('DHI (W/m²)')
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=1))  
    plt.gca().xaxis.set_minor_locator(mdates.WeekdayLocator())  
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    plt.legend()

    plt.subplot(2, 2, 4)
    plt.plot(df['Timestamp'], df['Tamb'], color='red', label='Tamb')
    plt.title(f'Ambient Temperature (Tamb) Over Time')
    plt.xlabel('Time')
    plt.ylabel('Temperature (°C)')
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=1))  
    plt.gca().xaxis.set_minor_locator(mdates.WeekdayLocator())  
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    plt.legend()

    plt.tight_layout()
    plt.show()


def plot_sensor_readings_with_cleaning(df):
    plt.figure(figsize=(14, 8))

    plt.subplot(2, 1, 1)
    plt.plot(df['Timestamp'], df['ModA'], color='purple', label='ModA')
    plt.scatter(df[df['Cleaning'] == 1]['Timestamp'], 
                df[df['Cleaning'] == 1]['ModA'], 
                color='red', label='Cleaning Event', s=10)
    plt.title(f'ModA Over Time with Cleaning Events')
    plt.xlabel('Time')
    plt.ylabel('ModA (W/m²)')
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=1))  
    plt.gca().xaxis.set_minor_locator(mdates.WeekdayLocator())  
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.plot(df['Timestamp'], df['ModB'], color='teal', label='ModB')
    plt.scatter(df[df['Cleaning'] == 1]['Timestamp'], 
                df[df['Cleaning'] == 1]['ModB'], 
                color='red', label='Cleaning Event', s=10)
    plt.title(f'ModB Over Time with Cleaning Events')
    plt.xlabel('Time')
    plt.ylabel('ModB (W/m²)')
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=1))  
    plt.gca().xaxis.set_minor_locator(mdates.WeekdayLocator())  
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    plt.legend()

    plt.tight_layout()
    plt.show()


#Heatmap
def plot_correlation_heatmap(df):
    correlation_columns = ['GHI', 'DNI', 'DHI', 'TModA', 'TModB']
    
    corr_matrix = df[correlation_columns].corr()
    
    plt.figure(figsize=(10, 8))
    
    sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', square=True, 
                cbar_kws={"shrink": .8}, linewidths=.5, linecolor='black')
    
    plt.title('Correlation Heatmap')
    plt.show()

#pair plot
def plot_pair_plot(df):
    pair_plot_columns = ['GHI', 'DNI', 'DHI', 'TModA', 'TModB']
    
    sns.pairplot(df[pair_plot_columns], diag_kind='kde', markers='o')
    
    plt.suptitle('Pair Plot of Solar Radiation and Temperature Measures', y=1.02)
    plt.show()

#scatter matrix
def plot_scatter_matrix(df):
    scatter_matrix_columns = ['GHI', 'DNI', 'DHI', 'WS', 'WSgust', 'WD']
    
    scatter_matrix(df[scatter_matrix_columns], alpha=0.5, figsize=(12, 12), diagonal='kde')
    
    plt.suptitle('Scatter Matrix of Solar Irradiance and Wind Conditions', y=1.02)
    plt.show()


def plot_wind_analysis(df):
    wind_direction_rad = np.deg2rad(df['WD'])

    plt.figure(figsize=(10, 8))

    ax = plt.subplot(111, projection='polar')

    scatter = ax.scatter(wind_direction_rad, df['WS'], 
                         c=df['WS'], cmap='viridis', alpha=0.75, edgecolors='k', s=50)

    cbar = plt.colorbar(scatter, ax=ax, orientation='vertical')
    cbar.set_label('Wind Speed (m/s)')

    ax.set_title('Wind Speed and Direction Analysis', va='bottom')
    plt.show()


def temperature_humidity_analysis(df):
    plt.figure(figsize=(14, 6))
    plt.subplot(1, 2, 1)
    sns.scatterplot(x=df['RH'], y=df['Tamb'], alpha=0.6, edgecolor=None)
    plt.title('Relative Humidity vs Ambient Temperature')
    plt.xlabel('Relative Humidity (%)')
    plt.ylabel('Temperature (°C)')
    
    plt.subplot(1, 2, 2)
    sns.scatterplot(x=df['RH'], y=df['GHI'], alpha=0.6, edgecolor=None)
    plt.title('Relative Humidity vs Global Horizontal Irradiance (GHI)')
    plt.xlabel('Relative Humidity (%)')
    plt.ylabel('GHI (W/m²)')
    
    plt.tight_layout()
    plt.show()

    # Correlation Analysis
    corr_tamb, _ = pearsonr(df['RH'], df['Tamb'])
    corr_ghi, _ = pearsonr(df['RH'], df['GHI'])
    
    print(f'Correlation between RH and Tamb: {corr_tamb:.2f}')
    print(f'Correlation between RH and GHI: {corr_ghi:.2f}')
    
    X = df['RH']
    y = df['Tamb']
    
    X = sm.add_constant(X)
    
    model = sm.OLS(y, X).fit()
    predictions = model.predict(X) 

    # Plotting regression
    plt.figure(figsize=(8, 6))
    plt.scatter(df['RH'], df['Tamb'], alpha=0.5, label='Data')
    plt.plot(df['RH'], predictions, color='red', label='Regression Line')
    plt.title('Regression Analysis: RH vs Tamb')
    plt.xlabel('Relative Humidity (%)')
    plt.ylabel('Temperature (°C)')
    plt.legend()
    plt.show()

    print(model.summary())


def plot_histograms(df):
    variables = ['GHI', 'DNI', 'DHI', 'WS', 'Tamb', 'TModA', 'TModB']
    titles = ['Global Horizontal Irradiance (GHI)', 
              'Direct Normal Irradiance (DNI)', 
              'Diffuse Horizontal Irradiance (DHI)', 
              'Wind Speed (WS)', 
              'Ambient Temperature (Tamb)', 
              'Module Temperature A (TModA)', 
              'Module Temperature B (TModB)']
    
    plt.figure(figsize=(18, 14))

    for i, var in enumerate(variables):
        plt.subplot(4, 2, i+1)
        sns.histplot(df[var], bins=50, kde=True, color='skyblue')
        plt.title(titles[i])
        plt.xlabel(var)
        plt.ylabel('Frequency')

    plt.tight_layout()
    plt.show()

def plot_bubble_chart(df, x_var, y_var, size_var, color_var, xlabel, ylabel, title):
    
    plt.figure(figsize=(10, 6))
    
    size = (df[size_var] - df[size_var].min()) / (df[size_var].max() - df[size_var].min()) * 1000
    
    plt.scatter(df[x_var], df[y_var], s=size, c=df[color_var], alpha=0.6, cmap='viridis')
    
    plt.colorbar(label=color_var)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    
    plt.show()