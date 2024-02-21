import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load and prepare the dataset
df = pd.read_csv('heart_data_set.csv')

# Ensure binary variables are of type 'category'
categorical_vars = ['anaemia', 'diabetes', 'high_blood_pressure', 'sex', 'smoking', 'DEATH_EVENT']
df[categorical_vars] = df[categorical_vars].astype('category')

# Descriptive Statistics
descriptive_stats = df.describe()
print("Descriptive Statistics:\n", descriptive_stats)

# Correlation Analysis
correlation_matrix = df.corr()
print("\nCorrelation Matrix:\n", correlation_matrix)

# Function 1: Histogram of Age Distribution
def plot_age_distribution(dataframe):
    """
    This function plots the age distribution of patients.
    :param dataframe: pandas DataFrame containing the dataset
    """
    plt.figure(figsize=(10, 6))
    sns.histplot(dataframe['age'], bins=30, kde=True, color='skyblue')
    plt.title('Age Distribution of Patients')
    plt.xlabel('Age')
    plt.ylabel('Frequency')
    plt.show()

# Function 2: Scatter Plot of Ejection Fraction vs. Serum Creatinine
def plot_ef_vs_sc(dataframe):
    """
    This function plots the scatter plot of ejection fraction vs. serum creatinine,
    colored by the death event outcome.
    :param dataframe: pandas DataFrame containing the dataset
    """
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=dataframe, x='ejection_fraction', y='serum_creatinine', hue='DEATH_EVENT', style='DEATH_EVENT', palette='coolwarm', s=100)
    plt.title('Ejection Fraction vs. Serum Creatinine by Death Event')
    plt.xlabel('Ejection Fraction (%)')
    plt.ylabel('Serum Creatinine (mg/dL)')
    plt.legend(title='Death Event', labels=['Survived', 'Deceased'])
    plt.show()

# Function 3: Heatmap of Correlation Matrix
def plot_correlation_matrix(dataframe):
    """
    This function plots the heatmap of the correlation matrix for numerical variables in the dataset,
    using a unique 'viridis' color map. Adjustments are made to ensure all correlation coefficients
    are fully visible.
    :param dataframe: pandas DataFrame containing the dataset
    """
    corr_matrix = dataframe.select_dtypes(include=['float64', 'int64']).corr()
    plt.figure(figsize=(12, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='viridis', fmt=".2f", linewidths=.5, cbar_kws={'shrink': .5})
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=45)
    plt.title('Correlation Matrix of Clinical Features')
    plt.show()

# Example of how to call the functions
plot_age_distribution(df)
plot_ef_vs_sc(df)
plot_correlation_matrix(df)
