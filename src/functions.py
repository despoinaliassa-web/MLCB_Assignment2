import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def load_heart_disease_data(filepath):
    """
    Loads the UCI Heart Disease dataset.
    Defines the num column as a binary target (0=no disease, 1=disease). 
    """
    # Load the dataset
    df = pd.read_csv(filepath)
    
    # Conversion to binary target: 0 (healthy), 1 (sick) 
    # If num > 0, then it is considered as presence of disease (1)
    df['target'] = (df['num'] > 0).astype(int)
    
    # Removal of the original num column to avoid redundancy
    df = df.drop(columns=['num'])
    
    return df
def overview(df):
    """
    Overview of the dataset:
    """
    print("\n--- First 10 Rows of the Dataset ---")
    return df.head(10)

def get_data_summary(df):
    """
    Returns basic information about the dataset: 
    missing values, types, duplicates, and class imbalance. 
    """
    summary = {
        "shape": df.shape,
        "missing_values": df.isnull().sum(),
        "duplicates": df.duplicated().sum(),
        "dtypes": df.dtypes,
        "class_balance": df['target'].value_counts(normalize=True) * 100
    }
    return summary

def plot_correlation_heatmap(df, save_path=None):
    """
    Creation of a correlation heatmap for all features in the dataset
    """
    plt.figure(figsize=(12, 10))
    # Calculation of the Pearson correlation coefficient
    correlation_matrix = df.corr()
    
    # Heatmap creation
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
    plt.title('Correlation Heatmap of Heart Disease Features')
    plt.show()
    
    if save_path:
        # Creation of the directory if it doesn't exist
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved graph to: {save_path}")
    
    plt.show()

def plot_categorical_distributions(df, categorical_features):
    """
    Creates count plots for the categorical variables in relation to the target
    """
    num_features = len(categorical_features)
    rows = (num_features + 1) // 2
    plt.figure(figsize=(15, rows * 5), dpi=150)
    
    for i, col in enumerate(categorical_features, 1):
        plt.subplot(rows, 2, i)
        sns.countplot(x=col, hue='target', data=df, palette='rocket')
        plt.title(f'Distribution of {col} by Target')
    
    plt.tight_layout()

    plot_path = '../figures/cat_distributions.png'
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')

    plt.show()



def format_results_table(summary_df, stage_label):
    """
    Formats the statistical summary into a clean, human-readable table
    with 95% CI in a single column.
    """
    final_rows = []
    
    for _, row in summary_df.iterrows():
        res = {
            'Model': row['Algorithm'],
            'Stage': stage_label,
            'MCC_median': round(row['MCC_median'], 3),
            '95% CI (MCC)': f"[{row['MCC_95CI'][0]:.3f}, {row['MCC_95CI'][1]:.3f}]",
            'AUC_median': round(row['AUC_median'], 3),
            'BA_median': round(row['BA_median'], 3),
            'F1_median': round(row['F1_median'], 3),
            'Recall_median': round(row['Recall_median'], 3),
            'Precision_median': round(row['Precision_median'], 3)
        }
        final_rows.append(res)
    
    return pd.DataFrame(final_rows)



def plot_simple_comparison(df_raw_combined, metric='MCC'):
    """
    Creates a comparison between Baseline and Tuned models using the raw combined dataframe
    Seaborn automatically calculates the confidence intervals from the data
    """
    if not os.path.exists('../figures'):
        os.makedirs('../figures')

    plt.figure(figsize=(12, 7))
    
    
    ax = sns.barplot(
        data=df_raw_combined, 
        x='Algorithm', 
        y=metric, 
        hue='Stage', 
        palette='rocket',
        capsize=.1,
        errorbar=('ci', 95) # Seaborn automatically calculates the 95% CI
    )

    plt.title(f'Model Performance Comparison: {metric} (95% CI)', fontsize=15)
    plt.ylabel(f'Score {metric}')
    plt.xlabel('Algorithm')
    plt.xticks(rotation=45)
    plt.grid(axis='y', linestyle='--', alpha=0.4)
    plt.legend(title='Configuration', loc='upper right')
    plt.tight_layout()
    
    # Save the plot
    plt.savefig(f'../figures/comparison_{metric}.png', dpi=300)
    plt.show()