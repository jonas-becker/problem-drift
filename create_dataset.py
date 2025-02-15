
import pandas as pd
import os
import uuid
import seaborn as sns
import matplotlib.pyplot as plt
from collections import Counter
from itertools import combinations
import logging
import nltk
import numpy as np
from scipy.stats import pearsonr
from scipy.stats import spearmanr
from scipy.stats import kendalltau
import textwrap
nltk.download('punkt')


logger = logging.getLogger("mallm")

def corr_heatmap_with_pval(df, method = 'pearson', figsize=(26, 18), filename=None, title=None):
    """
    df: dataframe to be used. Ensured the dataframe has been sliced to contain only the column you need. It accepts only numerical columns
    method: default uses the pearson method. It overall permits 3 methods; 'pearson', 'spearman' and 'kendall'
    figsize: default is (20, 10) but you can change it based on your preference
    title: Specify the title for your chart, default is None
    """
    # Make a copy of the df
    data = df.copy()
   
    # Log initial size and NaN counts
    initial_size = len(data)
    nan_counts = data.isna().sum()
    logger.info(f"\nInitial data size: {initial_size} rows")
    logger.info("NaN counts per column:")
    for col, count in nan_counts.items():
        logger.info(f"  {col}: {count} NaN values ({(count/initial_size)*100:.1f}%)")
    data = data.dropna()
    final_size = len(data)
    dropped_rows = initial_size - final_size
    logger.info(f"Dropped {dropped_rows} rows ({(dropped_rows/initial_size)*100:.1f}% of data) containing NaN values")
    logger.info(f"Final data size: {final_size} rows")
    if final_size == 0:
        logger.warning(f"No valid data points remaining after dropping NaN values for {filename}")
        return

    # Check features correlation
    corr = data.corr(method = method)
    
    # Check if correlation matrix contains any valid values
    if corr.isnull().all().all():
       logger.warning(f"Correlation matrix is all NaN for {filename}")
       return

    # Create a mask to hide the upper triangle
    mask = np.zeros_like(corr, dtype=bool)
    mask[np.triu_indices_from(mask)] = True

    # Set the diagonal elements of the mask to False to display self-correlation
    np.fill_diagonal(mask, False)

    fig, ax = plt.subplots(figsize=figsize)
    
    plt.title(title, fontsize=40)
    #plt.tight_layout()

    sns.heatmap(corr,
                annot=True,
                annot_kws={"fontsize": 40},  # Adjust annotation font size
                fmt='.2f',
                linewidths=0.5,
                cmap=sns.diverging_palette(240, 10, center="dark", as_cmap=True),
                mask=mask,
                ax=ax,
                center=0
    )

    # Create a function to calculate and format p-values
    p_values = np.full((corr.shape[0], corr.shape[1]), np.nan)
    for i in range(corr.shape[0]):
        for j in range(i+1, corr.shape[1]):
            x = data.iloc[:, i].astype(float)
            y = data.iloc[:, j].astype(float)
            mask = ~np.logical_or(np.isnan(x), np.isnan(y))
            if np.sum(mask) > 0:
                if method == 'pearson':
                    p_values[i, j] = pearsonr(x[mask], y[mask])[1] #Changes based on the method chosen in the function
                elif method == 'kendall':
                    p_values[i, j] = kendalltau(x[mask], y[mask])[1]
                elif method == 'spearman':
                    p_values[i, j] = spearmanr(x[mask], y[mask])[1]

    p_values = pd.DataFrame(p_values, columns=corr.columns, index=corr.index)

    # Create a mask for the p-values heatmap
    mask_pvalues = np.triu(np.ones_like(p_values), k=1)

    # Calculate the highest and lowest correlation coefficients
    np.max(corr.max())
    np.min(corr.min())

    # Annotate the heatmap with p-values and change text color based on correlation value
    for i in range(p_values.shape[0]):
        for j in range(p_values.shape[1]):
            if mask_pvalues[i, j]:
                p_value = p_values.iloc[i, j]
                if not np.isnan(p_value):
                    corr.iloc[i, j]
                    text_color = 'white'
                    if p_value <= 0.001:
                        ax.text(i + 0.5, j + 0.8, '(p < 0.001)',
                                horizontalalignment='center',
                                verticalalignment='center',
                                fontsize=30,
                                color=text_color)
                    elif p_value <= 0.01:
                        #include double asterisks for p-value <= 0.01
                        ax.text(i + 0.5, j + 0.8, f'(p = {p_value:.3f})',
                                horizontalalignment='center',
                                verticalalignment='center',
                                fontsize=30,
                                color=text_color)
                    elif p_value <= 0.05:
                        #include single asterisks for p-value <= 0.05
                        ax.text(i + 0.5, j + 0.8, f'(p = {p_value:.3f})',
                                horizontalalignment='center',
                                verticalalignment='center',
                                fontsize=30,
                                color=text_color)
                    else:
                        ax.text(i + 0.5, j + 0.8, f'(p = {p_value:.3f})',
                                horizontalalignment='center',
                                verticalalignment='center',
                                fontsize=30,
                                color=text_color)

    # Customize x-axis labels
    x_labels = [textwrap.fill(label.get_text(), 12) for label in ax.get_xticklabels()]
    ax.set_xticklabels(x_labels, rotation=0, ha="center", fontsize=40)

    # Customize y-axis labels
    y_labels = [textwrap.fill(label.get_text(), 12) for label in ax.get_yticklabels()]
    ax.set_yticklabels(y_labels, rotation=0, ha="right", fontsize=40)
    ax.grid(False)
    ax.collections[0].set_clim(-1,1)  

    # Fix the legend scale to be between -0.1 and 1.0
    cbar = ax.collections[0].colorbar
    cbar.set_ticks([-1.0, -0.9, -0.8,-0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1, 0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    cbar.set_ticklabels(['-1.0', '', '', '', '', '', '', '', '', '', '0.0', '', '', '', '', '', '', '', '', '', '1.0'])
    # Increase font size of colorbar ticks
    cbar.ax.tick_params(labelsize=40)
    
    output_file = os.path.join("data/DRIFTEval", filename)
    plt.savefig(output_file, bbox_inches='tight')
    plt.close()

csv_files = [f for f in os.listdir('data/DRIFTEval/csv') if f.endswith('.csv')]

dfs = []
for csv_file in csv_files:
    df = pd.read_csv(os.path.join('data/DRIFTEval/csv', csv_file))
    df = df.dropna(subset=['Dataset', 'Error Type Label'])
    dfs.append(df)

df_selected = pd.DataFrame(columns=[
    'sampleId', 'dataset', 'input', 'instruction', 'context', 'reference',
    'personas', 'personaDiversity', 'driftTurn', 'driftStrength',
    'extractedMessages', 'solutionBefore', 'solutionAfter', 'errorTypes',
    'errorTypesExplanation'
])


for df in dfs:
    for idx, row in df.iterrows():
        next_idx = len(df_selected) if len(df_selected) > 0 else 0
        
        if row['Sample ID'] == "data/datasets/ethics.json":
            df_selected.at[next_idx, 'sampleId'] = str(uuid.uuid4())
            df_selected.at[next_idx, 'dataset'] = row['Sample ID'].split(".")[0].split("/")[-1]
            df_selected.at[next_idx, 'input'] = row['Dataset']
            df_selected.at[next_idx, 'instruction'] = row['Input']
            df_selected.at[next_idx, 'context'] = row['Instruction']
        else:
            df_selected.at[next_idx, 'sampleId'] = row['Sample ID']
            df_selected.at[next_idx, 'dataset'] = row['Dataset'].split(".")[0].split("/")[-1]
            df_selected.at[next_idx, 'input'] = row['Input']
            df_selected.at[next_idx, 'instruction'] = row['Instruction']
            df_selected.at[next_idx, 'context'] = row['Context']
        df_selected.at[next_idx, 'reference'] = row['Reference']
        df_selected.at[next_idx, 'personas'] = row['Personas']
        df_selected.at[next_idx, 'personaDiversity'] = float(row['Persona Diversity'].replace(',', '.'))
        df_selected.at[next_idx, 'driftTurn'] = row['Strongest Drift Turn']
        df_selected.at[next_idx, 'driftStrength'] = row['Drift Strength']
        df_selected.at[next_idx, 'extractedMessages'] = {
            1: row['Extracted Messages (4-6 are the drifting messages)\n\nMessages 1-3 are one turn. Messages 4-6 are the next turn. Labels should refer to messages 4-6 (the drifting turn)'],
            2: row['Unnamed: 17'],
            3: row['Unnamed: 18'],
            4: row['Unnamed: 19'],
            5: row['Unnamed: 20'],
            6: row['Unnamed: 21']
        }
        df_selected.at[next_idx, 'solutionBefore'] = row['Voted Solution (Before)']
        df_selected.at[next_idx, 'solutionAfter'] = row['Voted Solution (After)']
        df_selected.at[next_idx, 'errorTypes'] = row['Error Type Label'].split(", ")
        df_selected.at[next_idx, 'errorTypesExplanation'] = row['Explanation']

print(f"Done. Length of the dataset: {len(df_selected)}")

all_error_types = [error for sublist in df_selected['errorTypes'] for error in sublist]
error_type_counts = Counter(all_error_types)
co_occurrence_counts = {error: Counter() for error in error_type_counts}
for error_list in df_selected['errorTypes']:
    for error1, error2 in combinations(error_list, 2):
        co_occurrence_counts[error1][error2] += 1
        co_occurrence_counts[error2][error1] += 1

correlations = {}
for error1 in co_occurrence_counts:
    correlations[error1] = {}
    for error2 in co_occurrence_counts[error1]:
        co_occurrence = co_occurrence_counts[error1][error2]
        correlation = co_occurrence / min(error_type_counts[error1], error_type_counts[error2])
        correlations[error1][error2] = correlation

print("Correlations of labels in df_selected['errorTypes']:")
sorted_correlations = sorted(
    [(error1, error2, correlations[error1][error2]) for error1 in correlations for error2 in correlations[error1]],
    key=lambda x: x[2],
    reverse=True
)
for error1, error2, correlation in sorted_correlations:
    print(f"{error1} - {error2}: {correlation:.2f}")

correlation_matrix = pd.DataFrame(correlations).fillna(0)
corr_heatmap_with_pval(correlation_matrix, method='pearson', figsize=(43, 20), filename='error_types_correlation_matrix.pdf', title='Correlation Matrix of Error Types')

output_path = os.path.join('data/DRIFTEval', 'DriftEval.json')
df_selected.to_json(output_path, orient='records', indent=4)
print(f"Exported dataset to {output_path}")
