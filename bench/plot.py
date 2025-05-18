import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FormatStrFormatter
import seaborn as sns
sns.set(style="whitegrid")
fontsize = 18
datasets = ['amazon0505', 'DD', 'ppi', 'reddit', 'amazon0601', 'com-amazon', 'ddi', 'FraudYelp-RSR', 'web-BerkStan', 'protein', 'YeastH', 'Yeast']
# Feature dimensions
featdims = [256, 512, 1024]
# Set global font size
plt.rc('font', size=14)

# Read the CSV file
df = pd.read_csv('results.csv')


# Get all unique datasets and feature dimensions
# datasets = sorted(df['Dataset'].unique()) # This line was commented out in the original code
featdims = sorted(df['FeatDim'].unique())

# Define the list of methods (remove 'PT-Embedding')
methods = ['cuSPARSE', 'Sputnik', 'GE-SPMM', 'RoDe',
           'TC-GNN', 'DTC-SPMM', 'Voltrix']
methods_legend = ['cuSPARSE', 'Sputnik-SpMM', 'GE-SpMM', 'RoDe-SpMM',
           'TC-GNN', 'DTC-SpMM', 'Voltrix-SpMM']
colors = plt.cm.viridis(np.linspace(0, 1, len(methods)))

# Create a dictionary to store the data
# Data structure: data[dataset][featdim][method] = time
data = {dataset: {featdim: {method: np.nan for method in methods}
                 for featdim in featdims}
        for dataset in datasets}

# Populate the data dictionary
for _, row in df.iterrows():
    method = row['Method']
    dataset = row['Dataset']
    featdim = row['FeatDim']
    time = row['Time (ms)']

    # Only process defined methods and feature dimensions
    if method in methods and featdim in featdims:
        # Handle 'NAN' strings, convert them to np.nan
        try:
            time = float(time)
        except ValueError:
            time = np.nan
        data[dataset][featdim][method] = time

# Create a 3x4 subplot layout (because there are 12 datasets)
fig, axs = plt.subplots(3, 4, figsize=(25, 11))
axs = axs.flatten()  # Flatten the subplot array for easy iteration

# Define the width of the bars in the bar chart
width = 0.1

# Iterate over each dataset
for i, dataset in enumerate(datasets):
    ax = axs[i]
    x = np.arange(len(featdims))  # Positions of feature dimensions

    # Get cuSPARSE times for different feature dimensions as baseline
    cusparse_times = [data[dataset][featdim]['cuSPARSE'] for featdim in featdims]

    # Plot speedup for each method
    for j, method in enumerate(methods):
        if method == 'cuSPARSE':
            speedups = [1.0 for _ in featdims]  # Speedup of cuSPARSE is 1
        else:
            speedups = []
            for k, featdim in enumerate(featdims):
                method_time = data[dataset][featdim][method]
                cusparse_time = cusparse_times[k]
                if pd.isna(method_time) or method_time == 0:
                    speedups.append(np.nan)
                else:
                    speedups.append(cusparse_time / method_time)

        # Plot the bar chart
        bars = ax.bar(x + j * width, speedups, width, label=methods_legend[j], color = colors[j], edgecolor='black')

        # Iterate over each speedup and check if it's NaN
        for idx, speedup in enumerate(speedups):
            if pd.isna(speedup):
                # Get the position of the bar
                bar_x = x[idx] + j * width
                bar_y = 0.15  # Set the y-position of the text, ensuring it's above y=0.1
                # Add "CUDA ERROR" text, displayed vertically
                ax.text(bar_x, bar_y, 'CUDA ERROR', rotation=90,
                        ha='center', va='bottom', fontsize=10, color=colors[j], fontweight='bold')

    # Set title and labels
    ax.set_title(dataset, fontsize=fontsize)
    ax.set_xticks(x + width * (len(methods) - 1) / 2)
    ax.set_xticklabels(featdims, fontsize = fontsize)
    ax.set_ylim(bottom=0.1)  # Avoid log(0) issues
    ax.tick_params(axis='y', labelsize=fontsize)
    # if i % 4 == 0: # This line was commented out in the original code
    #     ax.set_ylabel('Speedup over cuSPARSE', fontsize=14) # This line was commented out

    # Add grid lines
    ax.grid(True, which="both", ls="--", linewidth=0.5, alpha=0.7)

    # Calculate the speedup range of Voltrix relative to cuSPARSE
    voltrix_speedups = []
    for k, featdim in enumerate(featdims):
        voltrix_time = data[dataset][featdim]['Voltrix']
        cusparse_time = cusparse_times[k]
        if not pd.isna(voltrix_time) and voltrix_time != 0:
            speedup = cusparse_time / voltrix_time
            voltrix_speedups.append(speedup)

    if voltrix_speedups:
        min_speedup = np.min(voltrix_speedups)
        max_speedup = np.max(voltrix_speedups)
        speedup_str = f"{min_speedup:.2f}x - {max_speedup:.2f}x speedup"
    else:
        speedup_str = "N/A speed"

    # Set the x-axis title to 'Voltrix Speedup Range'
    # Use newline character for multi-line labels
    ax.set_xlabel(f"{speedup_str}", fontsize=fontsize)
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))

# Add an overall y-axis label
fig.text(0.045, 0.5, 'Speedup over cuSPARSE', va='center', rotation='vertical', fontsize=fontsize)

# Set the legend (get from the last subplot)
handles, labels = axs[-1].get_legend_handles_labels()
fig.legend(handles, labels, loc='upper center',
           bbox_to_anchor=(0.52, 0.98), fontsize=fontsize, ncol=7, frameon=False,
            columnspacing=3.0,       # Adjust column spacing
            handletextpad=0.7,       # Adjust spacing between symbol and text
            handlelength=3.0,          # Adjust symbol length
)

# Adjust subplot spacing
plt.tight_layout(rect=[0.05, 0.05, 1, 0.95])

# Display the chart
plt.savefig('results.png', dpi=300, bbox_inches='tight')
plt.show()