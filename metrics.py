import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def violins():
    df = pd.read_csv("cleaned_data/stimuli_metrics.csv")

    # Exclude the "drawing" and "size" columns as per the user's request
    columns_to_plot = df.columns.difference(['drawing', 'size', 'ksm'])


    metric_names = {"ar": "Angular Resolution", "asp": "Aspect Ratio", "ca": "Crossing Angle", "ec": "Edge Crossings", "el":"Edge Lengths", 
                            "eo":"Edge Orthogonality", "gr":"Gabriel Ratio", "np":"Neighbourhood Preservation", "nr":"Node Resolution", "nu":"Node Uniformity"}

    # Print the min and max values for each column in the DataFrame (excluding "drawing" and "size")
    min_max_values = df[columns_to_plot].agg(['min', 'max'])

    # Display the min and max values for each column
    print(min_max_values)
    
    # Calculate if the columns need to be split differently for an uneven number
    half_columns = (len(columns_to_plot) + 1) // 2

    # Create a grid with 2 rows and half the number of columns in each row
    fig, axes = plt.subplots(2, half_columns, figsize=(15, 8))

    palette = sns.color_palette("Set2", len(df['size'].unique()))

    # Plot the full data for all sizes, distributing the columns across 2 rows with a y-axis range of 0 to 1
    for i, column in enumerate(columns_to_plot):
        row = i // half_columns
        col = i % half_columns
        sns.violinplot(x='size', hue='size', y=column, data=df, palette=palette, ax=axes[row, col], cut=0, legend=False)
        axes[row, col].set_xlabel('Graph Size')
        axes[row, col].set_ylabel('')  # Hide y-axis label
        axes[row, col].set_title(metric_names[column])
        axes[row, col].set_ylim(-0.1, 1.1)  # Set y-axis limits to 0 and 1

    # Remove empty subplot if the number of columns is odd
    if len(columns_to_plot) % 2 != 0:
        fig.delaxes(axes[1, -1])

    # Adjust layout for better spacing
    plt.tight_layout()
    plt.savefig('figures/stimuli_metric_violins.pdf', format='pdf')
    plt.show()

def corrs():
    df = pd.read_csv("cleaned_data/stimuli_metrics.csv")

    # Exclude the "drawing" and "size" columns as per the user's request
    columns_to_plot = df.columns.difference(['drawing', 'size', 'ksm'])


    metric_names = {"ar": "Angular Resolution", "asp": "Aspect Ratio", "ca": "Crossing Angle", "ec": "Edge Crossings", "el":"Edge Lengths", 
                            "eo":"Edge Orthogonality", "gr":"Gabriel Ratio", "np":"Neighbourhood Preservation", "nr":"Node Resolution", "nu":"Node Uniformity"}

    # Calculate correlations
    correlations = {col: df['ksm'].corr(df[col]) for col in columns_to_plot}

    # Map metric names to columns for better display
    renamed_correlations = {metric_names.get(col, col): corr for col, corr in correlations.items()}

    # Convert to a DataFrame for easier plotting
    correlation_df = pd.DataFrame.from_dict(renamed_correlations, orient='index', columns=['Correlation'])

    # Sort the DataFrame by correlation values
    correlation_df = correlation_df.sort_values(by='Correlation', ascending=False)

    # Create a 1-dimensional heatmap
    plt.figure(figsize=(10, 3))
    sns.heatmap(
        correlation_df.T, 
        annot=True, 
        cmap="YlGnBu", 
        cbar=True, 
        yticklabels=False, 
        fmt=".2f"
    )

    plt.title("Correlations of each metric with Kruskal's Stress Metric", fontsize=11)
    plt.xticks(rotation=45, ha='right', fontsize=9)
    plt.tight_layout()
    plt.savefig('figures/stimuli_metric_corrs.pdf', format='pdf')

    plt.show()

def main():
    violins()
    corrs()

if __name__ == "__main__":
    main()