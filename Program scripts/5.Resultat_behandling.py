import pandas as pd

# Define the function to transform and compare the results
def transform_and_compare(results_path, labels_path, transformed_output_path, comparison_output_path):
    # Load the original results and labels data
    results_df = pd.read_csv(results_path)
    labels_df = pd.read_csv(labels_path)

    # Extract image numbers from image paths
    results_df['Image Number'] = results_df['Image'].str.extract('(\d+)').astype(int)

    # Mapping of tile types to their respective score columns
    tile_type_to_column = {
        'Lake': 'Lake Score',
        'Forest': 'Forest Score',
        'Field': 'Field Score',
        'Grassland': 'Grassland Score',
        'Swamp': 'Swamp Score',
        'Mine': 'Mine Score'
    }
    
    # Pivot the results dataframe based on image number and tile type
    pivot_df = results_df.pivot_table(index='Image Number', columns='Tile Type', values='Score', aggfunc='sum')
    
    # Rename columns according to the mapping and fill missing columns with zeros
    pivot_df = pivot_df.rename(columns=tile_type_to_column)
    for column in labels_df.columns[1:-1]:
        if column not in pivot_df.columns:
            pivot_df[column] = 0
    pivot_df = pivot_df[labels_df.columns[1:-1]]
    
    # Calculate total score
    pivot_df['Total score'] = pivot_df.sum(axis=1)
    pivot_df.reset_index(inplace=True)
    
    # Reorder and type-cast to ensure matching formats
    pivot_df = pivot_df[labels_df.columns].fillna(0).astype(int)
    
    # Save the transformed dataframe
    pivot_df.to_csv(transformed_output_path, index=False)

    # Load transformed results and original labels again to ensure both are synchronized in format
    transformed_df = pd.read_csv(transformed_output_path)
    labels_df = pd.read_csv(labels_path)

    # Merge dataframes on 'Image Number' to compare corresponding entries only
    merged_df = pd.merge(transformed_df, labels_df, on='Image Number', suffixes=('_trans', '_orig'))

    # Compare and mark differences
    comparison_df = merged_df[['Image Number']].copy()
    for column in labels_df.columns[1:]:  # Start from 1 to skip 'Image Number'
        comparison_df[f'{column} Comparison'] = merged_df[f'{column}_trans'].eq(merged_df[f'{column}_orig'])

    # Mark non-matching cells with a specific flag (e.g., False for mismatches)
    comparison_df.replace({True: '', False: 'MISMATCH'}, inplace=True)

    # Save the comparison results
    comparison_df.to_csv(comparison_output_path, index=False)

transform_and_compare(
    # "CSV_filer/Resultater.csv",
    "Program scripts/Resultater3.csv",
    "CSV_filer/Labels.csv",
    "CSV_filer/Transformerede_resultater2.csv",
    "CSV_filer/Sammenligning_af_resultater2.csv"
)
