# Data Directory

This directory contains the code to generate the TNBC csv for the full Shape-Expression Modeling framework.

### Data
- **CellData**: Cells table, with normalized expression per segment.
- **idx_to_cell_type**: Mapper, the cell type classification per segment.
- **Raw imaging data**: Download and put in the same dir. Available at [Link](https://mibi-share.ionpath.com/), from the paper [Link](https://pmc.ncbi.nlm.nih.gov/articles/PMC6785247/)

## Processing Example

To convert raw data to processed format, see the example in `Shape2Exp/ProcessData.py`:

```python
import pandas as pd
import numpy as np
from tqdm import tqdm
from Shape2Exp.ProcessData import CellsDataSetTNBC

# Process raw data to extract features
processed_data = CellsDataSetTNBC(
    data_path='data/',
    cells_data_df=raw_df,
    types_data_df=types_df,
    mode='baseline_morph'
)

# Lists to collect rows
rows = []

# Loop through all samples
for i in tqdm(range(len(processed_data))):
    X, Y, (patientID, cellIndex) = processed_data[i]
    
    # Flatten everything into a 1D row
    row = np.concatenate([
        [patientID, cellIndex],  # metadata
        X.flatten(),             # input features
        Y.flatten()              # output targets
    ])
    
    rows.append(row)

# Convert to DataFrame
df = pd.DataFrame(rows)
shape_features = [
        'cellSize', 'minor_axis_length', 'perimeter', 
        'equivalent_diameter_area', 'extent', 'feret_diameter_max', 
        'orientation', 'perimeter_crofton', 'solidity'
    ]
    
# protein expression
markers = [
        'dsDNA', 'Vimentin', 'SMA', 'FoxP3', 'Lag3', 'CD4', 'CD16', 'CD56', 'PD1',
        'CD31', 'PD-L1', 'EGFR', 'Ki67', 'CD209', 'CD11c', 'CD138', 'CD68',
        'CD8', 'CD3', 'IDO', 'Keratin17', 'CD63', 'CD45RO', 'CD20', 'p53',
        'Beta catenin', 'HLA-DR', 'CD11b', 'CD45', 'H3K9ac', 'Pan-Keratin',
        'H3K27me3', 'phospho-S6', 'MPO', 'Keratin6', 'HLA_Class_1'
    ]
num_X = len(X)
num_Y = len(Y)
columns = ['SampleID', 'CellIndex'] + markers + shape_features
df.columns = columns

# Save to CSV
df.to_csv('ProcessedCellsTNBC.csv', index=False)

```
