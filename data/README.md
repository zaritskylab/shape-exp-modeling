# Data Directory

This directory contains datasets for the Shape-Expression Modeling framework.

## Data Types

### Processed Data (Ready-to-Use)
- **File**: `ProcessedCellsTNBC.csv`
- **Description**: Out-of-the-box demo dataset ready for immediate analysis
- **Usage**: Can be used directly with the main analysis pipeline

```bash
python main.py --data_path data/ProcessedCellsTNBC.csv --direction shape2pro
```

### Raw Data
- **CellDataDemo**: Cells table, with normalized expression per segment.
- **idx_to_cell_type**: Mapper, the cell type classification per segment.
- **Raw imaging data**: Download and put in the same dir. Available at [Google Drive](https://drive.google.com/drive/folders/1HyPIAKVM44XB4ef_h-1_CQiGHmveWTID?usp=sharing)

## Processing Example

To convert raw data to processed format, see the example in `Shape2Exp/ProcessData.py`:
צ
```python
from Shape2Exp.ProcessData import CellsDataSetTNBC

# Process raw data to extract features
processed_data = CellsDataSetTNBC(
    data_path='data/',
    cells_data_df=raw_df,
    types_data_df=types_df,
    mode='neighbors_morph'
)
```
