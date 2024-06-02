import sys
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.neural_network import MLPRegressor as model
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
import numpy as np

### Load the dataset
file_path = sys.argv[1]
data = pd.read_csv(file_path)

### Columns for protein expressions, excluding SampleID for modeling. use here the relevant pannel.
protein_expression_columns = [
    'dsDNA', 'Vimentin', 'SMA', 'FoxP3', 'Lag3', 'CD4', 'CD16', 'CD56', 'PD1', 
    'CD31', 'PD-L1', 'EGFR', 'Ki67', 'CD209', 'CD11c', 'CD138', 'CD68', 'CD8', 
    'CD3', 'IDO', 'Keratin17', 'CD63', 'CD45RO', 'CD20', 'p53', 'Beta catenin', 
    'HLA-DR', 'CD11b', 'CD45', 'H3K9ac', 'Pan-Keratin', 'H3K27me3', 'phospho-S6', 
    'MPO', 'Keratin6', 'HLA_Class_1'
] 

shape_features_columns = [
    'area', 'eccentricity', 'major_axis_length', 'minor_axis_length', 
    'perimeter', 'equivalent_diameter_area', 'area_convex', 'extent', 
    'feret_diameter_max', 'orientation', 'perimeter_crofton', 'solidity'
]

# Encoding CellType for modeling
data_encoded = pd.get_dummies(data, columns=['CellType'])

# Extracting relevant data for cell type to shape
X_cell_type = data_encoded.filter(regex='CellType_')
y = data[shape_features_columns]

def normalize_protein_expressions(protein_data):
    scaler = StandardScaler()
    normalized_data = scaler.fit_transform(protein_data)
    return normalized_data

X_protein = data[protein_expression_columns]
X_protein_normalized = normalize_protein_expressions(X_protein)

def combine_features(normalized_protein_data, cell_type_data):
    combined_data = np.hstack((normalized_protein_data, cell_type_data))
    return combined_data

X_combined = combine_features(X_protein_normalized, X_cell_type)

def calculate_cv_r2_scores(X, y, model):
    def check_data_type(data):
        if isinstance(data, pd.DataFrame):
            return data.values
        return data

    kf = KFold(n_splits=3, shuffle=True, random_state=42)
    r2_scores = []

    X = check_data_type(X)
    y = check_data_type(y)

    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        r2_scores.append(r2_score(y_test, y_pred, multioutput='variance_weighted'))

    return r2_scores

def evaluate_linear_model(cell_type_data, combined_data, target_data):
    r2_scores_linear_ct = calculate_cv_r2_scores(cell_type_data, target_data, LinearRegression())
    r2_scores_linear_combined = calculate_cv_r2_scores(combined_data, target_data, LinearRegression())

    print("Linear Model R2 Scores (CellType to Shape):", np.mean(r2_scores_linear_ct))
    print("Linear Model R2 Scores (CellType + Proteins to Shape):", np.mean(r2_scores_linear_combined))

if __name__ == 'main':
    
    evaluate_linear_model(X_cell_type, X_combined, y)

