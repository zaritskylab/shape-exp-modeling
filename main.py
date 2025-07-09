#!/usr/bin/env python3
"""
Main script for Shape-Expression Modeling (Bidirectional)
Supports both directions: shape2pro (shape -> protein) and pro2shape (protein -> shape)

Usage:
    python main.py --data_path ONLY_CELLS.csv --direction shape2pro --output_dir ./results
    python main.py --data_path ONLY_CELLS.csv --direction pro2shape --output_dir ./results
"""

import argparse
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')


def setup_args():
    """Setup command line arguments"""
    parser = argparse.ArgumentParser(description='Shape-Expression Modeling (Bidirectional)')
    
    # Data arguments
    parser.add_argument('--data_path', type=str, default='ProcessedCellsTNBC_sample.csv',
                       help='Path to processed CSV file')
    parser.add_argument('--direction', type=str, choices=['shape2pro', 'pro2shape'], 
                       default='shape2pro',
                       help='Direction of modeling: shape2pro or pro2shape')
    
    # Model arguments
    parser.add_argument('--model', type=str, choices=['ridge', 'mlp'], default='ridge',
                       help='Model type: ridge (Ridge regression) or mlp (MLP regressor)')
    parser.add_argument('--test_size', type=float, default=0.2,
                       help='Test set size (default: 0.2)')
    parser.add_argument('--random_state', type=int, default=42,
                       help='Random state for reproducibility')
    
    # Ridge-specific arguments
    parser.add_argument('--alpha', type=float, default=0.5,
                       help='Ridge regression alpha parameter')
    
    # MLP-specific arguments
    parser.add_argument('--hidden_layer_sizes', type=str, default='100,50',
                       help='MLP hidden layer sizes (comma-separated, e.g., "100,50")')
    parser.add_argument('--max_iter', type=int, default=1000,
                       help='MLP maximum iterations')
    parser.add_argument('--learning_rate_init', type=float, default=0.001,
                       help='MLP initial learning rate')
    
    # Output arguments
    parser.add_argument('--output_dir', type=str, default='./results',
                       help='Output directory for results and plots')
    parser.add_argument('--save_detailed', action='store_true',
                       help='Save detailed per-patient results')
    
    # Analysis arguments
    parser.add_argument('--min_cells_per_patient', type=int, default=2,
                       help='Minimum cells per patient for analysis')
    parser.add_argument('--plot_format', type=str, default='png', choices=['png', 'svg', 'pdf'],
                       help='Plot output format')
    parser.add_argument('--dpi', type=int, default=300,
                       help='Plot DPI')
    
    return parser.parse_args()


def load_and_validate_data(data_path):
    """Load and validate the CSV data"""
    print(f"Loading data from: {data_path}")
    
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found: {data_path}")
    
    df = pd.read_csv(data_path)
    print(f"Loaded dataset with shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    
    # Validate required columns
    if 'SampleID' not in df.columns:
        raise ValueError("Missing 'SampleID' column in dataset!")
    
    if 'CellType' not in df.columns:
        raise ValueError("Missing 'CellType' column in dataset!")
    
    return df


def define_feature_groups(df):
    """Define shape features and functional markers"""
    
    # Shape features (commonly available morphological features)
    shape_features = [
        'cellSize', 'minor_axis_length', 'perimeter', 
        'equivalent_diameter_area', 'extent', 'feret_diameter_max', 
        'orientation', 'perimeter_crofton', 'solidity'
    ]
    
    # Functional markers (protein expression)
    functional_markers = [
        'dsDNA', 'Vimentin', 'SMA', 'FoxP3', 'Lag3', 'CD4', 'CD16', 'CD56', 'PD1',
        'CD31', 'PD-L1', 'EGFR', 'Ki67', 'CD209', 'CD11c', 'CD138', 'CD68',
        'CD8', 'CD3', 'IDO', 'Keratin17', 'CD63', 'CD45RO', 'CD20', 'p53',
        'Beta catenin', 'HLA-DR', 'CD11b', 'CD45', 'H3K9ac', 'Pan-Keratin',
        'H3K27me3', 'phospho-S6', 'MPO', 'Keratin6', 'HLA_Class_1'
    ]
    
    # Filter features that actually exist in the dataset
    available_shape = [f for f in shape_features if f in df.columns]
    available_markers = [f for f in functional_markers if f in df.columns]
    
    print(f"Available shape features ({len(available_shape)}): {available_shape}")
    print(f"Available functional markers ({len(available_markers)}): {available_markers}")
    
    if not available_shape:
        raise ValueError("No shape features found in dataset!")
    if not available_markers:
        raise ValueError("No functional markers found in dataset!")
    
    return available_shape, available_markers


def prepare_datasets(df, shape_features, functional_markers, direction):
    """Prepare feature and target datasets based on direction"""
    
    # One-hot encode cell types
    cell_type_ohe = pd.get_dummies(df['CellType'], prefix='CellType')
    print(f"Cell types encoded: {cell_type_ohe.shape[1]} categories")
    
    if direction == 'shape2pro':
        # Predicting protein expression from shape
        X_base = cell_type_ohe  # Model 1: Cell Type Only
        X_enhanced = pd.concat([X_base, df[shape_features]], axis=1)  # Model 2: Cell Type + Shape
        y = df[functional_markers]  # Target: functional markers
        
        print("Direction: Shape → Protein Expression")
        print(f"Base features (cell type only): {X_base.shape[1]}")
        print(f"Enhanced features (cell type + shape): {X_enhanced.shape[1]}")
        print(f"Targets (proteins): {y.shape[1]}")
        
    elif direction == 'pro2shape':
        # Predicting shape from protein expression
        X_base = cell_type_ohe  # Model 1: Cell Type Only
        X_enhanced = pd.concat([X_base, df[functional_markers]], axis=1)  # Model 2: Cell Type + Proteins
        y = df[shape_features]  # Target: shape features
        
        print("Direction: Protein Expression → Shape")
        print(f"Base features (cell type only): {X_base.shape[1]}")
        print(f"Enhanced features (cell type + proteins): {X_enhanced.shape[1]}")
        print(f"Targets (shape): {y.shape[1]}")
    
    return X_base, X_enhanced, y


def adjusted_r2_score(y_true, y_pred, n_features):
    """Calculate adjusted R² score"""
    n_samples = y_true.shape[0]
    r2 = r2_score(y_true, y_pred, multioutput='variance_weighted')
    
    # Adjusted R² formula: 1 - (1 - R²) * (n - 1) / (n - p - 1)
    # where n = number of samples, p = number of features
    if n_samples <= n_features + 1:
        return r2  # Return regular R² if not enough samples
    
    adjusted_r2 = 1 - (1 - r2) * (n_samples - 1) / (n_samples - n_features - 1)
    return adjusted_r2


def create_model(model_type, args):
    """Create model based on type and arguments"""
    if model_type == 'ridge':
        return Ridge(alpha=args.alpha, random_state=args.random_state)
    
    elif model_type == 'mlp':
        # Parse hidden layer sizes
        hidden_sizes = tuple(map(int, args.hidden_layer_sizes.split(',')))
        
        return MLPRegressor(
            hidden_layer_sizes=hidden_sizes,
            max_iter=args.max_iter,
            learning_rate_init=args.learning_rate_init,
            random_state=args.random_state,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=20
        )
    
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def train_and_evaluate(X, y, model_type, args):
    """Trains a model and computes adjusted R²"""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.random_state
    )
    
    # Create model
    model = create_model(model_type, args)
    
    # For MLP, we need to scale the features
    if model_type == 'mlp':
        scaler_X = StandardScaler()
        scaler_y = StandardScaler()
        
        X_train_scaled = scaler_X.fit_transform(X_train)
        X_test_scaled = scaler_X.transform(X_test)
        y_train_scaled = scaler_y.fit_transform(y_train)
        
        # Train model
        model.fit(X_train_scaled, y_train_scaled)
        
        # Predict and inverse transform
        y_pred_scaled = model.predict(X_test_scaled)
        y_pred = scaler_y.inverse_transform(y_pred_scaled)
        
    else:  # Ridge
        # Train model
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
    
    # Calculate adjusted R²
    n_features = X.shape[1]
    adj_r2 = adjusted_r2_score(y_test, y_pred, n_features)
    
    return adj_r2, model


def analyze_global_performance(X_base, X_enhanced, y, args):
    """Analyze global model performance"""
    print("\n" + "="*50)
    print("GLOBAL ANALYSIS")
    print("="*50)
    
    print(f"Using model: {args.model.upper()}")
    
    # Train models
    adj_r2_base_global, model_base = train_and_evaluate(X_base, y, args.model, args)
    adj_r2_enhanced_global, model_enhanced = train_and_evaluate(X_enhanced, y, args.model, args)
    
    improvement = adj_r2_enhanced_global - adj_r2_base_global
    
    print(f'Global Adjusted R² - Base Model: {adj_r2_base_global:.4f}')
    print(f'Global Adjusted R² - Enhanced Model: {adj_r2_enhanced_global:.4f}')
    print(f'Global Improvement: {improvement:.4f} ({improvement/abs(adj_r2_base_global)*100:.1f}%)')
    
    return adj_r2_base_global, adj_r2_enhanced_global, model_base, model_enhanced


def analyze_per_patient_performance(df, X_base, X_enhanced, y, args):
    """Analyze per-patient model performance"""
    print("\n" + "="*50)
    print("PER-PATIENT ANALYSIS")
    print("="*50)
    
    results = []
    patients = df['SampleID'].unique()
    print(f"Analyzing {len(patients)} patients...")
    
    for patient in patients:
        patient_data = df[df['SampleID'] == patient]
        
        # Skip if not enough data
        if len(patient_data) < args.min_cells_per_patient:
            print(f"Skipping patient {patient}: only {len(patient_data)} cells")
            continue
        
        # Get patient-specific data
        X_base_patient = X_base.loc[patient_data.index]
        X_enhanced_patient = X_enhanced.loc[patient_data.index]
        y_patient = y.loc[patient_data.index]
        
        try:
            adj_r2_base, _ = train_and_evaluate(X_base_patient, y_patient, args.model, args)
            adj_r2_enhanced, _ = train_and_evaluate(X_enhanced_patient, y_patient, args.model, args)
            
            results.append({
                'SampleID': patient,
                'Adj_R2_Base': adj_r2_base,
                'Adj_R2_Enhanced': adj_r2_enhanced,
                'Delta_Adj_R2': adj_r2_enhanced - adj_r2_base,
                'Improved': adj_r2_enhanced > adj_r2_base,
                'Cells_Count': len(patient_data)
            })
            
        except Exception as e:
            print(f"Error processing patient {patient}: {e}")
            continue
    
    # Convert results to DataFrame
    results_df = pd.DataFrame(results)
    
    if len(results_df) == 0:
        print("No patients could be analyzed!")
        return results_df
    
    # Summary statistics
    improved_count = results_df['Improved'].sum()
    mean_improvement = results_df['Delta_Adj_R2'].mean()
    
    print(f'Total patients analyzed: {len(results_df)}')
    print(f'Patients improved with enhanced model: {improved_count} ({improved_count/len(results_df)*100:.1f}%)')
    print(f'Mean improvement: {mean_improvement:.4f}')
    print(f'Std improvement: {results_df["Delta_Adj_R2"].std():.4f}')
    
    return results_df


def create_visualizations(results_df, args):
    """Create and save visualization plots"""
    print("\n" + "="*50)
    print("CREATING VISUALIZATIONS")
    print("="*50)
    
    if len(results_df) == 0:
        print("No data to visualize!")
        return
    
    # Set style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # 1. Scatter plot: Base vs Enhanced Adjusted R²
    plt.figure(figsize=(8, 8))
    plt.scatter(results_df['Adj_R2_Base'], results_df['Adj_R2_Enhanced'], 
               alpha=0.7, s=60, color='steelblue', edgecolors='black', linewidth=0.5)
    
    # Add diagonal line
    min_r2 = min(results_df['Adj_R2_Base'].min(), results_df['Adj_R2_Enhanced'].min())
    max_r2 = max(results_df['Adj_R2_Base'].max(), results_df['Adj_R2_Enhanced'].max())
    plt.plot([min_r2, max_r2], [min_r2, max_r2], 'r--', linewidth=2, alpha=0.8)
    
    plt.xlabel('Base Model Adjusted R²', fontsize=12)
    plt.ylabel('Enhanced Model Adjusted R²', fontsize=12)
    
    direction_label = "Shape → Protein" if args.direction == 'shape2pro' else "Protein → Shape"
    model_label = args.model.upper()
    plt.title(f'Adjusted R² Comparison per Patient\n({direction_label}, {model_label})', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    filename = f'adj_r2_comparison_{args.direction}_{args.model}.{args.plot_format}'
    plt.savefig(os.path.join(args.output_dir, filename), 
                dpi=args.dpi, bbox_inches='tight')
    plt.close()
    print(f"Saved: {filename}")
    
    # 2. Improvement distribution
    plt.figure(figsize=(10, 6))
    plt.hist(results_df['Delta_Adj_R2'], bins=20, alpha=0.7, color='lightgreen', 
             edgecolor='black', linewidth=0.5)
    plt.axvline(0, color='red', linestyle='--', linewidth=2, alpha=0.8)
    plt.xlabel('Adjusted R² Improvement (Enhanced - Base)', fontsize=12)
    plt.ylabel('Number of Patients', fontsize=12)
    plt.title(f'Distribution of Adjusted R² Improvements\n({direction_label}, {model_label})', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    filename = f'adj_r2_improvement_distribution_{args.direction}_{args.model}.{args.plot_format}'
    plt.savefig(os.path.join(args.output_dir, filename), 
                dpi=args.dpi, bbox_inches='tight')
    plt.close()
    print(f"Saved: {filename}")
    
    # 3. Per-patient comparison (if reasonable number of patients)
    if len(results_df) <= 50:  # Only plot if manageable number
        plt.figure(figsize=(14, 8))
        x_pos = np.arange(len(results_df))
        width = 0.35
        
        plt.bar(x_pos - width/2, results_df['Adj_R2_Base'], width, 
                label='Base Model', alpha=0.7, color='lightcoral')
        plt.bar(x_pos + width/2, results_df['Adj_R2_Enhanced'], width,
                label='Enhanced Model', alpha=0.7, color='lightblue')
        
        plt.xlabel('Patient ID', fontsize=12)
        plt.ylabel('Adjusted R² Score', fontsize=12)
        plt.title(f'Per-Patient Adjusted R² Scores\n({direction_label}, {model_label})', fontsize=14)
        plt.xticks(x_pos, results_df['SampleID'], rotation=45)
        plt.legend()
        plt.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        
        filename = f'per_patient_adj_r2_{args.direction}_{args.model}.{args.plot_format}'
        plt.savefig(os.path.join(args.output_dir, filename), 
                    dpi=args.dpi, bbox_inches='tight')
        plt.close()
        print(f"Saved: {filename}")


def save_detailed_results(results_df, global_results, args):
    """Save detailed results to CSV files"""
    print("\n" + "="*50)
    print("SAVING DETAILED RESULTS")
    print("="*50)
    
    # Save per-patient results
    if len(results_df) > 0:
        patient_filename = f'per_patient_results_{args.direction}_{args.model}.csv'
        results_df.to_csv(os.path.join(args.output_dir, patient_filename), index=False)
        print(f"Saved per-patient results: {patient_filename}")
    
    # Save global summary
    summary = {
        'direction': args.direction,
        'model': args.model,
        'global_adj_r2_base': global_results[0],
        'global_adj_r2_enhanced': global_results[1],
        'global_improvement': global_results[1] - global_results[0],
        'test_size': args.test_size,
        'random_state': args.random_state
    }
    
    # Add model-specific parameters
    if args.model == 'ridge':
        summary['alpha'] = args.alpha
    elif args.model == 'mlp':
        summary['hidden_layer_sizes'] = args.hidden_layer_sizes
        summary['max_iter'] = args.max_iter
        summary['learning_rate_init'] = args.learning_rate_init
    
    # Add per-patient statistics if available
    if len(results_df) > 0:
        summary.update({
            'patients_analyzed': len(results_df),
            'patients_improved': results_df['Improved'].sum(),
            'improvement_rate': results_df['Improved'].mean(),
            'mean_improvement': results_df['Delta_Adj_R2'].mean(),
            'std_improvement': results_df['Delta_Adj_R2'].std(),
            'min_improvement': results_df['Delta_Adj_R2'].min(),
            'max_improvement': results_df['Delta_Adj_R2'].max()
        })
    
    summary_df = pd.DataFrame([summary])
    summary_filename = f'global_summary_{args.direction}_{args.model}.csv'
    summary_df.to_csv(os.path.join(args.output_dir, summary_filename), index=False)
    print(f"Saved global summary: {summary_filename}")


def main():
    """Main function"""
    args = setup_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("=" * 70)
    print("SHAPE-EXPRESSION MODELING (BIDIRECTIONAL)")
    print("=" * 70)
    print(f"Direction: {args.direction}")
    print(f"Model: {args.model.upper()}")
    print(f"Data: {args.data_path}")
    print(f"Output: {args.output_dir}")
    
    # Load and validate data
    df = load_and_validate_data(args.data_path)
    
    # Define feature groups
    shape_features, functional_markers = define_feature_groups(df)
    
    # Prepare datasets based on direction
    X_base, X_enhanced, y = prepare_datasets(df, shape_features, functional_markers, args.direction)
    
    # Global analysis
    global_results = analyze_global_performance(X_base, X_enhanced, y, args)
    
    # Per-patient analysis
    results_df = analyze_per_patient_performance(df, X_base, X_enhanced, y, args)
    
    # Create visualizations
    create_visualizations(results_df, args)
    
    # Save detailed results if requested
    if args.save_detailed:
        save_detailed_results(results_df, global_results, args)
    
    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE!")
    print(f"Results saved in: {args.output_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()