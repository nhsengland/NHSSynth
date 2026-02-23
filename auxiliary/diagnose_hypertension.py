"""
Diagnostic script to investigate hypertension synthetic data issues
"""

import os
import sys

import pandas as pd

# Add the src directory to path
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "../src")))

from nhssynth.modules.dataloader.metadata import MetaData
from nhssynth.modules.dataloader.metatransformer import MetaTransformer

print("=" * 80)
print("DIAGNOSTIC: Hypertension Synthetic Data Issues")
print("=" * 80)

# Load original dataset
dataset = pd.read_csv("../data/hypertension_synthetic.csv")
dataset["Date_of_Birth"] = pd.to_datetime(dataset["Date_of_Birth"], errors="coerce")
dataset["Medication_Start_Date"] = pd.to_datetime(dataset["Medication_Start_Date"], errors="coerce")
dataset["Last_Followup_Date"] = pd.to_datetime(dataset["Last_Followup_Date"], errors="coerce")

print("\n1. ORIGINAL DATASET")
print(f"   Shape: {dataset.shape}")
print(f"   Columns: {list(dataset.columns)}")
print("\n   Column types:")
for col in dataset.columns:
    print(f"   - {col}: {dataset[col].dtype}, unique values: {dataset[col].nunique()}")

# Load metadata and create transformer
md = MetaData.from_path(dataset, "../data/hypertension_metadata.yaml")
mt = MetaTransformer(dataset, md)

print("\n2. METADATA CONFIGURATION")
print("   Categorical columns:")
for col_name, col_meta in md._metadata.items():
    if col_meta.categorical:
        print(
            f"   - {col_name}: categorical={col_meta.categorical}, dtype={col_meta.dtype}, boolean={col_meta.boolean}"
        )

# Apply transformation
transformed_dataset = mt.apply()

print("\n3. TRANSFORMED DATASET (after mt.apply())")
print(f"   Shape: {transformed_dataset.shape}")
print(f"   Column count change: {dataset.shape[1]} → {transformed_dataset.shape[1]}")
print("   This is EXPECTED due to one-hot encoding of categorical variables")
print("\n   First few transformed columns:")
print(f"   {list(transformed_dataset.columns[:10])}")

# Check for any NaN values in transformed data
print("\n4. NaN CHECK - Transformed Dataset")
nan_cols = transformed_dataset.columns[transformed_dataset.isna().any()].tolist()
if nan_cols:
    print(f"   WARNING: Transformed dataset has NaN in columns: {nan_cols}")
    for col in nan_cols:
        nan_count = transformed_dataset[col].isna().sum()
        print(f"   - {col}: {nan_count} NaN values ({nan_count/len(transformed_dataset)*100:.1f}%)")
else:
    print("   ✓ No NaN values in transformed dataset")

# Check constraints on original data
print("\n5. CONSTRAINT VALIDATION ON ORIGINAL DATA")
print("   Binary variable ranges (should all be 0 or 1):")
binary_cols = [
    "Gender",
    "Hypertension_Diagnosis",
    "Diabetes",
    "Heart_Disease",
    "Obesity",
    "Chronic_Kidney_Disease",
    "Family_History_Hypertension",
]
for col in binary_cols:
    if col in dataset.columns:
        unique_vals = sorted(dataset[col].dropna().unique())
        print(f"   - {col}: {unique_vals}")

# Now let's check if we can generate and what happens
print("\n6. ATTEMPTING SMALL GENERATION TEST")
print("   (This will help us see where issues occur)")

try:
    from nhssynth.modules.model.models import VAE

    # Create a small model just for testing
    model = VAE(transformed_dataset, mt)

    print("   Model created successfully")
    print(f"   - Input dim: {model.input_dim}")
    print(f"   - Latent dim: {model.latent_dim}")

    # Don't train, just try to generate from random latent codes
    # This will help us see the inverse transformation process
    print("\n   Skipping training for diagnostic purposes...")
    print("   (In real usage, you would train first)")

except Exception as e:
    print(f"   ERROR during model creation: {e}")
    import traceback

    traceback.print_exc()

print(f"\n{'='*80}")
print("DIAGNOSTIC COMPLETE")
print(f"{'='*80}")
print("\nKEY FINDINGS:")
print(f"1. Column count mismatch (22→{transformed_dataset.shape[1]}) is EXPECTED")
print("   - Categorical variables are one-hot encoded during transformation")
print("   - This is necessary for the VAE to process them")
print("   - They get reverted back to original form during generation")
print("\n2. To diagnose other issues, we need to see the actual synthetic data")
print("   - Run the full training and generation in mwe_hypertension.ipynb")
print("   - Then check the synthetic_dataset for:")
print("     * NaN values (causing t-SNE error)")
print("     * Continuous values in categorical columns (causing constraint violations)")
print("     * Distribution of values (checking for clipping artifacts)")
