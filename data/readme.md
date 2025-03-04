# Data Directory

This directory is for storing your longitudinal datasets.

## Directory Structure

```
data/
├── raw/           # Original, immutable data files
├── processed/     # Cleaned and processed datasets
├── sample/        # Sample datasets for testing
└── metadata.json  # Metadata about datasets and transformations
```

## Using Your Own Data

To use your own longitudinal datasets:

1. Place your CSV files in the `raw/` directory
2. Create a metadata file describing transformations (optional)
3. Use the `--data` flag when running the application:

```bash
# Web interface
streamlit run app.py -- --data ./data/raw

# CLI
python -m src.cli interactive --data ./data/raw
```

## Metadata File Format

The metadata file should be in JSON format and contain information about dataset versions and column evolution:

```json
{
  "dataset_versions": [
    {
      "dataset_name": "your_dataset",
      "version": "1.0",
      "parent_versions": [],
      "transformations": [
        {
          "operation": "rename_columns",
          "parameters": {"old_name": "new_name"},
          "rationale": "For consistency",
          "timestamp": "2023-01-01T00:00:00"
        }
      ]
    }
  ],
  "column_evolutions": [
    {
      "original_name": "old_column",
      "dataset": "your_dataset",
      "versions": [
        {
          "name": "new_column",
          "transformation": "renamed",
          "reason": "Standardization",
          "timestamp": "2023-01-01T00:00:00"
        }
      ]
    }
  ]
}
```

## Sample Datasets

The application comes with sample medical datasets for demonstration:

- `patient_demographics_v1.csv` - Basic patient information (original)
- `patient_demographics_v2.csv` - Updated patient information with renamed columns
- `patient_visits.csv` - Record of clinical visits
- `patient_outcomes_v1.csv` - Clinical outcomes by visit (original)
- `patient_outcomes_v2.csv` - Updated outcomes with derived metrics
