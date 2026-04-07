# DVC Setup Guide - Telco Customer Churn Project

This guide explains how to use DVC (Data Version Control) in this project for data and model versioning.

## Installation

DVC has been added to `requirements.txt`. Install it with:

```bash
pip install -r requirements.txt
```

## Key DVC Configuration Files

### `.dvc/config`
Contains remote storage configuration. Currently set to use local cache storage.
To use cloud storage (S3, Google Drive, Azure, etc.), update this file:

**Example: AWS S3**
```ini
['remote "myremote"']
    url = s3://my-bucket/dvc-storage
[core]
    remote = myremote
```

**Example: Google Drive**
```ini
['remote "myremote"']
    url = gdrive://folder-id
[core]
    remote = myremote
```

### `dvc.yaml`
Defines your ML pipeline stages and dependencies. Currently configured to track:
- **Data**: CSV dataset and prepared pkl files
- **Models**: Scalers and feature columns
- **Artifacts**: MLflow experiment directory

### `params.yaml`
Stores all hyperparameters and configuration. Changes here are tracked by DVC.

### `.dvcignore`
Specifies patterns of files to ignore (similar to .gitignore).

## Common DVC Commands

### Track Data Files
```bash
# Add a data file to DVC
dvc add WA_Fn-UseC_-Telco-Customer-Churn.csv

# This creates WA_Fn-UseC_-Telco-Customer-Churn.csv.dvc file
# Commit this .dvc file to git
git add WA_Fn-UseC_-Telco-Customer-Churn.csv.dvc
git commit -m "Track churn dataset with DVC"
```

### Track Models and Artifacts
```bash
# Track the scaler
dvc add scaler.pkl

# Track the feature columns
dvc add feature_columns.pkl

# Track mlruns directory
dvc add mlruns
```

### Run Pipeline
```bash
# Run all stages defined in dvc.yaml
dvc repro

# Run specific stage
dvc repro prepare

# Run with specific parameters
dvc repro --param-set train.test_size=0.25
```

### View Pipeline Dependencies
```bash
# Show pipeline structure
dvc dag

# Show detailed pipeline information
dvc pipeline show
```

### Push/Pull Data to Remote
```bash
# Push tracked files to remote storage
dvc push

# Pull tracked files from remote storage
dvc pull

# Get status of remote
dvc status
```

### Experiments Tracking (DVC + MLflow)
```bash
# View all experiments
dvc exp show

# Run experiment with different parameters
dvc exp run -S train.test_size=0.3

# Compare experiments
dvc plots diff
```

## Best Practices

1. **Track Data Files**: Always use DVC to track large data files and models
   ```bash
   dvc add large_dataset.csv
   ```

2. **Commit .dvc Files**: Add .dvc metadata files to Git, not the actual data
   ```bash
   git add *.dvc
   git commit -m "Track data versions"
   ```

3. **Use Remote Storage**: For collaboration, configure a remote
   ```bash
   dvc remote add -d myremote s3://bucket/path
   dvc push  # Share with team
   ```

4. **Version Parameters**: Keep all hyperparameters in params.yaml
   - DVC tracks changes automatically
   - Makes experiments reproducible

5. **Reproducible Pipelines**: Use dvc.yaml to define stages
   - Each stage has clear inputs/outputs
   - Run `dvc repro` to ensure everything is current

## Integration with MLflow

- MLflow tracks experiments (models, metrics, parameters)
- DVC tracks data and pipeline reproducibility
- Together they provide complete ML project tracking:
  - **Data lineage**: DVC shows what data was used
  - **Experiment history**: MLflow shows model performance
  - **Reproducibility**: Both ensure experiments can be recreated

## Troubleshooting

### Reset DVC cache
```bash
dvc gc --workspace --not-in-remote
```

### Check what DVC is tracking
```bash
dvc status
```

### View DVC logs
```bash
dvc version
dvc status -c  # Check cache status
```

## Next Steps

1. Configure remote storage (S3, Google Drive, etc.)
2. Run `dvc add` for your primary datasets
3. Create pipeline with `dvc repro`
4. Use DVC experiments for hyperparameter tuning
5. Share DVC cache with team via remote storage
