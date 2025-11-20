#!/usr/bin/env python3
"""Quick script to check MLflow metrics"""

import mlflow
import mlflow.tracking
from mlflow.tracking import MlflowClient
import json
import os

# Set tracking URI
mlflow.set_tracking_uri("file:./mlruns")

# Get client
client = MlflowClient()

# Search experiments
experiments = client.search_experiments()

print("=" * 80)
print("MLflow Experiments and Metrics")
print("=" * 80)

for exp in experiments:
    print(f"\nExperiment: {exp.name} (ID: {exp.experiment_id})")
    print("-" * 80)
    
    # Get runs for this experiment
    runs = client.search_runs([exp.experiment_id], max_results=10)
    
    if not runs:
        print("  No runs found")
        continue
    
    for i, run in enumerate(runs, 1):
        print(f"\n  Run {i}: {run.info.run_id}")
        print(f"    Status: {run.info.status}")
        print(f"    Start Time: {run.info.start_time}")
        
        # Print metrics
        if run.data.metrics:
            print("    Metrics:")
            for key, value in run.data.metrics.items():
                if 'sharpe' in key.lower() or 'return' in key.lower() or 'drawdown' in key.lower() or 'profit' in key.lower() or 'win' in key.lower():
                    print(f"      {key}: {value}")
        
        # Print parameters
        if run.data.params:
            print("    Key Parameters:")
            for key, value in list(run.data.params.items())[:5]:
                print(f"      {key}: {value}")

# Also check the performance_metrics.json file
print("\n" + "=" * 80)
print("Performance Metrics File")
print("=" * 80)

metrics_file = "results/performance_metrics.json"
if os.path.exists(metrics_file):
    with open(metrics_file, 'r') as f:
        metrics = json.load(f)
    print(json.dumps(metrics, indent=2))
else:
    print(f"File not found: {metrics_file}")

