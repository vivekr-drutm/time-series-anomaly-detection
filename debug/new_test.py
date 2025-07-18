#!/usr/bin/env python
"""
Enhanced test script that displays which models are being used in the pipeline.
"""

import pandas as pd
import matplotlib.pyplot as plt
from darts import TimeSeries

# Import the refactored modules
from forecasters import LinearRegression
from scorers import EuclideanScorer
from thresholds import StaticThreshold
from meta_anomaly_detector import NABAnomalyDetector
from nab_utils import distort_time_series


def quick_test_with_model_info():
    """Quick test to verify the refactored code works with model info display"""

    print("Quick Test of Refactored Anomaly Detection System")
    print("=" * 50)

    # 1. Load data
    print("\n1. Loading data...")
    try:
        df = pd.read_csv("../data/nab_data_corpus/artificialWithAnomaly/artificialWithAnomaly/art_daily_jumpsdown.csv")
        print("   ✓ Data loaded successfully")
    except Exception as e:
        print(f"   ✗ Error loading data: {e}")
        return

    # 2. Create synthetic anomalies
    print("\n2. Creating synthetic anomalies...")
    anomaly_types = ['severe_drop', 'critical_drop']
    distorted_df, known_anomalies = distort_time_series(df, anoms=anomaly_types, return_anomalies=True)
    print(f"   ✓ Created anomalies: {anomaly_types}")

    # 3. Prepare time series data
    print("\n3. Preparing time series...")
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    distorted_df['timestamp'] = pd.to_datetime(distorted_df['timestamp'])

    # Resample to hourly
    original_hourly = df.set_index('timestamp').resample('h').mean()
    distorted_hourly = distorted_df.set_index('timestamp').resample('h').mean()

    # Convert to TimeSeries
    original_ts = TimeSeries.from_dataframe(original_hourly.reset_index(), time_col='timestamp')
    distorted_ts = TimeSeries.from_dataframe(distorted_hourly.reset_index(), time_col='timestamp')

    # Calculate percentage difference
    difference_ts = ((distorted_ts - original_ts) / original_ts) * 100
    difference_df = difference_ts.to_dataframe().set_axis(['value'], axis=1)

    # Split data
    train, test = TimeSeries.split_after(difference_ts, 0.8)
    print(f"   ✓ Train size: {len(train)}, Test size: {len(test)}")

    # 4. Initialize components
    print("\n4. Initializing anomaly detection components...")
    forecaster = LinearRegression(timesteps=24)
    scorer = EuclideanScorer()
    threshold = StaticThreshold(threshold=20, target_series=difference_df)
    detector = NABAnomalyDetector(forecaster, scorer, threshold)

    # Store model info
    model_info = {
        'Forecaster': forecaster.__class__.__name__,
        'Scorer': scorer.__class__.__name__,
        'Threshold': threshold.__class__.__name__,
        'Forecast Model': type(forecaster.model).__name__ if hasattr(forecaster, 'model') else 'N/A',
        'Threshold Value': threshold.threshold
    }

    print("   ✓ Components initialized")
    print("\n   Model Configuration:")
    for key, value in model_info.items():
        print(f"   - {key}: {value}")

    # 5. Train forecaster
    print("\n5. Training forecaster...")
    forecaster.fit(train)
    print("   ✓ Forecaster trained")

    # 6. Detect anomalies
    print("\n6. Detecting anomalies...")
    try:
        results = detector.detect(n=len(test), test=test)
        print(f"   ✓ Anomaly detection complete")
        print(f"   - Forecast shape: {results['forecast'].shape}")
        print(f"   - Anomaly scores shape: {results['anomaly_scores'].shape}")
        print(f"   - Number of anomalies detected: {len(results['anomalies'])}")
    except Exception as e:
        print(f"   ✗ Error during detection: {e}")
        import traceback
        traceback.print_exc()
        return

    # 7. Enhanced visualization with model info
    print("\n7. Creating enhanced visualization...")
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

    # Create title with model info
    fig.suptitle(f'Anomaly Detection Pipeline\n' +
                 f'Forecaster: {model_info["Forecaster"]} ({model_info["Forecast Model"]}) | ' +
                 f'Scorer: {model_info["Scorer"]} | ' +
                 f'Threshold: {model_info["Threshold"]} (value={model_info["Threshold Value"]})',
                 fontsize=14, fontweight='bold')

    # Plot 1: Test data vs Forecast
    test.plot(ax=axes[0], label='Test Data (% Difference)')
    forecast_ts = TimeSeries.from_dataframe(results['forecast'].reset_index(), time_col='timestamp')
    forecast_ts.plot(ax=axes[0], label=f'Forecast ({model_info["Forecast Model"]})', linestyle='--', linewidth=2)
    axes[0].set_title('Test Data vs Forecast', fontsize=12)
    axes[0].legend(loc='upper right')
    axes[0].set_ylabel('% Difference')
    axes[0].grid(True, alpha=0.3)

    # Add text box with model details
    textstr = f'Forecast Model: {model_info["Forecast Model"]}\nLags: 24 hours'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    axes[0].text(0.02, 0.95, textstr, transform=axes[0].transAxes, fontsize=10,
                 verticalalignment='top', bbox=props)

    # Plot 2: Anomaly scores
    scores_ts = TimeSeries.from_dataframe(results['anomaly_scores'].reset_index(), time_col='timestamp')
    scores_ts.plot(ax=axes[1], label=f'Anomaly Scores ({model_info["Scorer"]})', color='orange', linewidth=2)
    axes[1].axhline(y=20, color='r', linestyle=':', label=f'Threshold={model_info["Threshold Value"]}', linewidth=2)
    axes[1].set_title('Anomaly Scores', fontsize=12)
    axes[1].legend(loc='upper right')
    axes[1].set_ylabel('Score')
    axes[1].grid(True, alpha=0.3)

    # Add scorer info
    scorer_info = f'Scorer: {model_info["Scorer"]}\nNorm: L2 (Euclidean distance)'
    axes[1].text(0.02, 0.95, scorer_info, transform=axes[1].transAxes, fontsize=10,
                 verticalalignment='top', bbox=props)

    # Plot 3: Detected anomalies
    difference_ts.plot(ax=axes[2], label='% Difference', alpha=0.5)
    if len(results['anomalies']) > 0:
        anomaly_times = pd.to_datetime(results['anomalies'].index)
        anomaly_values = difference_df.loc[results['anomalies'].index, 'value']
        axes[2].scatter(anomaly_times, anomaly_values, color='red', s=100,
                        label=f'Detected Anomalies (n={len(results["anomalies"])})',
                        marker='o', edgecolors='darkred', linewidth=2, zorder=5)
    axes[2].set_title('Detected Anomalies', fontsize=12)
    axes[2].legend(loc='upper right')
    axes[2].set_ylabel('% Difference')
    axes[2].set_xlabel('Timestamp')
    axes[2].grid(True, alpha=0.3)

    # Add threshold info
    threshold_info = f'Threshold: {model_info["Threshold"]}\nValue: {model_info["Threshold Value"]}'
    axes[2].text(0.02, 0.95, threshold_info, transform=axes[2].transAxes, fontsize=10,
                 verticalalignment='top', bbox=props)

    plt.tight_layout()
    plt.show()

    # Print pipeline summary
    print("\n" + "=" * 60)
    print("ANOMALY DETECTION PIPELINE SUMMARY")
    print("=" * 60)
    print(f"\n1. FORECASTING:")
    print(f"   - Model: {model_info['Forecaster']} using {model_info['Forecast Model']}")
    print(f"   - Configuration: 24-hour lag window")
    print(f"   - Purpose: Predict expected % difference between time series")

    print(f"\n2. SCORING:")
    print(f"   - Model: {model_info['Scorer']}")
    print(f"   - Method: L2 norm (Euclidean distance) between forecast and actual")
    print(f"   - Purpose: Quantify deviation from expected behavior")

    print(f"\n3. THRESHOLDING:")
    print(f"   - Model: {model_info['Threshold']}")
    print(f"   - Threshold Value: {model_info['Threshold Value']}")
    print(f"   - Purpose: Identify points where anomaly score exceeds threshold")

    print(f"\n✓ Pipeline completed successfully!")
    print(f"  - Processed {len(train) + len(test)} hourly data points")
    print(f"  - Detected {len(results['anomalies'])} anomalies")
    if len(results['anomalies']) > 0:
        print(
            f"  - Anomaly dates: {results['anomalies'].index.tolist()[:5]}{'...' if len(results['anomalies']) > 5 else ''}")

    return results, model_info


if __name__ == "__main__":
    results, model_info = quick_test_with_model_info()
