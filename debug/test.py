#!/usr/bin/env python
"""
Quick test script to verify the refactored anomaly detection system works correctly.
This is a simplified version for quick testing.
"""

import pandas as pd
import matplotlib.pyplot as plt
from darts import TimeSeries

# Import the refactored modules
from forecasters import LinearRegression
from scorers import EuclideanScorer
from thresholds import StaticThreshold
from nab_anomaly_detector import NABAnomalyDetector
from nab_utils import distort_time_series


def quick_test():
    """Quick test to verify the refactored code works"""

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
    print("   ✓ Components initialized")

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
        return

    # 7. Visualize results
    print("\n7. Creating visualization...")
    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)

    # Plot difference and forecast
    test.plot(ax=axes[0], label='Test Data (% Difference)')
    forecast_ts = TimeSeries.from_dataframe(results['forecast'].reset_index(), time_col='timestamp')
    forecast_ts.plot(ax=axes[0], label='Forecast', linestyle='--')
    axes[0].set_title('Test Data vs Forecast')
    axes[0].legend()
    axes[0].set_ylabel('% Difference')

    # Plot anomaly scores
    scores_ts = TimeSeries.from_dataframe(results['anomaly_scores'].reset_index(), time_col='timestamp')
    scores_ts.plot(ax=axes[1], label='Anomaly Scores', color='orange')
    axes[1].axhline(y=20, color='r', linestyle=':', label='Threshold=20')
    axes[1].set_title('Anomaly Scores')
    axes[1].legend()
    axes[1].set_ylabel('Score')

    # Plot detected anomalies
    difference_ts.plot(ax=axes[2], label='% Difference', alpha=0.5)
    if len(results['anomalies']) > 0:
        anomaly_times = pd.to_datetime(results['anomalies'].index)
        anomaly_values = difference_df.loc[results['anomalies'].index, 'value']
        axes[2].scatter(anomaly_times, anomaly_values, color='red', s=100,
                        label=f'Detected Anomalies (n={len(results["anomalies"])})',
                        marker='o', edgecolors='darkred', linewidth=2)
    axes[2].set_title('Detected Anomalies')
    axes[2].legend()
    axes[2].set_ylabel('% Difference')
    axes[2].set_xlabel('Timestamp')

    plt.tight_layout()
    plt.show()

    print("\n✓ Quick test completed successfully!")
    print(f"\nSummary:")
    print(f"  - Processed {len(train) + len(test)} hourly data points")
    print(f"  - Detected {len(results['anomalies'])} anomalies")
    if len(results['anomalies']) > 0:
        print(
            f"  - Anomaly dates: {results['anomalies'].index.tolist()[:5]}{'...' if len(results['anomalies']) > 5 else ''}")

    return results


if __name__ == "__main__":
    results = quick_test()