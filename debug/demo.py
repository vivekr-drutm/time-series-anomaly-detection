import numpy as np
import pandas as pd
from darts import TimeSeries
from matplotlib import pyplot as plt

# Import our modular components
from forecasters import LinearRegression, NaiveSeasonal
from scorers import EuclideanScorer
from thresholds import StaticThreshold, RollingWindowThreshold, ZScoreThreshold
from nab_anomaly_detector import NABAnomalyDetector
from nab_utils import distort_time_series, load_nab

# Set random seed for reproducibility
np.random.seed(42)

# Configure plotting
plt.style.use('seaborn-v0_8-darkgrid')
plt.rcParams['figure.figsize'] = (14, 8)
plt.rcParams['font.size'] = 11

# ## 2. Data Preparation
#
# We'll use synthetic data that simulates the scenario where two advertising metrics (like Pixel and CAPI) diverge.

# Load base data
df = pd.read_csv("../data/nab_data_corpus/artificialWithAnomaly/artificialWithAnomaly/art_daily_jumpsdown.csv")

# Create synthetic anomalies to simulate metric misalignment
anomaly_types = ['zero', 'severe_drop', 'critical_drop', 'moderate_drop']
distorted_df, known_anomalies = distort_time_series(df, anoms=anomaly_types, return_anomalies=True)

print(f"Created {len(anomaly_types)} types of anomalies: {anomaly_types}")

# Convert to time series format
df['timestamp'] = pd.to_datetime(df['timestamp'])
distorted_df['timestamp'] = pd.to_datetime(distorted_df['timestamp'])

# Resample to hourly (simulating hourly metric checks)
original_hourly = df.set_index('timestamp').resample('h').mean()
distorted_hourly = distorted_df.set_index('timestamp').resample('h').mean()

# Plot the two "metrics" we're comparing
fig, ax = plt.subplots(1, 1, figsize=(14, 6))
original_hourly.plot(ax=ax, label='Metric 1 (e.g., Pixel Events)', alpha=0.8)
distorted_hourly.plot(ax=ax, label='Metric 2 (e.g., CAPI Events)', alpha=0.8)
ax.set_title('Comparison of Two Advertising Metrics', fontsize=14, fontweight='bold')
ax.set_ylabel('Event Count')
ax.legend()
plt.tight_layout()
plt.show()
# plt.savefig('demo_images/Comparison of Two Advertising Metrics', format='png')

# ## 3. Calculate Percentage Difference
#
# In real-world scenarios, we monitor the percentage difference between expected and actual values.

# Convert to TimeSeries objects
original_ts = TimeSeries.from_dataframe(original_hourly.reset_index(), time_col='timestamp')
distorted_ts = TimeSeries.from_dataframe(distorted_hourly.reset_index(), time_col='timestamp')

# Calculate percentage difference
difference_ts = ((distorted_ts - original_ts) / original_ts) * 100
difference_df = difference_ts.to_dataframe().set_axis(['value'], axis=1)

# Plot the percentage difference
fig, ax = plt.subplots(1, 1, figsize=(14, 6))
difference_ts.plot(ax=ax, label='% Difference between Metrics')
ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
ax.axhline(y=-20, color='red', linestyle='--', alpha=0.5, label='20% Threshold')
ax.axhline(y=20, color='red', linestyle='--', alpha=0.5)
ax.set_title('Percentage Difference Between Metrics', fontsize=14, fontweight='bold')
ax.set_ylabel('% Difference')
ax.legend()
plt.tight_layout()
plt.show()
# plt.savefig('demo_images/Percentage Difference Between Metrics', format='png')

# Split data for training and testing
train, test = TimeSeries.split_after(difference_ts, 0.8)
print(f"\nData split: Train size = {len(train)}, Test size = {len(test)}")

# ## 4. Demonstration 1: Linear Regression with Static Threshold
#
# This approach uses linear regression to forecast expected values and a static threshold to detect anomalies.

print("\n" + "="*60)
print("DEMO 1: Linear Regression + Static Threshold")
print("="*60)

# Initialize components
forecaster_lr = LinearRegression(timesteps=24)  # 24-hour lag window
scorer = EuclideanScorer()  # L2 norm for measuring deviation
threshold_static = StaticThreshold(threshold=20, target_series=difference_df)

# Create detector
detector_lr_static = NABAnomalyDetector(forecaster_lr, scorer, threshold_static)

# Train and detect
forecaster_lr.fit(train)
results_lr_static = detector_lr_static.detect(n=len(test), test=test)

print(f"✓ Detected {len(results_lr_static['anomalies'])} anomalies using Linear Regression + Static Threshold")

# Visualize results
def plot_detection_results(test, results, difference_df, title_suffix=""):
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

    # Plot 1: Test vs Forecast
    test.plot(ax=axes[0], label='Actual % Difference', alpha=0.8)
    forecast_ts = TimeSeries.from_dataframe(results['forecast'].reset_index(), time_col='timestamp')
    forecast_ts.plot(ax=axes[0], label='Forecast', linestyle='--', linewidth=2)
    axes[0].set_title(f'Forecast vs Actual {title_suffix}', fontsize=12)
    axes[0].set_ylabel('% Difference')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Plot 2: Anomaly Scores
    scores_ts = TimeSeries.from_dataframe(results['anomaly_scores'].reset_index(), time_col='timestamp')
    scores_ts.plot(ax=axes[1], label='Anomaly Scores', color='orange', linewidth=2)
    axes[1].set_title('Anomaly Scores (Higher = More Anomalous)', fontsize=12)
    axes[1].set_ylabel('Score')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # Plot 3: Detected Anomalies
    difference_ts.plot(ax=axes[2], label='% Difference', alpha=0.5)
    if len(results['anomalies']) > 0:
        anomaly_times = pd.to_datetime(results['anomalies'].index)
        anomaly_values = difference_df.loc[results['anomalies'].index, 'value']
        axes[2].scatter(anomaly_times, anomaly_values, color='red', s=100,
                       label=f'Anomalies (n={len(results["anomalies"])})',
                       marker='o', edgecolors='darkred', linewidth=2, zorder=5)
    axes[2].set_title('Detected Anomalies', fontsize=12)
    axes[2].set_ylabel('% Difference')
    axes[2].set_xlabel('Timestamp')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()
    # plt.savefig('demo_images/Detected Anomalies', format='png')

plot_detection_results(test, results_lr_static, difference_df,
                      "(Linear Regression + Static Threshold)")

# ## 5. Demonstration 2: Naive Seasonal with Rolling Window Threshold
#
# This approach uses seasonal patterns and a dynamic threshold based on recent history.

print("\n" + "="*60)
print("DEMO 2: Naive Seasonal + Rolling Window Threshold")
print("="*60)

# Initialize components
forecaster_ns = NaiveSeasonal(timesteps=24)  # Daily seasonality
threshold_rolling = RollingWindowThreshold(threshold=0.95, target_series=difference_df)

# Create detector
detector_ns_rolling = NABAnomalyDetector(forecaster_ns, scorer, threshold_rolling)

# Train and detect
forecaster_ns.fit(train)
results_ns_rolling = detector_ns_rolling.detect(n=24, test=test)  # window=24 for rolling

print(f"✓ Detected {len(results_ns_rolling['anomalies'])} anomalies using Naive Seasonal + Rolling Window")

# Compare the two approaches
fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

# Plot anomalies from both methods
difference_ts.plot(ax=axes[0], label='% Difference', alpha=0.5)
if len(results_lr_static['anomalies']) > 0:
    times = pd.to_datetime(results_lr_static['anomalies'].index)
    values = difference_df.loc[results_lr_static['anomalies'].index, 'value']
    axes[0].scatter(times, values, color='red', s=80, label='Linear Regression + Static',
                   marker='o', alpha=0.7)
axes[0].set_title('Comparison of Detection Methods', fontsize=14, fontweight='bold')
axes[0].set_ylabel('% Difference')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

difference_ts.plot(ax=axes[1], label='% Difference', alpha=0.5)
if len(results_ns_rolling['anomalies']) > 0:
    times = pd.to_datetime(results_ns_rolling['anomalies'].index)
    values = difference_df.loc[results_ns_rolling['anomalies'].index, 'value']
    axes[1].scatter(times, values, color='blue', s=80, label='Naive Seasonal + Rolling',
                   marker='s', alpha=0.7)
axes[1].set_ylabel('% Difference')
axes[1].set_xlabel('Timestamp')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
# plt.savefig('demo_images/Comparison of Detection Methods', format='png')

# ## 6. Demonstration 3: Z-Score Threshold
#
# This statistical approach identifies outliers based on standard deviations from the mean.

print("\n" + "="*60)
print("DEMO 3: Linear Regression + Z-Score Threshold")
print("="*60)

# Initialize Z-score threshold
threshold_zscore = ZScoreThreshold(threshold=2, target_series=difference_df)  # 2 standard deviations

# Create detector
detector_lr_zscore = NABAnomalyDetector(forecaster_lr, scorer, threshold_zscore)

# Detect anomalies (forecaster already trained)
results_lr_zscore = detector_lr_zscore.detect(n=24, test=test)  # window=24

print(f"✓ Detected {len(results_lr_zscore['anomalies'])} anomalies using Z-Score Threshold")

# ## 7. Performance Comparison
#
# Let's compare all three threshold methods side by side.

# Collect all results
methods = {
    'Static Threshold': results_lr_static['anomalies'],
    'Rolling Window': results_ns_rolling['anomalies'],
    'Z-Score': results_lr_zscore['anomalies']
}

# Create comparison visualization
fig, ax = plt.subplots(1, 1, figsize=(14, 8))

# Plot base difference
difference_ts.plot(ax=ax, label='% Difference', alpha=0.3, color='gray')

# Plot anomalies from each method
colors = ['red', 'blue', 'green']
markers = ['o', 's', '^']
for i, (method_name, anomalies) in enumerate(methods.items()):
    if len(anomalies) > 0:
        times = pd.to_datetime(anomalies.index)
        values = difference_df.loc[anomalies.index, 'value']
        ax.scatter(times, values, color=colors[i], s=100,
                  label=f'{method_name} (n={len(anomalies)})',
                  marker=markers[i], alpha=0.7, edgecolors='black', linewidth=1)

ax.set_title('Comparison of All Threshold Methods', fontsize=16, fontweight='bold')
ax.set_ylabel('% Difference')
ax.set_xlabel('Timestamp')
ax.legend(loc='upper right')
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
# plt.savefig('demo_images/Comparison of All Threshold Methods', format='png')

# Summary statistics
print("\n" + "="*60)
print("SUMMARY STATISTICS")
print("="*60)
for method_name, anomalies in methods.items():
    print(f"\n{method_name}:")
    print(f"  - Total anomalies detected: {len(anomalies)}")
    if len(anomalies) > 0:
        anomaly_values = difference_df.loc[anomalies.index, 'value']
        print(f"  - Average anomaly magnitude: {abs(anomaly_values).mean():.2f}%")
        print(f"  - Max anomaly magnitude: {abs(anomaly_values).max():.2f}%")

# ## 8. Real-World Application: Meta Advertising Alignment
#
# This system can be used to monitor alignment between different advertising metrics:

print("\n" + "="*60)
print("REAL-WORLD APPLICATION: ADVERTISING METRIC ALIGNMENT")
print("="*60)

print("""
Use Cases for this Anomaly Detection System:

1. **Pixel vs CAPI Alignment**
   - Monitor discrepancies between browser (Pixel) and server (CAPI) events
   - Alert when difference exceeds 20% threshold

2. **Event Coverage Monitoring**
   - Ensure event coverage stays above 75% (Meta best practice)
   - Detect sudden drops in event reporting

3. **Event Deduplication**
   - Flag when additional conversions exceed 20-30%
   - Identify potential double-counting issues

4. **Match Quality Tracking**
   - Monitor aggregated match quality scores
   - Alert on degradation in data quality

Key Benefits:
- **Modular Design**: Easy to swap models for different use cases
- **Multiple Methods**: Choose appropriate threshold for your metric
- **Extensible**: Add new forecasters, scorers, or thresholds
- **Production Ready**: Clean architecture for deployment
""")

# ## 9. Customization Example
#
# Here's how easy it is to create a custom configuration:

print("\n" + "="*60)
print("CUSTOMIZATION EXAMPLE")
print("="*60)

# Example: Strict anomaly detection for critical metrics
print("Creating a strict detector for critical metrics...")

# Use tighter thresholds
strict_threshold = StaticThreshold(threshold=10, target_series=difference_df)  # Lower threshold
strict_detector = NABAnomalyDetector(forecaster_lr, scorer, strict_threshold)

# Detect with strict criteria
results_strict = strict_detector.detect(n=len(test), test=test)
print(f"✓ Strict detector found {len(results_strict['anomalies'])} anomalies (vs {len(results_lr_static['anomalies'])} with normal threshold)")

# ## 10. Conclusion
#
# print("\n" + "="*60)
# print("CONCLUSION")
# print("="*60)
# print("""
# This modular anomaly detection system provides:
#
# 1. **Flexibility**: Choose components based on your data characteristics
# 2. **Transparency**: Clear pipeline shows how anomalies are detected
# 3. **Extensibility**: Easy to add new models or methods
# 4. **Production Ready**: Clean architecture suitable for deployment
#
# Next Steps:
# - Connect to real Meta API data
# - Implement Slack notifications
# - Add scheduling for automated monitoring
# - Create dashboard for visualization
# """)

# Save a summary plot
fig, ax = plt.subplots(1, 1, figsize=(14, 6))
difference_ts.plot(ax=ax, label='Metric Difference %', alpha=0.7)
ax.fill_between(difference_ts.time_index, -20, 20, alpha=0.2, color='green', label='Normal Range (±20%)')
ax.set_title('Anomaly Detection System for Advertising Metrics', fontsize=16, fontweight='bold')
ax.set_ylabel('% Difference')
ax.set_xlabel('Date')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
# plt.savefig('demo_images/Comparison of Two Advertising Metrics', format='png')

print("\n✅ Demo notebook completed successfully!")