# 🏭 Predictive Maintenance Algorithm - Complete Implementation Guide

## Overview

This is a comprehensive machine learning system designed to predict when industrial equipment vibrations will reach dangerous levels. The system analyzes vibration data from multiple sensors (Axial, Horizontal, Vertical) and provides early warnings before equipment failure.

### Key Capabilities

✅ **Real-time Risk Assessment**: Classifies current equipment status (Normal → Alert → Warning → Critical)
✅ **Predictive Forecasting**: Estimates when equipment will reach critical vibration levels  
✅ **Days-to-Failure Estimation**: Calculates time remaining before failure
✅ **Equipment-Specific Advice**: Tailored maintenance recommendations
✅ **Trend Analysis**: Detects linear and exponential degradation patterns
✅ **Risk Scoring**: Comprehensive 0-100 risk score considering multiple factors

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│           RAW VIBRATION DATA (A, E, V Sensors)              │
│         Time-series CSV files from equipment                │
└──────────────────┬──────────────────────────────────────────┘
                   │
        ┌──────────▼──────────┐
        │ Raw Data Processor  │  (raw_data_processor.py)
        │ - FFT Analysis      │
        │ - Signal Features   │
        │ - Feature Extraction│
        └──────────┬──────────┘
                   │
        ┌──────────▼──────────────────────┐
        │ Feature-Enriched Dataset        │
        │ (features_*.csv)                │
        │ [RMS, Peak, Kurtosis, Freq]    │
        └──────────┬──────────────────────┘
                   │
      ┌────────────▼────────────┐
      │ Predictive Maintenance  │  (predictive_maintenance_system.py)
      │ System                  │
      │ - Trend Analysis        │
      │ - Forecasting           │
      │ - Risk Scoring          │
      │ - Maintenance Advisor   │
      └────────────┬────────────┘
                   │
        ┌──────────▼──────────────────────┐
        │ MAINTENANCE REPORTS              │
        │ - Risk rankings                 │
        │ - Failure predictions           │
        │ - Maintenance schedules         │
        │ - Equipment-specific advice     │
        └─────────────────────────────────┘
```

---

## Module Descriptions

### 1. **raw_data_processor.py** - Data Ingestion & Feature Extraction

Converts raw CSV vibration files into analytical features.

**Key Classes:**

- `RawVibrationDataProcessor`: Main processing engine
  - `read_vibration_csv()` - Read and parse vibration CSV files
  - `calculate_fft_features()` - Frequency domain analysis
  - `extract_signal_features()` - Time domain features (RMS, Peak, Crest Factor)
  - `process_equipment_folder()` - Batch process all files in a folder
  - `process_all_equipment()` - Recursive processing of all equipment

**Input Format:**
```csv
t[s];Y[mm/s];
0;-1.94;0;
4;-3.23;0;
8;-3.04;0;
```

**Output Features:**
```csv
filename,date,measurement_type,rms_mm_s,peak_mm_s,crest_factor,kurtosis,peak_frequency_hz,peak_amplitude,energy_low_band,energy_mid_band,energy_high_band
5-MANCAL-LA-VERT.csv,25/05/2022,Vertical,4.341,5.164,1.191,0.234,16.54,4.341,0.123,0.456,0.789
```

**Extracted Features:**
- **RMS** (mm/s): Overall vibration energy
- **Peak** (mm/s): Maximum amplitude
- **Crest Factor**: Peak/RMS ratio (indicates impulsive nature)
- **Kurtosis**: Peakedness of signal (high = fault conditions)
- **Peak Frequency** (Hz): Dominant frequency
- **Energy Bands**: Low/Mid/High frequency distribution

---

### 2. **predictive_maintenance_system.py** - Intelligence Engine

Core ML system for prediction and risk assessment.

**Key Components:**

#### A. VibrationThresholds
Severity zone classification following ISO 20816 standards:
- **Zone A** (0-2.3 mm/s): Normal operation ✅
- **Zone B** (2.3-7.1 mm/s): Alert - Continue monitoring ⚠️
- **Zone C** (7.1-11.2 mm/s): Warning - Plan maintenance 🟡
- **Zone D** (>11.2 mm/s): Critical - Immediate action required 🔴

#### B. FeatureEngineer
Transforms data into predictive features:

```python
extract_trend_features(history_df, window=5)
# Returns:
# - slope: mm/s per day (trend direction)
# - acceleration: rate of change
# - volatility: std dev of recent changes
# - recent_mean: average of recent measurements

calculate_days_to_failure(current_amplitude, slope, threshold)
# Estimates how many days until critical threshold
# Returns: (days_to_failure, confidence)
```

#### C. VibrationForecaster
Predicts future vibration levels using Random Forest:

```python
forecaster = VibrationForecaster()
forecaster.train(history_df, lookback=5)
forecast = forecaster.forecast_next_steps(history_df, steps=30)
# Returns 30-day vibration forecast
```

#### D. RiskScorer
Calculates composite risk (0-100 scale):

```python
score, level, recommendations = calculate_composite_risk_score(
    current_amplitude,     # Current vibration level
    trend_slope,          # Daily increase rate
    acceleration,         # Rate of trend change
    volatility,           # Instability measure
    days_to_failure,      # Projected failure time
    critical_threshold    # Danger threshold
)
# score: 0-100 risk value
# level: 'Low' | 'Medium' | 'High' | 'Critical'
# recommendations: List of action items
```

**Risk Score Composition:**
- 40% Current severity level
- 30% Trend direction and rate
- 20% Stability (volatility)
- 10% Acceleration of degradation

#### E. MaintenanceAdvisor
Equipment-specific recommendations:

```python
equipment_type, profile = identify_equipment_type(machine_name)
actions = get_maintenance_actions(equipment_type, risk_level, days_to_failure)
```

**Supported Equipment:**
- MANCAL (Bearings)
- MOTOR (Electric Motors)
- REDUTOR (Gearboxes)
- EXAUSTOR (Fans/Exhausters)

---

### 3. **complete_pipeline.py** - End-to-End Workflow

Orchestrates all components into a complete analysis pipeline.

**Workflow Steps:**

1. **Feature Extraction** - Process raw CSV files
2. **Equipment Inventory** - Identify unique machines
3. **Risk Assessment** - Analyze each machine
4. **Priority Ranking** - Sort by risk score
5. **Critical Alerts** - Highlight urgent issues
6. **Maintenance Schedule** - Group by time frame
7. **Detailed Reports** - Individual machine analysis
8. **Summary Statistics** - Fleet-wide metrics

---

## Getting Started

### Installation

1. **Ensure dependencies are installed:**
```bash
pip install pandas numpy scikit-learn scipy
```

2. **Place your raw data in the correct structure:**
```
dados_brutos/
├── Área 1000/
│   ├── 06-TPA-1008/
│   │   ├── ROLO ACIONAMENTO/
│   │   │   ├── 5-MANCAL-LA-VERT.csv
│   │   │   ├── 5-MANCAL-LA-HORIZ.csv
│   │   │   └── 5-MANCAL-LA-AXIAL.csv
│   │   └── REDUTOR/
│   │       ├── 4-REDUT-EIXO SAIDA-LOA-HOR.csv
│   │       └── ...
│   └── ...
└── ...
```

### Quick Start Examples

**Option 1: Run Complete Pipeline (Recommended)**
```bash
cd /workspaces/TEKNIKAO-I/src
python complete_pipeline.py 2
```

**Option 2: Process Specific Equipment**
```bash
python complete_pipeline.py 1
```

**Option 3: Single Machine Analysis**
```bash
python complete_pipeline.py 3
```

---

## Usage Examples

### Example 1: Load and Analyze Single Machine

```python
import pandas as pd
from predictive_maintenance_system import PredictiveMaintenanceSystem

# Create PM system
pm_system = PredictiveMaintenanceSystem()

# Load vibration history (example: bearing vibrations)
history_df = pd.DataFrame({
    'date': ['01/01/2024', '08/01/2024', '15/01/2024', '22/01/2024', '29/01/2024'],
    'amplitude': [2.5, 3.1, 4.2, 5.8, 7.5]  # mm/s
})

# Analyze
report = pm_system.analyze_machine(
    '5-MANCAL-LA-VERT',
    history_df,
    days_forecast=30
)

# Print results
pm_system.print_report(report)
```

**Output:**
```
============================================================
ANALYZING: 5-MANCAL-LA-VERT
============================================================

📊 MACHINE: 5-MANCAL-LA-VERT
   Equipment Type: Bearing

🔍 CURRENT STATUS:
   Vibration: 7.5 mm/s
   Zone: Warning

📈 TRENDS:
   Trend Direction: +1.2500 mm/s/day
   Pattern: linear

⏰ FAILURE PREDICTION:
   Days to Failure: 3
   Confidence: 95%

⚠️ RISK ASSESSMENT:
   Risk Score: 78.5/100
   Risk Level: High
   • 🔴 IMMEDIATE ACTION: Schedule maintenance in next 48 hours
   • 📈 Rapid degradation detected: Plan preventive maintenance

🔧 MAINTENANCE ACTIONS:
   • 🔴 Schedule maintenance within 1 week
   • 🔍 Perform detailed inspection
   • 📞 Contact maintenance team
```

### Example 2: Batch Process Equipment Folder

```python
from raw_data_processor import RawVibrationDataProcessor

processor = RawVibrationDataProcessor()

# Process all files in equipment folder
df = processor.process_equipment_folder(
    'dados_brutos/Área 1000/06-TPA-1008',
    output_file='data/TPA_1008_features.csv'
)

print(f"Processed {len(df)} measurements")
print(df.head())
```

### Example 3: Custom Risk Scoring

```python
from predictive_maintenance_system import RiskScorer

# Calculate risk for specific machine condition
score, level, actions = RiskScorer.calculate_composite_risk_score(
    current_amplitude=8.5,      # mm/s (Warning zone)
    trend_slope=0.3,            # mm/s per day
    acceleration=0.02,          # Increasing trend
    volatility=1.2,             # Some instability
    days_to_failure=15,
    critical_threshold=11.2
)

print(f"Risk Score: {score}/100")
print(f"Risk Level: {level}")
for action in actions:
    print(f"  • {action}")
```

---

## Output Files Generated

After running the pipeline, check these files:

```
data/
├── features_consolidated.csv          # All extracted features
├── features_Área*.csv                # Equipment-specific features
└── maintenance_priority_ranking.csv   # Risk ranking

reports/
├── 5-MANCAL-LA-VERT_analysis.json     # Detailed JSON report
├── 01-MOT-LOA-HOR_analysis.json
└── 4-REDUT-EIXO_SAIDA-LOA-HOR.json

alerts_report.txt                     # Concise alert summary
```

---

## Understanding the Output

### Risk Level Classification

| Risk Level | Score Range | Action Required | Timeline |
|-----------|------------|-----------------|----------|
| **Low** | 0-25 | Continue monitoring | Routine |
| **Medium** | 25-50 | Schedule inspection | 2 weeks |
| **High** | 50-75 | Plan maintenance | 1 week |
| **Critical** | 75-100 | Immediate action | 48 hours |

### Days-to-Failure Interpretation

- **< 7 days**: Equipment requires immediate shutdown preparation
- **7-30 days**: Schedule emergency maintenance
- **30-90 days**: Plan preventive maintenance
- **> 90 days**: Monitor in routine schedule

### Pattern Recognition

- **Linear**: Steady degradation at constant rate
- **Exponential**: Accelerating degradation (urgent)
- **Stable**: No significant change (safe to operate)

---

## ISO 20816 Standards Reference

This system uses ISO 20816 (replacing ISO 10816) severity zones:

| Zone | Velocity (mm/s RMS) | Classification | Action |
|------|------------------|-----------------|--------|
| A | 0 - 2.3 | Good | No action |
| B | 2.3 - 7.1 | Acceptable | Monitor |
| C | 7.1 - 11.2 | Just Tolerable | Inspect soon |
| D | > 11.2 | Unacceptable | Stop machine |

---

## Troubleshooting

### Problem: "Insufficient data for analysis"
**Solution**: Need at least 2-3 historical measurements. Check that history_df has multiple date entries.

### Problem: Forecaster shows low R² score
**Solution**: Insufficient training data. Collect at least 10+ historical measurements for accurate forecasting.

### Problem: CSV file not being read
**Solution**: Check delimiter (comma, semicolon, or tab) and encoding. Files should use UTF-8 encoding.

### Problem: "No CSV files found"
**Solution**: Ensure CSV files are in the equipment folder. Subfolder structure should be:
```
equipment_folder/
├── COMPONENT1/
│   └── sensor_measurements.csv
├── COMPONENT2/
│   └── sensor_measurements.csv
```

---

## Performance Optimization

### For Large Datasets (>10,000 files):
```python
# Process in batches
for batch in chunks:
    processor.process_equipment_folder(batch, output_file=...)
```

### For Real-Time Monitoring:
```python
# Update model incrementally
forecaster.train(updated_history_df)  # Retrains on new data

# Make fast predictions
latest_forecast = forecaster.forecast_next_steps(history_df, steps=7)
```

---

## Advanced Customization

### Modify Danger Threshold
```python
# Change Z-Zone boundary from 11.2 to custom value
VibrationThresholds.SEVERITY_ZONES['D']['min'] = 10.0
```

### Add Equipment Type
```python
MaintenanceAdvisor.EQUIPMENT_PROFILES['PUMP'] = {
    'type': 'Centrifugal Pump',
    'critical_frequency_band': (100, 500),
    'failure_modes': ['cavitation', 'seal wear', 'imbalance']
}
```

### Custom Risk Weights
```python
# Modify RiskScorer.calculate_composite_risk_score()
# Change weights from 40%, 30%, 20%, 10% to your preference
severity_component = severity_scores.get(zone, 0) * 0.50  # 50% weight
trend_component = min(30, max(0, trend_slope * 5)) * 0.30  # 30% weight
```

---

## Integration with Production Systems

### Export Risk Rankings (CSV)
```python
risk_df.to_csv('data/maintenance_priority_ranking.csv', index=False)
# Use this for work order systems
```

### Generate Daily Alerts (JSON)
```python
import json
for machine, report in analysis_results.items():
    with open(f'alerts/{machine}.json', 'w') as f:
        json.dump(report, f)
```

### Real-Time Dashboard Integration
```python
# Extract key metrics for database
metrics = {
    'machine': report['machine'],
    'risk_score': report['risk_assessment']['risk_score'],
    'risk_level': report['risk_assessment']['risk_level'],
    'days_to_failure': report['failure_prediction']['days_to_failure'],
    'timestamp': datetime.now()
}
# Send to database/dashboard API
```

---

## FAQ

**Q: How often should I run the pipeline?**
A: Weekly for routine monitoring, daily if critical machines are flagged.

**Q: What's the difference between A, E, and V sensors?**
A: Different accelerometer mounting styles that capture vibration in various configurations for comprehensive analysis.

**Q: Can I use this with historical data only?**
A: Yes! The system works with any collection of time-series measurements. More history = better predictions.

**Q: How accurate are the failure predictions?**
A: Confidence ranges from 0.5-1.0. Linear trends (>0.95 confidence) are most accurate. Days-to-failure is an estimate based on current trend.

**Q: What if vibrations suddenly spike?**
A: The system detects anomalies through acceleration parameter and alerts with updated DTF.

---

## Support & Debugging

For detailed troubleshooting:
```python
# Enable verbose output
import logging
logging.basicConfig(level=logging.DEBUG)

# Or check individual components
print(forecaster.model.score(X_test, y_test))  # Model R² score
print(trend_features)  # View computed trends
```

---

## Next Steps

1. ✅ Run pipeline on existing data: `python complete_pipeline.py 2`
2. ✅ Review generated reports in `reports/` directory
3. ✅ Check maintenance_priority_ranking.csv for risk status
4. ✅ Integrate risk scores into your maintenance system
5. ✅ Set up automated alerts for Critical/High equipment

---

**Version:** 1.0.0  
**Last Updated:** 2024  
**Requirements:** pandas, numpy, scikit-learn, scipy
