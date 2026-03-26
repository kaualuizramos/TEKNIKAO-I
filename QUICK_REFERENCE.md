# 📋 PREDICTIVE MAINTENANCE - QUICK REFERENCE CARD

## 🚀 START HERE

### Option A: Interactive Setup (Easiest)
```bash
cd src/
python quick_start.py
```

### Option B: Run Complete Analysis
```bash
cd src/
python complete_pipeline.py 2
```

### Option C: Check System Health
```bash
cd src/
python quick_start.py --auto
```

---

## 📊 WORKFLOW AT A GLANCE

```
Raw CSV Data (A, E, V)
        ↓
  Feature Extraction
        ↓
  Risk Analysis
        ↓
  Maintenance Schedule
        ↓
  Alerts & Reports
```

---

## 🎯 RISK LEVEL QUICK GUIDE

| Score | Level    | Zone | Status            | Action            |
|-------|----------|------|-------------------|-------------------|
| 0-25  | **Low**  | A    | ✅ Normal          | Continue monitor  |
| 25-50 | **Med**  | B    | ⚠️ Alert           | Schedule inspect  |
| 50-75 | **High** | C    | 🟡 Warning         | Plan maintenance  |
| 75+   | **CRIT** | D    | 🔴 Critical        | **IMMEDIATE**     |

---

## 📁 FILE ORGANIZATION

```
/src/
├── predictive_maintenance_system.py ← ML Engine
├── raw_data_processor.py           ← Data processor
├── complete_pipeline.py            ← Full workflow
└── quick_start.py                  ← Interactive menu

/data/
├── features_consolidated.csv       ← All features
├── maintenance_priority_ranking.csv ← Risk rankings
└── dataset_treino.csv             ← Training data

/reports/
└── machine_name_analysis.json      ← Detailed reports
```

---

## 🔥 CRITICAL INDICATORS

🚨 **IMMEDIATE ACTION NEEDED IF:**
- Risk Score > 75
- Days-to-Failure < 5
- Vibration > 11.2 mm/s (Zone D)
- Exponential degradation pattern

---

## 💡 KEY FEATURES

✅ **Automatic Detection:** A/E/V sensor data processing  
✅ **Trend Analysis:** Linear and exponential degradation  
✅ **Predictive:** Days-to-failure forecasting  
✅ **Risk Scoring:** 0-100 composite score  
✅ **Equipment-Specific:** Tailored recommendations  
✅ **Batch Processing:** Analyze entire fleet at once  

---

## 🔧 COMMON COMMANDS

### Process One Equipment
```python
from raw_data_processor import RawVibrationDataProcessor

processor = RawVibrationDataProcessor()
df = processor.process_equipment_folder('dados_brutos/Área 1000/06-TPA-1008')
```

### Analyze One Machine
```python
import pandas as pd
from predictive_maintenance_system import PredictiveMaintenanceSystem

pm = PredictiveMaintenanceSystem()
history = pd.DataFrame({
    'date': ['01/01/2024', '08/01/2024'],
    'amplitude': [2.5, 4.2]
})
report = pm.analyze_machine('MANCAL-01', history)
pm.print_report(report)
```

### Check Risk Score
```python
from predictive_maintenance_system import RiskScorer

score, level, actions = RiskScorer.calculate_composite_risk_score(
    current_amplitude=8.5,
    trend_slope=0.5,
    acceleration=0.1,
    volatility=1.0,
    days_to_failure=10
)
```

---

## ⏱️ TIMING GUIDE

| Task | Time | Frequency |
|------|------|-----------|
| Process new data | 5-10 min | Weekly |
| Full pipeline run | 5-15 min | Weekly |
| Equipment analysis | < 1 min | As needed |
| Demo/test | < 1 min | Setup only |

---

## 📞 TROUBLESHOOTING

**CSV not reading?**
→ Check delimiter (,;⟹) and encoding (UTF-8)

**Low forecasting score?**
→ Need more historical data (10+ measurements min)

**Module import error?**
→ Run: `pip install pandas numpy scikit-learn scipy`

**No data found?**
→ Verify CSV files exist in `dados_brutos/equipment_name/`

---

## 📈 SEVERITY ZONES (ISO 20816)

- **Zone A (0-2.3):** Good - No action needed ✅
- **Zone B (2.3-7.1):** Acceptable - Monitor ⚠️
- **Zone C (7.1-11.2):** Tolerable - Inspect soon 🟡
- **Zone D (>11.2):** Unacceptable - Stop machine 🔴

---

## 🎓 METRICS EXPLAINED

- **RMS (mm/s):** Overall vibration energy
- **Peak (mm/s):** Maximum amplitude spike
- **Crest Factor:** Peak/RMS ratio (high = impulsive)
- **Kurtosis:** Distribution peakedness (high = fault)
- **Slope (mm/s/day):** Trend rate of change
- **DTF (days):** Estimated days to failure

---

## ✨ PRO TIPS

1. **Run weekly** for consistent monitoring
2. **Check alerts_report.txt** first for urgent items
3. **Use maintenance_priority_ranking.csv** for work orders
4. **More data = Better predictions** (collect 3+ years for accuracy)
5. **Baseline your equipment** before starting (know normal ranges)

---

## 📊 EXPECTED OUTPUTS

✅ `features_consolidated.csv` - Raw features for all equipment  
✅ `maintenance_priority_ranking.csv` - Risk ranking table  
✅ `reports/*.json` - Detailed analysis per machine  
✅ `alerts_report.txt` - Human-readable alert summary  

---

## 🔄 WORKFLOW EXAMPLE

```
Monday:
1. New vibration data arrives → data/raw/
2. Run: python complete_pipeline.py 2
3. Check: data/maintenance_priority_ranking.csv
4. Schedule maintenance for High/Critical machines
5. Send alerts to operations team

Friday:
1. Review reports/ for progress
2. Validate actual maintenance vs predictions
3. Adjust machine hours if needed
```

---

## 📚 ALGORITHMS USED

- **Random Forest Regression:** Vibration forecasting
- **Random Forest Classification:** Severity prediction
- **Linear Regression:** Trend analysis
- **FFT (Fast Fourier Transform):** Frequency analysis
- **ISO 20816 Standards:** Severity mapping

---

## 🎯 SUCCESS METRICS

✅ **Reduced breakdown:** Predict failures before they happen  
✅ **Optimized maintenance:** Schedule before critical  
✅ **Equipment life:** Extend through preventive care  
✅ **Cost savings:** Avoid emergency repairs  
✅ **Uptime:** Minimize production stops  

---

## 🔗 QUICK LINKS

- Full Guide: [PREDICTIVE_MAINTENANCE_GUIDE.md](../PREDICTIVE_MAINTENANCE_GUIDE.md)
- System Code: [predictive_maintenance_system.py](predictive_maintenance_system.py)
- Data Processor: [raw_data_processor.py](raw_data_processor.py)
- Pipeline: [complete_pipeline.py](complete_pipeline.py)
- Interactive: [quick_start.py](quick_start.py)

---

**Version:** 1.0.0 | **Updated:** 2024 | **Status:** Production Ready ✅
