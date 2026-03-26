"""
PREDICTIVE MAINTENANCE SYSTEM
Predicts when machine vibrations will reach danger levels using ML and time-series forecasting.

Features:
- Analyzes vibration trends (A, E, V sensors)
- Forecasts amplitude progression  
- Calculates days-to-failure (DTF)
- Provides risk scoring and early warnings
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, classification_report
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# 1. VIBRATION SEVERITY THRESHOLDS (ISO 20816 Standards)
# =============================================================================

class VibrationThresholds:
    """Define danger thresholds for machinery vibration levels."""
    
    # ISO 20816: Velocity thresholds (mm/s RMS)
    SEVERITY_ZONES = {
        'A': {'min': 0, 'max': 2.3, 'label': 'Normal', 'rgb': (0, 255, 0)},          # Green
        'B': {'min': 2.3, 'max': 7.1, 'label': 'Alert', 'rgb': (255, 165, 0)},       # Orange  
        'C': {'min': 7.1, 'max': 11.2, 'label': 'Warning', 'rgb': (255, 69, 0)},     # Red-Orange
        'D': {'min': 11.2, 'max': float('inf'), 'label': 'Critical', 'rgb': (255, 0, 0)}  # Red
    }
    
    @staticmethod
    def classify_amplitude(amplitude):
        """Classify vibration amplitude into severity zones."""
        for zone, limits in VibrationThresholds.SEVERITY_ZONES.items():
            if limits['min'] <= amplitude < limits['max']:
                return zone, limits['label']
        return 'D', 'Critical'
    
    @staticmethod
    def get_danger_threshold():
        """Get critical danger threshold (Zone D)."""
        return VibrationThresholds.SEVERITY_ZONES['D']['min']


# =============================================================================
# 2. FEATURE ENGINEERING MODULE
# =============================================================================

class FeatureEngineer:
    """Extract advanced features from vibration data for prediction."""
    
    @staticmethod
    def extract_statistical_features(time_series_data):
        """
        Extract statistical features from raw vibration signal.
        
        Returns:
        - RMS (Root Mean Square): Overall vibration energy
        - Crest Factor: Peak / RMS ratio (impulsive nature)
        - Kurtosis: Measure of sharp peaks
        - Skewness: Asymmetry of distribution
        """
        rms = np.sqrt(np.mean(np.array(time_series_data) ** 2))
        peak = np.max(np.abs(time_series_data))
        crest_factor = peak / rms if rms > 0 else 0
        kurtosis = pd.Series(time_series_data).kurtosis()
        skewness = pd.Series(time_series_data).skew()
        
        return {
            'rms': rms,
            'peak': peak,
            'crest_factor': crest_factor,
            'kurtosis': kurtosis,
            'skewness': skewness
        }
    
    @staticmethod
    def extract_trend_features(history_df, window=5):
        """
        Extract trend features from historical amplitude data.
        
        Parameters:
        - history_df: DataFrame with dates and amplitudes
        - window: Number of recent measurements to analyze
        
        Returns:
        - Slope: Linear trend direction (positive = increasing)
        - Acceleration: Rate of change of trend
        - Volatility: Standard deviation of recent changes
        - Days to threshold: Projected days until danger level
        """
        if len(history_df) == 0:
            return None
            
        recent = history_df.tail(window)
        amplitudes = recent['amplitude'].values
        
        # Linear regression to get trend
        X = np.arange(len(amplitudes)).reshape(-1, 1)
        y = amplitudes
        
        reg = LinearRegression()
        reg.fit(X, y)
        slope = reg.coef_[0]
        
        # Acceleration (second derivative)
        if len(amplitudes) >= 2:
            changes = np.diff(amplitudes)
            acceleration = np.mean(np.diff(changes)) if len(changes) > 1 else 0
        else:
            acceleration = 0
        
        # Volatility
        volatility = np.std(amplitudes) if len(amplitudes) > 1 else 0
        
        return {
            'slope': slope,
            'acceleration': acceleration,
            'volatility': volatility,
            'recent_mean': np.mean(amplitudes),
            'recent_max': np.max(amplitudes)
        }
    
    @staticmethod
    def calculate_days_to_failure(current_amplitude, slope, critical_threshold):
        """
        Estimate days until critical threshold is reached.
        
        Parameters:
        - current_amplitude: Current vibration level (mm/s)
        - slope: Daily increase rate (mm/s per day)
        - critical_threshold: Danger level threshold
        
        Returns:
        - Days to failure (negative = already past threshold)
        - Confidence (0-1): Based on trend stability
        """
        if slope <= 0:
            return None, 0.0  # Not trending toward danger
        
        days_to_failure = (critical_threshold - current_amplitude) / slope
        
        # Confidence based on days to failure
        if days_to_failure < 0:
            confidence = 1.0  # Already critical
        elif days_to_failure < 7:
            confidence = 0.95  # Very high, imminent failure
        elif days_to_failure < 30:
            confidence = 0.85  # High confidence
        elif days_to_failure < 90:
            confidence = 0.70  # Medium confidence
        else:
            confidence = 0.50  # Low confidence for distant failures
        
        return days_to_failure, confidence


# =============================================================================
# 3. FORECASTING MODULE
# =============================================================================

class VibrationForecaster:
    """Forecast future vibration levels and identify degradation patterns."""
    
    def __init__(self):
        self.model = RandomForestRegressor(n_estimators=50, random_state=42)
        self.scaler = StandardScaler()
        self.is_trained = False
    
    def prepare_training_data(self, history_df, lookback=5):
        """
        Prepare sequential data for training forecast model.
        
        Parameters:
        - history_df: DataFrame with time-series vibration data
        - lookback: Number of past measurements to use as features
        
        Returns:
        - X: Training features (past amplitudes)
        - y: Training targets (next amplitude)
        """
        X = []
        y = []
        
        amplitudes = history_df['amplitude'].values
        
        for i in range(len(amplitudes) - lookback):
            X.append(amplitudes[i:i+lookback])
            y.append(amplitudes[i+lookback])
        
        if len(X) == 0:
            return None, None
        
        return np.array(X), np.array(y)
    
    def train(self, history_df, lookback=5):
        """Train the forecasting model."""
        X, y = self.prepare_training_data(history_df, lookback)
        
        if X is None or len(X) < 5:
            print("⚠️ Insufficient training data (need at least 5 sequences)")
            return False
        
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)
        self.is_trained = True
        
        # Calculate R² score
        y_pred = self.model.predict(X_scaled)
        r2 = r2_score(y, y_pred)
        print(f"✓ Forecaster trained. R² Score: {r2:.4f}")
        
        return True
    
    def forecast_next_steps(self, history_df, steps=30, lookback=5):
        """
        Forecast vibration levels for next N steps (days).
        
        Parameters:
        - history_df: Historical vibration data
        - steps: Number of days to forecast
        - lookback: Number of past measurements used for each prediction
        
        Returns:
        - forecast: Array of predicted amplitudes
        - confidence_intervals: Upper and lower uncertainty bounds
        """
        if not self.is_trained:
            print("⚠️ Model not trained yet")
            return None
        
        amplitudes = history_df['amplitude'].values
        forecast = list(amplitudes[-lookback:])
        
        # Simulate the forecast by repeatedly using the model
        for _ in range(steps):
            last_values = np.array(forecast[-lookback:]).reshape(1, -1)
            last_values_scaled = self.scaler.transform(last_values)
            next_pred = self.model.predict(last_values_scaled)[0]
            forecast.append(max(next_pred, 0))  # Amplitude can't be negative
        
        return forecast[-steps:]
    
    def identify_degradation_pattern(self, history_df):
        """
        Identify if vibration follows an exponential or linear degradation pattern.
        
        Returns:
        - pattern_type: 'linear', 'exponential', or 'stable'
        - degradation_rate: Rate of increase
        """
        if len(history_df) < 3:
            return 'unknown', 0
        
        amplitudes = history_df['amplitude'].values
        time_points = np.arange(len(amplitudes)).reshape(-1, 1)
        
        # Try linear fit
        linear_reg = LinearRegression()
        linear_reg.fit(time_points, amplitudes)
        linear_predictions = linear_reg.predict(time_points)
        linear_residuals = np.sum((amplitudes - linear_predictions) ** 2)
        
        # Try exponential fit (fit log of values)
        try:
            log_amplitudes = np.log(np.maximum(amplitudes, 0.001))
            exp_reg = LinearRegression()
            exp_reg.fit(time_points, log_amplitudes)
            exp_predictions = np.exp(exp_reg.predict(time_points))
            exp_residuals = np.sum((amplitudes - exp_predictions) ** 2)
            
            if exp_residuals < linear_residuals and linear_reg.coef_[0] > 0:
                return 'exponential', exp_reg.coef_[0]
        except:
            pass
        
        # Determine if stable or degrading
        if linear_reg.coef_[0] > 0.1:
            return 'linear', linear_reg.coef_[0]
        else:
            return 'stable', linear_reg.coef_[0]


# =============================================================================
# 4. RISK SCORING ENGINE
# =============================================================================

class RiskScorer:
    """Calculate comprehensive risk scores for machinery."""
    
    @staticmethod
    def calculate_composite_risk_score(
        current_amplitude, 
        trend_slope, 
        acceleration,
        volatility,
        days_to_failure=None,
        critical_threshold=11.2
    ):
        """
        Calculate a composite risk score (0-100).
        
        Score Breakdown:
        - 40%: Current severity level
        - 30%: Trend direction and rate
        - 20%: Stability (volatility)
        - 10%: Acceleration of degradation
        
        Returns:
        - score: 0-100 risk score
        - risk_level: 'Low', 'Medium', 'High', 'Critical'
        - recommendations: List of action items
        """
        
        # 1. Severity component (0-40 points)
        zone, label = VibrationThresholds.classify_amplitude(current_amplitude)
        severity_scores = {'A': 10, 'B': 20, 'C': 30, 'D': 40}
        severity_component = severity_scores.get(zone, 0)
        
        # 2. Trend component (0-30 points)
        trend_component = min(30, max(0, trend_slope * 5))  # Scale: 1 mm/s/day = 5 points
        
        # 3. Volatility component (0-20 points) - HIGH volatility = HIGH risk
        volatility_component = min(20, volatility * 2) if volatility else 0
        
        # 4. Acceleration component (0-10 points)
        acceleration_component = min(10, max(0, acceleration * 2))
        
        # Total risk score
        total_score = severity_component + trend_component + volatility_component + acceleration_component
        
        # Classify risk level
        if total_score < 25:
            risk_level = 'Low'
        elif total_score < 50:
            risk_level = 'Medium'
        elif total_score < 75:
            risk_level = 'High'
        else:
            risk_level = 'Critical'
        
        # Generate recommendations
        recommendations = []
        if current_amplitude > critical_threshold * 0.8:
            recommendations.append("🔴 IMMEDIATE ACTION: Schedule maintenance in next 48 hours")
        if trend_slope > 0.5:
            recommendations.append("📈 Rapid degradation detected: Plan preventive maintenance")
        if volatility > 2.0:
            recommendations.append("⚡ High instability: Check mounting and alignment")
        if days_to_failure and days_to_failure < 14:
            recommendations.append(f"⏰ Estimated {days_to_failure:.0f} days to critical state")
        if len(recommendations) == 0:
            recommendations.append("✓ Continue monitoring routine schedule")
        
        return total_score, risk_level, recommendations


# =============================================================================
# 5. MACHINE-SPECIFIC MAINTENANCE ADVISOR
# =============================================================================

class MaintenanceAdvisor:
    """Provide machine-specific maintenance recommendations."""
    
    # Equipment-specific failure modes
    EQUIPMENT_PROFILES = {
        'MANCAL': {
            'type': 'Bearing',
            'critical_frequency_band': (2000, 5000),  # Hz
            'failure_modes': ['bearing race damage', 'lubricant degradation', 'misalignment']
        },
        'MOTOR': {
            'type': 'Electric Motor',
            'critical_frequency_band': (10, 100),
            'failure_modes': ['rotor imbalance', 'stator issues', 'bearing wear']
        },
        'REDUTOR': {
            'type': 'Gearbox',
            'critical_frequency_band': (500, 3000),
            'failure_modes': ['gear tooth damage', 'bearing wear', 'lubrication failure']
        },
        'EXAUSTOR': {
            'type': 'Fan/Exhauster',
            'critical_frequency_band': (50, 500),
            'failure_modes': ['blade damage', 'rotor imbalance', 'bearing failure']
        }
    }
    
    @staticmethod
    def identify_equipment_type(machine_name):
        """Identify equipment type from name."""
        machine_upper = machine_name.upper()
        for equipment_key, profile in MaintenanceAdvisor.EQUIPMENT_PROFILES.items():
            if equipment_key in machine_upper:
                return equipment_key, profile
        return 'UNKNOWN', {'type': 'Unknown Equipment', 'failure_modes': []}
    
    @staticmethod
    def get_maintenance_actions(equipment_type, risk_level, days_to_failure=None):
        """Get maintenance actions based on equipment and risk."""
        actions = {
            'Low': ['✓ Continue standard monitoring'],
            'Medium': [
                '📋 Schedule inspection within 2 weeks',
                '🔧 Verify alignment and mounting',
                '📊 Increase monitoring frequency'
            ],
            'High': [
                '🔴 Schedule maintenance within 1 week',
                '🔍 Perform detailed inspection',
                '📞 Contact maintenance team',
                '⚠️ Prepare replacement equipment'
            ],
            'Critical': [
                '🚨 IMMEDIATE maintenance required',
                '⛔ Consider taking equipment offline',
                '📞 Contact maintenance team immediately',
                '🔄 Prepare for equipment replacement',
                f'⏰ Failure expected in {days_to_failure:.1f} days' if days_to_failure else ''
            ]
        }
        
        return actions.get(risk_level, [])


# =============================================================================
# 6. MAIN PREDICTIVE MAINTENANCE SYSTEM
# =============================================================================

class PredictiveMaintenanceSystem:
    """Master system orchestrating all PM components."""
    
    def __init__(self):
        self.forecaster = VibrationForecaster()
        self.feature_engineer = FeatureEngineer()
        self.risk_scorer = RiskScorer()
        self.advisor = MaintenanceAdvisor()
    
    def analyze_machine(self, machine_name, history_df, days_forecast=30):
        """
        Complete analysis of a single machine's vibration data.
        
        Returns:
        - analysis_report: Dictionary with all findings
        """
        
        if history_df is None or len(history_df) < 2:
            return {'status': 'insufficient_data'}
        
        print(f"\n{'='*60}")
        print(f"ANALYZING: {machine_name}")
        print(f"{'='*60}")
        
        # Extract current metrics
        current_amplitude = history_df['amplitude'].iloc[-1]
        
        # Get trend features
        trend_features = self.feature_engineer.extract_trend_features(history_df, window=min(5, len(history_df)))
        
        if trend_features is None:
            return {'status': 'processing_error'}
        
        slope = trend_features['slope']
        acceleration = trend_features['acceleration']
        volatility = trend_features['volatility']
        
        # Calculate days to failure
        critical_threshold = VibrationThresholds.get_danger_threshold()
        days_to_failure, confidence = self.feature_engineer.calculate_days_to_failure(
            current_amplitude, slope, critical_threshold
        )
        
        # Calculate risk score
        risk_score, risk_level, risk_recommendations = self.risk_scorer.calculate_composite_risk_score(
            current_amplitude, slope, acceleration, volatility, days_to_failure, critical_threshold
        )
        
        # Forecast
        self.forecaster.train(history_df)
        forecast = self.forecaster.forecast_next_steps(history_df, steps=days_forecast)
        
        # Pattern detection
        pattern, pattern_rate = self.forecaster.identify_degradation_pattern(history_df)
        
        # Equipment-specific advice
        equipment_key, equipment_profile = self.advisor.identify_equipment_type(machine_name)
        maintenance_actions = self.advisor.get_maintenance_actions(equipment_key, risk_level, days_to_failure)
        
        # Build report
        report = {
            'machine': machine_name,
            'equipment_type': equipment_profile['type'],
            'timestamp': history_df['date'].iloc[-1] if 'date' in history_df.columns else 'N/A',
            'current_measurements': {
                'amplitude_mm_s': round(current_amplitude, 3),
                'severity_zone': VibrationThresholds.classify_amplitude(current_amplitude)[1],
                'within_limits': current_amplitude < critical_threshold
            },
            'trends': {
                'slope_mm_s_per_day': round(slope, 4),
                'acceleration': round(acceleration, 6),
                'volatility': round(volatility, 3),
                'pattern': pattern,
                'pattern_rate': round(pattern_rate, 4)
            },
            'forecast': {
                'days_predicted': days_forecast,
                'next_7_days_max': round(np.max(forecast[:7]), 3) if forecast else 0,
                'next_30_days_max': round(np.max(forecast), 3) if forecast else 0,
                'forecast_values': [round(v, 3) for v in forecast] if forecast else []
            },
            'failure_prediction': {
                'days_to_failure': round(days_to_failure, 1) if days_to_failure else None,
                'confidence': round(confidence, 2),
                'alert_status': 'CRITICAL' if days_to_failure and days_to_failure < 7 else 'MONITOR'
            },
            'risk_assessment': {
                'risk_score': round(risk_score, 1),
                'risk_level': risk_level,
                'recommendations': risk_recommendations
            },
            'maintenance': {
                'suggested_actions': maintenance_actions,
                'next_inspection': 'Within 48 hours' if risk_level == 'Critical' else 'Within 2 weeks'
            }
        }
        
        return report
    
    def print_report(self, report):
        """Pretty-print analysis report."""
        if report.get('status') == 'insufficient_data':
            print("⚠️ Insufficient data for analysis")
            return
        
        print(f"\n📊 MACHINE: {report['machine']}")
        print(f"   Equipment Type: {report['equipment_type']}")
        
        print(f"\n🔍 CURRENT STATUS:")
        print(f"   Vibration: {report['current_measurements']['amplitude_mm_s']} mm/s")
        print(f"   Zone: {report['current_measurements']['severity_zone']}")
        
        print(f"\n📈 TRENDS:")
        print(f"   Trend Direction: {report['trends']['slope_mm_s_per_day']:+.4f} mm/s/day")
        print(f"   Pattern: {report['trends']['pattern']}")
        
        if report['failure_prediction']['days_to_failure']:
            print(f"\n⏰ FAILURE PREDICTION:")
            print(f"   Days to Failure: {report['failure_prediction']['days_to_failure']}")
            print(f"   Confidence: {report['failure_prediction']['confidence']*100:.0f}%")
        
        print(f"\n⚠️ RISK ASSESSMENT:")
        print(f"   Risk Score: {report['risk_assessment']['risk_score']}/100")
        print(f"   Risk Level: {report['risk_assessment']['risk_level']}")
        for rec in report['risk_assessment']['recommendations']:
            print(f"   • {rec}")
        
        print(f"\n🔧 MAINTENANCE ACTIONS:")
        for action in report['maintenance']['suggested_actions']:
            if action:
                print(f"   • {action}")


# =============================================================================
# USAGE EXAMPLE
# =============================================================================

if __name__ == "__main__":
    # Example usage
    print("Predictive Maintenance System initialized")
    print("Import this module and use PredictiveMaintenanceSystem class")
