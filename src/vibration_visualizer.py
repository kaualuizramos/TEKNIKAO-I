"""
VIBRATION TENDENCY VISUALIZER
Creates graphics for visualizing vibration tendency curves and alert limits.

Features:
- Time-series plots of vibration amplitudes
- Alert limit overlays (ISO 20816 zones)
- Trend lines and degradation curves
- Multiple sensor comparison
- Risk level color coding
"""

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Set style for professional plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


class VibrationTendencyVisualizer:
    """Create visualization graphics for vibration tendency analysis."""

    # ISO 20816 vibration severity zones (mm/s RMS)
    ISO_ZONES = {
        'A': {'Good': 0.71, 'Satisfactory': 1.8, 'Unsatisfactory': 4.5, 'Unacceptable': 11.2},
        'B': {'Good': 1.12, 'Satisfactory': 2.8, 'Unsatisfactory': 7.1, 'Unacceptable': 18.0},
        'C': {'Good': 1.8, 'Satisfactory': 4.5, 'Unsatisfactory': 11.2, 'Unacceptable': 28.0},
        'D': {'Good': 2.8, 'Satisfactory': 7.1, 'Unsatisfactory': 18.0, 'Unacceptable': 45.0}
    }

    ZONE_COLORS = {
        'Good': '#2E8B57',        # Green
        'Satisfactory': '#FFD700', # Yellow
        'Unsatisfactory': '#FF6347', # Orange
        'Unacceptable': '#DC143C'    # Red
    }

    def __init__(self, output_dir="plots"):
        """Initialize visualizer with output directory."""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

    def plot_tendency_curve(self, df, machine_name="Machine", sensor_type="V",
                           iso_category='B', show_zones=True, show_trend=True,
                           figsize=(12, 8)):
        """
        Create tendency curve plot with alert limits.

        Parameters:
        - df: DataFrame with columns [date, amplitude]
        - machine_name: Name of the machine
        - sensor_type: Sensor type (V, E, A)
        - iso_category: ISO 20816 category (A, B, C, D)
        - show_zones: Whether to show ISO zone boundaries
        - show_trend: Whether to show trend line
        """
        if df.empty or 'date' not in df.columns or 'amplitude' not in df.columns:
            print("❌ Invalid data for plotting")
            return None

        fig, ax = plt.subplots(figsize=figsize)

        # Plot vibration data
        ax.plot(df['date'], df['amplitude'],
               marker='o', markersize=4, linewidth=2,
               color='#1f77b4', alpha=0.8,
               label=f'{sensor_type} Sensor Amplitude')

        # Add trend line if requested
        if show_trend and len(df) > 3:
            try:
                # Calculate linear trend
                x_numeric = mdates.date2num(df['date'])
                z = np.polyfit(x_numeric, df['amplitude'], 1)
                p = np.poly1d(z)

                trend_dates = pd.date_range(df['date'].min(), df['date'].max(), periods=100)
                trend_x = mdates.date2num(trend_dates)
                trend_y = p(trend_x)

                ax.plot(trend_dates, trend_y, 'r--', linewidth=2,
                       label='.3f')

                # Calculate degradation rate
                days_span = (df['date'].max() - df['date'].min()).days
                if days_span > 0:
                    degradation_rate = z[0] * 365.25  # per year
                    ax.text(0.02, 0.98, f'Degradation Rate: {degradation_rate:+.4f} mm/s/year',
                           transform=ax.transAxes, fontsize=10, verticalalignment='top',
                           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            except Exception as e:
                print(f"⚠️ Could not calculate trend: {e}")

        # Add ISO zones if requested
        if show_zones and iso_category in self.ISO_ZONES:
            zones = self.ISO_ZONES[iso_category]
            date_range = [df['date'].min(), df['date'].max()]

            # Fill zones from bottom to top
            prev_limit = 0
            for zone_name, limit in zones.items():
                ax.fill_between(date_range, prev_limit, limit,
                              color=self.ZONE_COLORS[zone_name], alpha=0.2,
                              label=f'{zone_name} ({prev_limit:.2f}-{limit:.2f} mm/s)')
                ax.axhline(y=limit, color=self.ZONE_COLORS[zone_name],
                          linestyle='--', linewidth=1, alpha=0.7)
                prev_limit = limit

            # Add zone labels
            for zone_name, limit in zones.items():
                ax.text(df['date'].iloc[-1] + timedelta(days=1), limit,
                       f'{zone_name}\n{limit:.2f}',
                       fontsize=8, verticalalignment='center',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

        # Formatting
        ax.set_title(f'Vibration Tendency Curve - {machine_name}\nSensor: {sensor_type} (ISO 20816 Category {iso_category})',
                    fontsize=14, fontweight='bold', pad=20)
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('Vibration Amplitude (mm/s RMS)', fontsize=12)
        ax.grid(True, alpha=0.3)

        # Date formatting
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%d/%m/%Y'))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

        # Legend
        ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1), fontsize=10)

        plt.tight_layout()

        # Save plot
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"tendency_curve_{machine_name}_{sensor_type}_{timestamp}.png"
        filepath = self.output_dir / filename
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"💾 Saved tendency curve plot: {filepath}")

        return filepath

    def plot_multi_sensor_comparison(self, df, machine_name="Machine",
                                   figsize=(14, 10)):
        """
        Plot comparison of multiple sensors on same machine.

        Parameters:
        - df: DataFrame with columns [date, sensor_type, amplitude]
        """
        if df.empty or 'sensor_type' not in df.columns:
            print("❌ Invalid data for multi-sensor comparison")
            return None

        sensor_types = df['sensor_type'].unique()
        if len(sensor_types) < 2:
            print("⚠️ Need at least 2 sensor types for comparison")
            return None

        fig, axes = plt.subplots(len(sensor_types), 1, figsize=figsize,
                                sharex=True)

        if len(sensor_types) == 1:
            axes = [axes]

        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

        for i, sensor_type in enumerate(sensor_types):
            sensor_data = df[df['sensor_type'] == sensor_type].copy()
            
            # Use amplitude_max column (from Brazilian processor) or amplitude
            amplitude_col = 'amplitude_max' if 'amplitude_max' in sensor_data.columns else 'amplitude'

            if sensor_data.empty:
                continue

            ax = axes[i]

            # Plot data
            ax.plot(sensor_data['date'], sensor_data[amplitude_col],
                   marker='o', markersize=3, linewidth=1.5,
                   color=colors[i % len(colors)],
                   label=f'Sensor {sensor_type}')

            # Add trend line
            if len(sensor_data) > 3:
                try:
                    x_numeric = mdates.date2num(sensor_data['date'])
                    z = np.polyfit(x_numeric, sensor_data[amplitude_col], 1)
                    p = np.poly1d(z)

                    trend_dates = pd.date_range(sensor_data['date'].min(),
                                               sensor_data['date'].max(), periods=50)
                    trend_x = mdates.date2num(trend_dates)
                    trend_y = p(trend_x)

                    ax.plot(trend_dates, trend_y, '--', linewidth=2,
                           color='red', alpha=0.8,
                           label='.3f')
                except:
                    pass

            ax.set_title(f'Sensor {sensor_type}', fontsize=12, fontweight='bold')
            ax.set_ylabel('Amplitude (mm/s)', fontsize=10)
            ax.grid(True, alpha=0.3)
            ax.legend(loc='upper right')

        # Common X axis formatting
        axes[-1].xaxis.set_major_formatter(mdates.DateFormatter('%d/%m/%Y'))
        axes[-1].xaxis.set_major_locator(mdates.MonthLocator(interval=1))
        plt.setp(axes[-1].xaxis.get_majorticklabels(), rotation=45, ha='right')
        axes[-1].set_xlabel('Date', fontsize=12)

        plt.suptitle(f'Multi-Sensor Vibration Comparison - {machine_name}',
                    fontsize=14, fontweight='bold', y=0.95)

        plt.tight_layout()

        # Save plot
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"multi_sensor_{machine_name}_{timestamp}.png"
        filepath = self.output_dir / filename
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"💾 Saved multi-sensor comparison: {filepath}")

        return filepath

    def plot_risk_timeline(self, risk_scores_df, machine_name="Machine",
                          figsize=(12, 6)):
        """
        Plot risk score timeline with alert levels.

        Parameters:
        - risk_scores_df: DataFrame with columns [date, risk_score, risk_level]
        """
        if risk_scores_df.empty:
            print("❌ No risk score data for plotting")
            return None

        fig, ax = plt.subplots(figsize=figsize)

        # Color mapping for risk levels
        risk_colors = {
            'Low': '#2E8B57',
            'Medium': '#FFD700',
            'High': '#FF6347',
            'Critical': '#DC143C'
        }

        # Plot risk scores
        for risk_level in risk_scores_df['risk_level'].unique():
            level_data = risk_scores_df[risk_scores_df['risk_level'] == risk_level]
            ax.scatter(level_data['date'], level_data['risk_score'],
                      color=risk_colors.get(risk_level, '#808080'),
                      s=50, alpha=0.7, label=f'{risk_level} Risk')

        # Add risk threshold lines
        thresholds = {'Low': 0.3, 'Medium': 0.6, 'High': 0.8, 'Critical': 1.0}
        for level, threshold in thresholds.items():
            ax.axhline(y=threshold, color=risk_colors.get(level, '#808080'),
                      linestyle='--', linewidth=1, alpha=0.5)
            ax.text(risk_scores_df['date'].iloc[0], threshold + 0.02,
                   f'{level}: {threshold}', fontsize=8, verticalalignment='bottom')

        ax.set_title(f'Risk Score Timeline - {machine_name}',
                    fontsize=14, fontweight='bold')
        ax.set_xlabel('Date')
        ax.set_ylabel('Risk Score (0-1)')
        ax.set_ylim(0, 1.1)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1))

        # Date formatting
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%d/%m/%Y'))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

        plt.tight_layout()

        # Save plot
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"risk_timeline_{machine_name}_{timestamp}.png"
        filepath = self.output_dir / filename
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"💾 Saved risk timeline: {filepath}")

        return filepath

    def create_comprehensive_report(self, df, risk_scores_df=None,
                                  machine_name="Machine", iso_category='B'):
        """
        Create comprehensive visualization report with multiple plots.
        """
        print(f"📊 Creating comprehensive visualization report for {machine_name}")

        plots_created = []

        # 1. Tendency curves for each sensor type
        for sensor_type in df['sensor_type'].unique():
            sensor_data = df[df['sensor_type'] == sensor_type].copy()
            if not sensor_data.empty:
                # Create time series for plotting
                time_series = sensor_data.groupby('date')['amplitude_max'].max().reset_index()
                time_series = time_series.rename(columns={'amplitude_max': 'amplitude'})

                plot_path = self.plot_tendency_curve(
                    time_series, machine_name, sensor_type, iso_category
                )
                if plot_path:
                    plots_created.append(('tendency', sensor_type, plot_path))

        # 2. Multi-sensor comparison
        if len(df['sensor_type'].unique()) > 1:
            plot_path = self.plot_multi_sensor_comparison(df, machine_name)
            if plot_path:
                plots_created.append(('comparison', 'all', plot_path))

        # 3. Risk timeline
        if risk_scores_df is not None and not risk_scores_df.empty:
            plot_path = self.plot_risk_timeline(risk_scores_df, machine_name)
            if plot_path:
                plots_created.append(('risk', 'timeline', plot_path))

        print(f"✓ Created {len(plots_created)} visualization plots")

        return plots_created


# =============================================================================
# USAGE EXAMPLE
# =============================================================================

if __name__ == "__main__":
    print("Vibration Tendency Visualizer initialized")
    print("Use VibrationTendencyVisualizer to create tendency curve graphics")
