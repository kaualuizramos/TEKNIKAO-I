"""
COMPLETE PREDICTIVE MAINTENANCE PIPELINE
End-to-end example demonstrating the full workflow from raw data to maintenance predictions.
"""

import pandas as pd
import numpy as np
import sys
import os

# Import our modules
from predictive_maintenance_system import (
    PredictiveMaintenanceSystem, VibrationThresholds, RiskScorer
)
from raw_data_processor import (
    RawVibrationDataProcessor, TimeSeriesDataBuilder
)
from brazilian_data_processor import BrazilianDataProcessor
from vibration_visualizer import VibrationTendencyVisualizer


class CompletePredictiveMaintenancePipeline:
    """
    Master pipeline orchestrating the complete PM workflow:
    1. Raw data ingestion
    2. Feature extraction
    3. Historical analysis
    4. Failure prediction
    5. Maintenance scheduling
    """
    
    def __init__(self):
        self.pm_system = PredictiveMaintenanceSystem()
        self.data_processor = RawVibrationDataProcessor()
        self.ts_builder = TimeSeriesDataBuilder()
        self.brazilian_processor = BrazilianDataProcessor()
        self.visualizer = VibrationTendencyVisualizer()
    
    def run_full_analysis(self, dados_brutos_path, features_dataset_path=None):
        """
        Execute complete analysis pipeline.
        
        Parameters:
        - dados_brutos_path: Path to raw vibration data
        - features_dataset_path: Path to pre-processed features CSV (if available)
        """
        
        print("\n" + "="*70)
        print("PREDICTIVE MAINTENANCE PIPELINE - FULL ANALYSIS")
        print("="*70)
        
        # =====================================================================
        # STEP 1: EXTRACT FEATURES FROM RAW DATA
        # =====================================================================
        print("\n🔍 STEP 1: Feature Extraction from Raw Data")
        print("-" * 70)
        
        if features_dataset_path and os.path.exists(features_dataset_path):
            print(f"✓ Loading pre-processed features from: {features_dataset_path}")
            features_df = pd.read_csv(features_dataset_path)
        else:
            print(f"📊 Processing raw vibration files from: {dados_brutos_path}")
            features_dict = self.data_processor.process_all_equipment(
                dados_brutos_path,
                output_dir='data'
            )
            
            if not features_dict:
                print("❌ No data could be extracted")
                return None
            
            # Consolidate all equipment into one dataset
            features_df = pd.concat(features_dict.values(), ignore_index=True)
        
        print(f"✓ Extracted features from {len(features_df)} measurements")
        print(f"  Columns: {', '.join(features_df.columns[:8])}...")
        
        # =====================================================================
        # STEP 2: IDENTIFY UNIQUE MACHINES
        # =====================================================================
        print("\n🏭 STEP 2: Equipment Inventory")
        print("-" * 70)
        
        unique_machines = features_df['filename'].unique()
        print(f"✓ Found {len(unique_machines)} unique measurement files")
        
        # Group by machine base name (remove axis designation)
        machine_groups = {}
        for machine in unique_machines:
            # Extract base name (e.g., "5-MANCAL-LA" from "5-MANCAL-LA-VERT.csv")
            base_name = machine.rsplit('-', 1)[0] if '-' in machine else machine
            if base_name not in machine_groups:
                machine_groups[base_name] = []
            machine_groups[base_name].append(machine)
        
        print(f"✓ Grouped into {len(machine_groups)} equipment units")
        
        # =====================================================================
        # STEP 3: RISK ASSESSMENT FOR EACH MACHINE
        # =====================================================================
        print("\n⚠️  STEP 3: Risk Assessment")
        print("-" * 70)
        
        risk_summary = []
        analysis_results = {}
        
        for base_machine, file_variants in machine_groups.items():
            # Aggregate measurements for this machine
            machine_data = features_df[features_df['filename'].isin(file_variants)]
            
            if len(machine_data) == 0:
                continue
            
            # Build time series
            try:
                machine_data['date_parsed'] = pd.to_datetime(
                    machine_data['date'], format='%d/%m/%Y', errors='coerce'
                )
                machine_data = machine_data.sort_values('date_parsed')
                
                history_df = machine_data[['date_parsed', 'rms_mm_s']].copy()
                history_df.columns = ['date', 'amplitude']
                history_df = history_df.reset_index(drop=True)
            except:
                history_df = pd.DataFrame({
                    'date': machine_data['date'].values,
                    'amplitude': machine_data['rms_mm_s'].values
                })
            
            # Analyze
            report = self.pm_system.analyze_machine(
                base_machine,
                history_df,
                days_forecast=30
            )
            
            analysis_results[base_machine] = report
            
            # Collect for summary
            if 'risk_assessment' in report:
                risk_summary.append({
                    'Machine': base_machine,
                    'Risk_Score': report['risk_assessment']['risk_score'],
                    'Risk_Level': report['risk_assessment']['risk_level'],
                    'Current_Amplitude': report['current_measurements']['amplitude_mm_s'],
                    'Days_to_Failure': report['failure_prediction']['days_to_failure'],
                    'Status': 'CRITICAL' if report['failure_prediction']['days_to_failure'] 
                             and report['failure_prediction']['days_to_failure'] < 7 else 'MONITOR'
                })
        
        # =====================================================================
        # STEP 4: PRIORITY RANKING
        # =====================================================================
        print("\n🎯 STEP 4: Maintenance Priority Ranking")
        print("-" * 70)
        
        if risk_summary:
            risk_df = pd.DataFrame(risk_summary)
            risk_df = risk_df.sort_values('Risk_Score', ascending=False)
            
            print("\n📋 RISK RANKING (Top Priority First):\n")
            print(risk_df.to_string(index=False))
            
            # Save ranking
            risk_df.to_csv('data/maintenance_priority_ranking.csv', index=False)
            print("\n✓ Ranking saved to: data/maintenance_priority_ranking.csv")
        
        # =====================================================================
        # STEP 5: CRITICAL ALERTS
        # =====================================================================
        print("\n🚨 STEP 5: Critical Alerts")
        print("-" * 70)
        
        critical_machines = [
            m for m, r in analysis_results.items()
            if r.get('failure_prediction', {}).get('alert_status') == 'CRITICAL'
        ]
        
        if critical_machines:
            print(f"\n⛔ {len(critical_machines)} MACHINES REQUIRE IMMEDIATE ATTENTION:\n")
            for machine in critical_machines:
                report = analysis_results[machine]
                print(f"   🔴 {machine}")
                print(f"       Risk: {report['risk_assessment']['risk_level']}")
                print(f"       Days to Failure: {report['failure_prediction']['days_to_failure']}")
                for action in report['maintenance']['suggested_actions']:
                    if action:
                        print(f"       → {action}")
                print()
        else:
            print("✓ No critical alerts at this time")
        
        # =====================================================================
        # STEP 6: MAINTENANCE SCHEDULE
        # =====================================================================
        print("\n📅 STEP 6: Recommended Maintenance Schedule")
        print("-" * 70)
        
        schedule = {
            'Immediate (Next 48h)': [],
            'Urgent (Next Week)': [],
            'Soon (Next 30 Days)': [],
            'Routine (Within 90 Days)': []
        }
        
        for machine, report in analysis_results.items():
            risk_level = report['risk_assessment']['risk_level']
            dtf = report['failure_prediction']['days_to_failure']
            
            if risk_level == 'Critical':
                schedule['Immediate (Next 48h)'].append(machine)
            elif risk_level == 'High':
                schedule['Urgent (Next Week)'].append(machine)
            elif risk_level == 'Medium':
                schedule['Soon (Next 30 Days)'].append(machine)
            else:
                schedule['Routine (Within 90 Days)'].append(machine)
        
        for time_frame, machines in schedule.items():
            if machines:
                print(f"\n{time_frame}: ({len(machines)} machines)")
                for m in machines:
                    print(f"  • {m}")
        
        # =====================================================================
        # STEP 7: GENERATE DETAILED REPORTS
        # =====================================================================
        print("\n📄 STEP 7: Generating Detailed Reports")
        print("-" * 70)
        
        os.makedirs('reports', exist_ok=True)
        
        for machine, report in analysis_results.items():
            self.pm_system.print_report(report)
            
            # Save report as JSON
            import json
            safe_name = machine.replace('/', '_').replace(' ', '_')
            report_file = f'reports/{safe_name}_analysis.json'
            
            # Convert non-serializable types
            report_copy = self._serialize_report(report)
            with open(report_file, 'w') as f:
                json.dump(report_copy, f, indent=2, default=str)
        
        print(f"\n✓ Detailed reports saved to: reports/")
        
        # =====================================================================
        # STEP 8: SUMMARY STATISTICS
        # =====================================================================
        print("\n📊 STEP 8: Fleet Summary Statistics")
        print("-" * 70)
        
        if risk_summary:
            risk_df = pd.DataFrame(risk_summary)
            
            print(f"\nTotal Equipment Analyzed: {len(risk_df)}")
            print(f"\nRisk Distribution:")
            for level in ['Critical', 'High', 'Medium', 'Low']:
                count = len(risk_df[risk_df['Risk_Level'] == level])
                percentage = (count / len(risk_df)) * 100
                print(f"  {level:12}: {count:3} units ({percentage:5.1f}%)")
            
            print(f"\nAverage Risk Score: {risk_df['Risk_Score'].mean():.1f}/100")
            print(f"Average Vibration Level: {risk_df['Current_Amplitude'].mean():.2f} mm/s")
            
            # Machines needing attention within 30 days
            urgent = risk_df[
                (risk_df['Days_to_Failure'].notna()) & 
                (risk_df['Days_to_Failure'] < 30)
            ]
            if len(urgent) > 0:
                print(f"\n⚠️  {len(urgent)} machines need attention within 30 days")
        
        print("\n" + "="*70)
        print("ANALYSIS COMPLETE")
        print("="*70 + "\n")
        
        return analysis_results
    
    def run_brazilian_dataset_analysis(self, dataset_path, machine_name="Dataset_Machine",
                                     iso_category='B', create_visualizations=True):
        """
        Execute complete analysis pipeline for Brazilian Dataset.txt format.
        
        Parameters:
        - dataset_path: Path to Dataset.txt file
        - machine_name: Name for the machine/equipment
        - iso_category: ISO 20816 vibration category (A, B, C, D)
        - create_visualizations: Whether to generate tendency curve plots
        """
        
        print("\n" + "="*70)
        print("BRAZILIAN DATASET ANALYSIS - PREDICTIVE MAINTENANCE")
        print("="*70)
        
        # =====================================================================
        # STEP 1: VALIDATE AND PROCESS BRAZILIAN DATASET
        # =====================================================================
        print("\n🇧🇷 STEP 1: Brazilian Data Processing")
        print("-" * 70)
        
        # Validate format
        is_valid, message = self.brazilian_processor.validate_brazilian_format(dataset_path)
        if not is_valid:
            print(f"❌ Invalid Brazilian format: {message}")
            return None
        
        print(f"✓ Valid Brazilian format: {message}")
        
        # Process the dataset
        df = self.brazilian_processor.process_dataset_file(dataset_path, machine_name)
        if df is None or df.empty:
            print("❌ Failed to process dataset")
            return None
        
        print(f"✓ Processed {len(df)} measurements")
        print(f"  Date range: {df['date'].min().date()} to {df['date'].max().date()}")
        print(f"  Sensor types: {', '.join(df['sensor_type'].unique())}")
        
        # =====================================================================
        # STEP 2: CREATE VISUALIZATIONS
        # =====================================================================
        if create_visualizations:
            print("\n📊 STEP 2: Creating Tendency Visualizations")
            print("-" * 70)
            
            # Create comprehensive visualization report
            plots_created = self.visualizer.create_comprehensive_report(
                df, machine_name=machine_name, iso_category=iso_category
            )
            
            print(f"✓ Created {len(plots_created)} visualization plots")
        
        # =====================================================================
        # STEP 3: ANALYZE EACH SENSOR TYPE
        # =====================================================================
        print("\n🔍 STEP 3: Sensor-by-Sensor Analysis")
        print("-" * 70)
        
        analysis_results = {}
        risk_summary = []
        
        for sensor_type in df['sensor_type'].unique():
            sensor_data = df[df['sensor_type'] == sensor_type].copy()
            
            print(f"\n📡 Analyzing Sensor {sensor_type}:")
            
            # Create time series for this sensor
            time_series = self.brazilian_processor.create_time_series_for_prediction(
                sensor_data, target_column='amplitude_max'
            )
            
            if time_series is None or time_series.empty:
                print(f"  ⚠️  Insufficient data for sensor {sensor_type}")
                continue
            
            print(f"  ✓ {len(time_series)} measurements for analysis")
            
            # Run predictive maintenance analysis
            report = self.pm_system.analyze_machine(
                f"{machine_name}_{sensor_type}",
                time_series,
                days_forecast=30
            )
            
            analysis_results[sensor_type] = report
            
            # Collect risk summary
            if 'risk_assessment' in report:
                risk_summary.append({
                    'Sensor': sensor_type,
                    'Risk_Score': report['risk_assessment']['risk_score'],
                    'Risk_Level': report['risk_assessment']['risk_level'],
                    'Current_Amplitude': report['current_measurements']['amplitude_mm_s'],
                    'Days_to_Failure': report['failure_prediction']['days_to_failure'],
                    'ISO_Category': iso_category
                })
        
        # =====================================================================
        # STEP 4: CONSOLIDATED RISK ASSESSMENT
        # =====================================================================
        print("\n⚠️  STEP 4: Consolidated Risk Assessment")
        print("-" * 70)
        
        if risk_summary:
            risk_df = pd.DataFrame(risk_summary)
            risk_df = risk_df.sort_values('Risk_Score', ascending=False)
            
            print("\n📋 SENSOR RISK RANKING:\n")
            print(risk_df.to_string(index=False))
            
            # Overall machine risk (worst sensor)
            overall_risk = risk_df['Risk_Score'].max()
            overall_level = risk_df.loc[risk_df['Risk_Score'].idxmax(), 'Risk_Level']
            
            print(f"\n🏭 OVERALL MACHINE RISK: {overall_level} ({overall_risk:.1f}/100)")
        
        # =====================================================================
        # STEP 5: MAINTENANCE RECOMMENDATIONS
        # =====================================================================
        print("\n🔧 STEP 5: Maintenance Recommendations")
        print("-" * 70)
        
        critical_sensors = [
            sensor for sensor, report in analysis_results.items()
            if report.get('failure_prediction', {}).get('alert_status') == 'CRITICAL'
        ]
        
        if critical_sensors:
            print(f"\n🚨 CRITICAL SENSORS REQUIRING IMMEDIATE ATTENTION:\n")
            for sensor in critical_sensors:
                report = analysis_results[sensor]
                print(f"   🔴 Sensor {sensor}")
                print(f"       Risk: {report['risk_assessment']['risk_level']}")
                print(f"       Current: {report['current_measurements']['amplitude_mm_s']:.2f} mm/s")
                if report['failure_prediction']['days_to_failure']:
                    print(f"       Days to Failure: {report['failure_prediction']['days_to_failure']}")
                print(f"       Actions: {', '.join([a for a in report['maintenance']['suggested_actions'] if a])}")
                print()
        else:
            print("✓ No critical alerts detected")
        
        # =====================================================================
        # STEP 6: SAVE RESULTS
        # =====================================================================
        print("\n💾 STEP 6: Saving Analysis Results")
        print("-" * 70)
        
        os.makedirs('reports', exist_ok=True)
        os.makedirs('data', exist_ok=True)
        
        # Save processed data
        output_file = f"data/{machine_name.replace(' ', '_')}_processed.csv"
        df.to_csv(output_file, index=False)
        print(f"✓ Processed data saved: {output_file}")
        
        # Save risk summary
        if risk_summary:
            risk_file = f"reports/{machine_name.replace(' ', '_')}_risk_summary.csv"
            risk_df = pd.DataFrame(risk_summary)
            risk_df.to_csv(risk_file, index=False)
            print(f"✓ Risk summary saved: {risk_file}")
        
        # Save detailed reports
        for sensor_type, report in analysis_results.items():
            safe_name = f"{machine_name.replace(' ', '_')}_{sensor_type}"
            report_file = f'reports/{safe_name}_analysis.json'
            
            import json
            report_copy = self._serialize_report(report)
            with open(report_file, 'w') as f:
                json.dump(report_copy, f, indent=2, default=str)
            print(f"✓ Sensor {sensor_type} report saved: {report_file}")
        
        print("\n" + "="*70)
        print("BRAZILIAN DATASET ANALYSIS COMPLETE")
        print("="*70 + "\n")
        
        return {
            'processed_data': df,
            'analysis_results': analysis_results,
            'risk_summary': risk_summary if risk_summary else None,
            'visualizations': plots_created if create_visualizations else None
        }
    
    @staticmethod
    def _serialize_report(report):
        """Convert report to JSON-serializable format."""
        report_copy = {}
        for key, value in report.items():
            if isinstance(value, (dict, list)):
                report_copy[key] = CompletePredictiveMaintenancePipeline._serialize_report(value) \
                    if isinstance(value, dict) else [
                    CompletePredictiveMaintenancePipeline._serialize_report(v) 
                    if isinstance(v, dict) else str(v) for v in value
                ]
            elif isinstance(value, np.ndarray):
                report_copy[key] = value.tolist()
            elif isinstance(value, (np.integer, np.floating)):
                report_copy[key] = float(value)
            else:
                report_copy[key] = value
        return report_copy
    
    def generate_alert_report(self, analysis_results, output_file='alerts_report.txt'):
        """
        Generate a concise alert report for operations team.
        """
        with open(output_file, 'w') as f:
            f.write("="*70 + "\n")
            f.write("PREDICTIVE MAINTENANCE ALERT REPORT\n")
            f.write("="*70 + "\n\n")
            
            critical_count = 0
            
            for machine, report in analysis_results.items():
                risk_level = report['risk_assessment']['risk_level']
                
                if risk_level in ['Critical', 'High']:
                    critical_count += 1
                    f.write(f"\n{'='*70}\n")
                    f.write(f"MACHINE: {machine}\n")
                    f.write(f"{'='*70}\n")
                    f.write(f"Risk Level: {risk_level}\n")
                    f.write(f"Current Amplitude: {report['current_measurements']['amplitude_mm_s']} mm/s\n")
                    
                    if report['failure_prediction']['days_to_failure']:
                        f.write(f"Days to Failure: {report['failure_prediction']['days_to_failure']}\n")
                    
                    f.write(f"\nRecommended Actions:\n")
                    for action in report['maintenance']['suggested_actions']:
                        if action:
                            f.write(f"  • {action}\n")
            
            f.write(f"\n\n{'='*70}\n")
            f.write(f"Summary: {critical_count} machines require attention\n")
            f.write(f"{'='*70}\n")
        
        print(f"✓ Alert report saved: {output_file}")


# =============================================================================
# QUICK START EXAMPLES
# =============================================================================

def example_1_process_new_equipment():
    """
    Example 1: Process raw vibration data from a new equipment folder.
    """
    print("\n" + "="*70)
    print("EXAMPLE 1: Process New Equipment Data")
    print("="*70 + "\n")
    
    pipeline = CompletePredictiveMaintenancePipeline()
    
    # Process a specific equipment folder
    equipment_path = 'dados_brutos/Área 1000/06-TPA-1008'
    
    if os.path.exists(equipment_path):
        df = RawVibrationDataProcessor.process_equipment_folder(
            equipment_path,
            output_file='data/TPA_1008_features.csv'
        )
        
        if df is not None:
            print(f"\n✓ Successfully processed {len(df)} measurements")
            print(f"  Sample data:\n{df.head()}")
    else:
        print(f"Equipment path not found: {equipment_path}")


def example_2_full_pipeline():
    """
    Example 2: Run complete pipeline on existing dataset.
    """
    print("\n" + "="*70)
    print("EXAMPLE 2: Complete Predictive Maintenance Pipeline")
    print("="*70 + "\n")
    
    pipeline = CompletePredictiveMaintenancePipeline()
    
    # Check if we have existing features
    features_file = 'data/dataset_treino.csv'
    
    if os.path.exists(features_file):
        print(f"✓ Using existing features dataset: {features_file}\n")
        analysis_results = pipeline.run_full_analysis(
            'dados_brutos',
            features_dataset_path=features_file
        )
    else:
        print(f"Extracting features from raw data...\n")
        analysis_results = pipeline.run_full_analysis('dados_brutos')
    
    # Generate alert report
    if analysis_results:
        pipeline.generate_alert_report(analysis_results)


def example_3_single_machine_analysis():
    """
    Example 3: Detailed analysis of a single machine.
    """
    print("\n" + "="*70)
    print("EXAMPLE 3: Single Machine Deep Analysis")
    print("="*70 + "\n")
    
    from PredictiveMaintenanceSystem.predictive_maintenance_system import PredictiveMaintenanceSystem
    
    pm_system = PredictiveMaintenanceSystem()
    
    # Load or create sample data
    history_data = pd.DataFrame({
        'date': pd.date_range('2024-01-01', periods=10),
        'amplitude': [2.1, 2.3, 2.5, 3.0, 3.2, 3.8, 4.5, 5.2, 6.1, 7.0]
    })
    
    print("Analyzing bearing vibrations with acceleration trend...\n")
    
    report = pm_system.analyze_machine(
        '5-MANCAL-LA-VERT',
        history_data,
        days_forecast=30
    )
    
    pm_system.print_report(report)


def example_4_brazilian_dataset_analysis():
    """
    Example 4: Analyze Brazilian Dataset.txt with visualizations.
    """
    print("\n" + "="*70)
    print("EXAMPLE 4: Brazilian Dataset Analysis with Graphics")
    print("="*70 + "\n")
    
    pipeline = CompletePredictiveMaintenancePipeline()
    
    # Path to Dataset.txt
    dataset_path = 'data/Dataset.txt'
    
    if os.path.exists(dataset_path):
        print(f"📄 Analyzing Brazilian dataset: {dataset_path}\n")
        
        results = pipeline.run_brazilian_dataset_analysis(
            dataset_path,
            machine_name="Brazilian_Equipment",
            iso_category='B',  # ISO 20816 Category B (medium machinery)
            create_visualizations=True
        )
        
        if results:
            print("\n✅ Analysis completed successfully!")
            print(f"   📊 Visualizations saved to: plots/")
            print(f"   📋 Reports saved to: reports/")
            print(f"   💾 Processed data saved to: data/")
        else:
            print("\n❌ Analysis failed")
    else:
        print(f"❌ Dataset.txt not found at: {dataset_path}")
        print("   Please ensure Dataset.txt is in the data/ folder")


# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    import sys
    
    print("\n🔧 PREDICTIVE MAINTENANCE - COMPLETE PIPELINE")
    print("="*70)
    
    if len(sys.argv) > 1:
        example = sys.argv[1]
        
        if example == '1':
            example_1_process_new_equipment()
        elif example == '2':
            example_2_full_pipeline()
        elif example == '3':
            example_3_single_machine_analysis()
        elif example == '4':
            example_4_brazilian_dataset_analysis()
        else:
            print(f"Unknown example: {example}")
            print("Usage: python complete_pipeline.py [1|2|3|4]")
    else:
        print("\nUsage Examples:")
        print("  python complete_pipeline.py 1  - Process new equipment data")
        print("  python complete_pipeline.py 2  - Run complete pipeline")
        print("  python complete_pipeline.py 3  - Single machine analysis")
        print("  python complete_pipeline.py 4  - Brazilian dataset analysis")
        print("\nOr run: python complete_pipeline.py 4  (Brazilian dataset analysis)\n")
        
        # Run default (Brazilian dataset analysis)
        example_4_brazilian_dataset_analysis()
