"""
QUICK START SCRIPT - Predictive Maintenance System
Run this script to set up and test the system with your data.
"""

import sys
import os

def check_dependencies():
    """Verify all required libraries are installed."""
    print("🔍 Checking dependencies...")
    
    required = ['pandas', 'numpy', 'sklearn', 'scipy']
    missing = []
    
    for lib in required:
        try:
            __import__(lib)
            print(f"  ✓ {lib}")
        except ImportError:
            print(f"  ✗ {lib} - MISSING")
            missing.append(lib)
    
    if missing:
        print(f"\n⚠️  Missing libraries: {', '.join(missing)}")
        print("   Run: pip install pandas numpy scikit-learn scipy matplotlib")
        return False
    
    print("✓ All dependencies installed\n")
    return True


def check_data_structure():
    """Verify dat structure exists."""
    print("📁 Checking data structure...")
    
    paths_to_check = [
        'dados_brutos',
        'data',
        'src'
    ]
    
    all_exist = True
    for path in paths_to_check:
        if os.path.isdir(path):
            print(f"  ✓ {path}/")
        else:
            print(f"  ✗ {path}/ - MISSING")
            all_exist = False
    
    if not all_exist:
        print("\n⚠️  Missing directories. Workspace structure issue.")
        return False
    
    print("✓ Data structure OK\n")
    return True


def test_import_modules():
    """Test that our modules can be imported."""
    print("🔧 Testing module imports...")
    
    try:
        from predictive_maintenance_system import PredictiveMaintenanceSystem, VibrationThresholds
        print("  ✓ predictive_maintenance_system.py")
    except Exception as e:
        print(f"  ✗ predictive_maintenance_system.py: {e}")
        return False
    
    try:
        from raw_data_processor import RawVibrationDataProcessor, TimeSeriesDataBuilder
        print("  ✓ raw_data_processor.py")
    except Exception as e:
        print(f"  ✗ raw_data_processor.py: {e}")
        return False
    
    try:
        from complete_pipeline import CompletePredictiveMaintenancePipeline
        print("  ✓ complete_pipeline.py")
    except Exception as e:
        print(f"  ✗ complete_pipeline.py: {e}")
        return False
    
    print("✓ All modules imported successfully\n")
    return True


def run_demo():
    """Run a quick demo to verify system works."""
    print("🧪 Running demo analysis...")
    
    try:
        import pandas as pd
        from predictive_maintenance_system import PredictiveMaintenanceSystem, VibrationThresholds
        
        # Create sample data
        history = pd.DataFrame({
            'date': pd.date_range('2024-01-01', periods=8),
            'amplitude': [2.0, 2.5, 3.0, 3.8, 4.5, 5.3, 6.2, 7.1]
        })
        
        print("  Sample machine: MANCAL (Bearing)")
        print("  Historical vibrations (mm/s): 2.0 → 7.1 (trending upward)")
        
        # Analyze
        pm_system = PredictiveMaintenanceSystem()
        report = pm_system.analyze_machine(
            'TEST-BEARING',
            history,
            days_forecast=7
        )
        
        print("\n  Analysis Results:")
        print(f"    Risk Score: {report['risk_assessment']['risk_score']}/100")
        print(f"    Risk Level: {report['risk_assessment']['risk_level']}")
        print(f"    Current Vibration: {report['current_measurements']['amplitude_mm_s']} mm/s")
        print(f"    Severity Zone: {report['current_measurements']['severity_zone']}")
        
        if report['failure_prediction']['days_to_failure']:
            print(f"    Days to Failure: {report['failure_prediction']['days_to_failure']:.1f}")
        
        print("\n  Sample Recommendation:")
        for action in report['maintenance']['suggested_actions'][:2]:
            print(f"    • {action}")
        
        print("\n✓ Demo completed successfully!\n")
        return True
        
    except Exception as e:
        print(f"\n✗ Demo failed: {e}\n")
        import traceback
        traceback.print_exc()
        return False


def main_menu():
    """Display main menu and options."""
    print("\n" + "="*70)
    print("🏭 PREDICTIVE MAINTENANCE SYSTEM - QUICK START")
    print("="*70 + "\n")
    
    print("What would you like to do?\n")
    print("1. Run diagnostics & validation")
    print("2. Run demo analysis")
    print("3. Process equipment data from dados_brutos/")
    print("4. Run complete pipeline analysis")
    print("5. Show examples")
    print("0. Exit\n")
    
    choice = input("Select option (0-5): ").strip()
    
    return choice


def show_examples():
    """Display usage examples."""
    print("\n" + "="*70)
    print("USAGE EXAMPLES")
    print("="*70 + "\n")
    
    examples = """
EXAMPLE 1: Analyze Single Machine
──────────────────────────────────

import pandas as pd
from predictive_maintenance_system import PredictiveMaintenanceSystem

pm = PredictiveMaintenanceSystem()

# Load vibration history
data = pd.DataFrame({
    'date': ['01/01/2024', '08/01/2024', '15/01/2024'],
    'amplitude': [2.3, 3.5, 5.1]  # mm/s
})

# Analyze
report = pm.analyze_machine('MANCAL-01', data)
pm.print_report(report)


EXAMPLE 2: Process Raw Equipment Data
──────────────────────────────────────

from raw_data_processor import RawVibrationDataProcessor

processor = RawVibrationDataProcessor()

# Process all CSV files in equipment folder
df = processor.process_equipment_folder(
    'dados_brutos/Área 1000/06-TPA-1008',
    output_file='data/TPA_features.csv'
)

print(df.head())


EXAMPLE 3: Full Pipeline Analysis
──────────────────────────────────

from complete_pipeline import CompletePredictiveMaintenancePipeline

pipeline = CompletePredictiveMaintenancePipeline()

# Run complete analysis
results = pipeline.run_full_analysis('dados_brutos')

# Generate alerts
pipeline.generate_alert_report(results)


EXAMPLE 4: Risk Scoring
───────────────────────

from predictive_maintenance_system import RiskScorer

# Calculate risk for specific condition
score, level, actions = RiskScorer.calculate_composite_risk_score(
    current_amplitude=8.5,
    trend_slope=0.5,
    acceleration=0.1,
    volatility=1.0,
    days_to_failure=10
)

print(f"Risk: {score}/100 ({level})")
    """
    
    print(examples)


def run_full_diagnostics():
    """Run complete system validation."""
    print("\n" + "="*70)
    print("SYSTEM DIAGNOSTICS")
    print("="*70 + "\n")
    
    results = []
    
    # Check 1: Dependencies
    if not check_dependencies():
        results.append("❌ Dependencies check FAILED")
    else:
        results.append("✅ Dependencies check PASSED")
    
    # Check 2: Data structure
    if not check_data_structure():
        results.append("❌ Data structure check FAILED")
    else:
        results.append("✅ Data structure check PASSED")
    
    # Check 3: Module imports
    if not test_import_modules():
        results.append("❌ Module import check FAILED")
    else:
        results.append("✅ Module import check PASSED")
    
    # Check 4: Demo
    if not run_demo():
        results.append("⚠️  Demo execution had issues")
    else:
        results.append("✅ Demo execution PASSED")
    
    # Summary
    print("\n" + "="*70)
    print("DIAGNOSTIC SUMMARY")
    print("="*70 + "\n")
    
    for result in results:
        print(f"  {result}")
    
    if all("PASSED" in r or "ACCEPTED" in r for r in results):
        print("\n✅ System is ready to use!\n")
        print("Next steps:")
        print("  1. Place raw vibration CSV files in dados_brutos/")
        print("  2. Run: python complete_pipeline.py 2")
        print("  3. Check reports/ and data/ folders for results\n")
        return True
    else:
        print("\n❌ Some checks failed. Please fix issues above.\n")
        return False


def predict_maintenance_quick():
    """Quick prediction interface."""
    print("\n" + "="*70)
    print("QUICK PREDICTION TOOL")
    print("="*70 + "\n")
    
    try:
        import pandas as pd
        from predictive_maintenance_system import PredictiveMaintenanceSystem, VibrationThresholds
        
        pm = PredictiveMaintenanceSystem()
        
        print("Enter vibration measurements (mm/s), one per line.")
        print("When done, enter 'done':\n")
        
        measurements = []
        i = 1
        while True:
            val = input(f"Measurement {i}: ").strip()
            if val.lower() == 'done':
                break
            try:
                measurements.append(float(val.replace(',', '.')))
                i += 1
            except:
                print("  Invalid input. Please enter a number.")
        
        if len(measurements) < 2:
            print("Need at least 2 measurements for analysis.")
            return
        
        # Create history
        history = pd.DataFrame({
            'date': pd.date_range('2024-01-01', periods=len(measurements)),
            'amplitude': measurements
        })
        
        # Analyze
        report = pm.analyze_machine('USER-INPUT-MACHINE', history)
        
        print("\n" + "="*70)
        print("ANALYSIS RESULT")
        print("="*70 + "\n")
        
        print(f"Current Vibration: {report['current_measurements']['amplitude_mm_s']} mm/s")
        print(f"Severity Zone: {report['current_measurements']['severity_zone']}")
        print(f"Risk Score: {report['risk_assessment']['risk_score']}/100")
        print(f"Risk Level: {report['risk_assessment']['risk_level']}")
        
        if report['failure_prediction']['days_to_failure']:
            print(f"Days to Failure: {report['failure_prediction']['days_to_failure']:.1f}")
        
        print(f"\nTrend: {report['trends']['slope_mm_s_per_day']:+.4f} mm/s/day")
        print(f"Pattern: {report['trends']['pattern']}")
        
        print("\nRecommendations:")
        for action in report['maintenance']['suggested_actions']:
            if action:
                print(f"  • {action}")
        
        print()
        
    except Exception as e:
        print(f"Error: {e}")


def main():
    """Main program loop."""
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    
    if '--auto' in sys.argv:
        # Auto mode: run diagnostics
        if run_full_diagnostics():
            sys.exit(0)
        else:
            sys.exit(1)
    
    while True:
        choice = main_menu()
        
        if choice == '1':
            run_full_diagnostics()
        
        elif choice == '2':
            run_demo()
        
        elif choice == '3':
            try:
                from complete_pipeline import CompletePredictiveMaintenancePipeline
                print("\n📊 Processing equipment data...\n")
                pipeline = CompletePredictiveMaintenancePipeline()
                pipeline.run_full_analysis('dados_brutos')
            except Exception as e:
                print(f"\n❌ Error: {e}\n")
        
        elif choice == '4':
            try:
                from complete_pipeline import CompletePredictiveMaintenancePipeline
                print("\n🚀 Running complete pipeline...\n")
                pipeline = CompletePredictiveMaintenancePipeline()
                results = pipeline.run_full_analysis('dados_brutos')
                if results:
                    pipeline.generate_alert_report(results)
            except Exception as e:
                print(f"\n❌ Error: {e}\n")
        
        elif choice == '5':
            show_examples()
        
        elif choice == '0':
            print("\nGoodbye! 👋\n")
            break
        
        else:
            print("\n⚠️  Invalid option. Please select 0-5.\n")
        
        input("Press Enter to continue...")


if __name__ == "__main__":
    main()
