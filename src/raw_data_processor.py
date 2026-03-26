"""
RAW DATA PROCESSOR
Converts raw vibration CSV files into feature-enriched datasets for predictive maintenance.

Handles:
- Multiple sensor types (A, E, V measurements)
- Time-series signal processing
- FFT analysis for frequency domain features
- Batch processing of equipment data
"""

import pandas as pd
import numpy as np
import os
import glob
from pathlib import Path
from scipy import signal
from scipy.fft import fft, fftfreq
import warnings
warnings.filterwarnings('ignore')


class RawVibrationDataProcessor:
    """Process raw vibration CSV files into analytical datasets."""
    
    @staticmethod
    def read_vibration_csv(filepath):
        """
        Read raw vibration CSV file with auto-detection of format.
        
        Expected format:
        - CSV with time and acceleration columns
        - Headers may vary (e.g., 't[s]', 'Y[mm/s]')
        - Uses comma or semicolon as delimiter
        
        Returns:
        - time_array: Time points (seconds)
        - amplitude_array: Acceleration values
        - metadata: File metadata (machine, location, date, unit)
        """
        try:
            # Try different delimiters
            for delimiter in [';', ',', '\t']:
                try:
                    df = pd.read_csv(filepath, delimiter=delimiter, decimal=',', encoding='utf-8')
                    if len(df) > 10:
                        break
                except:
                    continue
            
            # Extract metadata from header comments
            metadata = RawVibrationDataProcessor._extract_metadata(filepath)
            
            # Find columns with time and amplitude data
            time_col = None
            amp_col = None
            
            for col in df.columns:
                col_lower = col.lower()
                if 't[s]' in col_lower or 'time' in col_lower:
                    time_col = col
                elif 'mm/s' in col_lower or 'y[' in col_lower or 'amp' in col_lower:
                    amp_col = col
            
            if time_col is None or amp_col is None:
                # Assume first two columns if headers not recognized
                time_col = df.columns[0]
                amp_col = df.columns[1]
            
            # Convert to numeric, replacing commas with dots
            time_data = pd.to_numeric(df[time_col].astype(str).str.replace(',', '.'), errors='coerce')
            amp_data = pd.to_numeric(df[amp_col].astype(str).str.replace(',', '.'), errors='coerce')
            
            # Remove NaN values
            valid_idx = ~(time_data.isna() | amp_data.isna())
            time_array = time_data[valid_idx].values
            amplitude_array = amp_data[valid_idx].values
            
            return time_array, amplitude_array, metadata
            
        except Exception as e:
            print(f"⚠️ Error reading {filepath}: {e}")
            return None, None, None
    
    @staticmethod
    def _extract_metadata(filepath):
        """Extract metadata from CSV file header."""
        metadata = {
            'filepath': filepath,
            'filename': os.path.basename(filepath),
            'date': 'Unknown',
            'measurement_type': 'Unknown',
            'location': 'Unknown',
            'unit': 'mm/s'
        }
        
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                header_lines = [f.readline() for _ in range(10)]
            
            header_text = ' '.join(header_lines)
            
            # Extract date (dd/mm/yyyy format)
            import re
            date_match = re.search(r'(\d{2}/\d{2}/\d{4})', header_text)
            if date_match:
                metadata['date'] = date_match.group(1)
            
            # Extract measurement type from filename
            filename = os.path.basename(filepath)
            if 'AXIAL' in filename.upper():
                metadata['measurement_type'] = 'Axial'
            elif 'HORIZ' in filename.upper() or 'HORIZONTAL' in filename.upper():
                metadata['measurement_type'] = 'Horizontal'
            elif 'VERT' in filename.upper():
                metadata['measurement_type'] = 'Vertical'
            
            if 'LOA' in filename.upper():
                metadata['location'] = 'LOA'
            elif 'LA' in filename.upper():
                metadata['location'] = 'LA'
            
        except:
            pass
        
        return metadata
    
    @staticmethod
    def calculate_fft_features(time_array, amplitude_array, sampling_rate=None):
        """
        Calculate FFT-based frequency domain features.
        
        Returns:
        - peak_frequency: Dominant frequency (Hz)
        - peak_amplitude: Amplitude at peak frequency
        - frequency_bands: Energy in different frequency ranges
        """
        if len(amplitude_array) < 10:
            return None
        
        # Estimate sampling rate if not provided
        if sampling_rate is None:
            if len(time_array) > 1:
                time_diff = np.diff(time_array)
                if np.all(time_diff > 0):
                    sampling_rate = 1.0 / np.mean(time_diff)
                else:
                    sampling_rate = 1000  # Default
            else:
                sampling_rate = 1000
        
        # Apply Hann window to reduce spectral leakage
        windowed_signal = amplitude_array * signal.windows.hann(len(amplitude_array))
        
        # Calculate FFT
        fft_result = fft(windowed_signal)
        frequencies = fftfreq(len(fft_result), 1/sampling_rate)
        magnitudes = np.abs(fft_result)
        
        # Only positive frequencies
        pos_mask = frequencies > 0
        frequencies_pos = frequencies[pos_mask]
        magnitudes_pos = magnitudes[pos_mask]
        
        if len(magnitudes_pos) == 0:
            return None
        
        # Find peak
        peak_idx = np.argmax(magnitudes_pos)
        peak_frequency = frequencies_pos[peak_idx]
        peak_magnitude = magnitudes_pos[peak_idx]
        
        # Frequency bands (ISO standard bands)
        bands = {
            'low': (0, 100),      # Hz
            'mid': (100, 1000),   # Hz
            'high': (1000, 5000)  # Hz
        }
        
        frequency_bands = {}
        for band_name, (f_min, f_max) in bands.items():
            mask = (frequencies_pos >= f_min) & (frequencies_pos < f_max)
            energy = np.sum(magnitudes_pos[mask]) if np.any(mask) else 0
            frequency_bands[band_name] = round(energy, 4)
        
        return {
            'peak_frequency_hz': round(peak_frequency, 2),
            'peak_amplitude': round(peak_magnitude, 4),
            'frequency_bands': frequency_bands
        }
    
    @staticmethod
    def extract_signal_features(time_array, amplitude_array):
        """
        Extract time-domain signal characteristics.
        
        Returns:
        - RMS: Root Mean Square (overall energy)
        - Peak: Maximum absolute value
        - Crest Factor: Peak / RMS
        - Kurtosis: Peakedness of distribution
        """
        if len(amplitude_array) < 2:
            return None
        
        rms = np.sqrt(np.mean(amplitude_array**2))
        peak = np.max(np.abs(amplitude_array))
        crest_factor = peak / rms if rms > 0 else 0
        
        # Remove DC component for kurtosis
        centered = amplitude_array - np.mean(amplitude_array)
        std_val = np.std(centered)
        if std_val > 0:
            kurtosis = np.mean((centered / std_val)**4) - 3
        else:
            kurtosis = 0
        
        return {
            'rms_mm_s': round(rms, 3),
            'peak_mm_s': round(peak, 3),
            'crest_factor': round(crest_factor, 3),
            'kurtosis': round(kurtosis, 3)
        }
    
    @staticmethod
    def process_single_file(filepath):
        """
        Process a single vibration CSV file into feature vector.
        
        Returns:
        - DataFrame row with all features
        """
        time_array, amplitude_array, metadata = RawVibrationDataProcessor.read_vibration_csv(filepath)
        
        if time_array is None:
            return None
        
        # Extract features
        signal_features = RawVibrationDataProcessor.extract_signal_features(time_array, amplitude_array)
        fft_features = RawVibrationDataProcessor.calculate_fft_features(time_array, amplitude_array)
        
        if signal_features is None or fft_features is None:
            return None
        
        # Create row
        row = {
            'filename': metadata['filename'],
            'filepath': filepath,
            'date': metadata['date'],
            'measurement_type': metadata['measurement_type'],
            'location': metadata['location'],
            'rms_mm_s': signal_features['rms_mm_s'],
            'peak_mm_s': signal_features['peak_mm_s'],
            'crest_factor': signal_features['crest_factor'],
            'kurtosis': signal_features['kurtosis'],
            'peak_frequency_hz': fft_features['peak_frequency_hz'],
            'peak_amplitude': fft_features['peak_amplitude'],
            'energy_low_band': fft_features['frequency_bands']['low'],
            'energy_mid_band': fft_features['frequency_bands']['mid'],
            'energy_high_band': fft_features['frequency_bands']['high'],
        }
        
        return row
    
    @staticmethod
    def process_equipment_folder(equipment_path, output_file=None):
        """
        Batch process all CSV files in an equipment folder.
        
        Parameters:
        - equipment_path: Path to equipment folder with CSV files
        - output_file: Optional CSV file to save results
        
        Returns:
        - DataFrame with features from all CSV files
        """
        if not os.path.isdir(equipment_path):
            print(f"⚠️ Path not found: {equipment_path}")
            return None
        
        csv_files = glob.glob(os.path.join(equipment_path, '**/*.csv'), recursive=True)
        
        if not csv_files:
            print(f"⚠️ No CSV files found in {equipment_path}")
            return None
        
        print(f"\n📁 Processing {equipment_path}")
        print(f"   Found {len(csv_files)} CSV files")
        
        rows = []
        for i, csv_file in enumerate(csv_files, 1):
            print(f"   [{i}/{len(csv_files)}] Processing {os.path.basename(csv_file)}...", end=' ')
            row = RawVibrationDataProcessor.process_single_file(csv_file)
            if row is not None:
                rows.append(row)
                print("✓")
            else:
                print("✗ (failed)")
        
        if not rows:
            print("⚠️ No data extracted")
            return None
        
        df = pd.DataFrame(rows)
        
        if output_file:
            df.to_csv(output_file, index=False)
            print(f"\n✓ Results saved to {output_file}")
        
        return df
    
    @staticmethod
    def process_all_equipment(dados_brutos_path, output_dir='data'):
        """
        Recursively process all equipment folders in dados_brutos.
        
        Creates separate dataset for each equipment.
        """
        os.makedirs(output_dir, exist_ok=True)
        
        all_datasets = {}
        
        # Get all first-level directories (equipment groups)
        equipment_dirs = [d for d in os.listdir(dados_brutos_path) 
                         if os.path.isdir(os.path.join(dados_brutos_path, d))
                         and not d.startswith('.')]
        
        print(f"🏭 Found {len(equipment_dirs)} equipment groups\n")
        
        for equipment_dir in equipment_dirs:
            equipment_path = os.path.join(dados_brutos_path, equipment_dir)
            safe_name = equipment_dir.replace('/', '_').replace(' ', '_')
            output_file = os.path.join(output_dir, f'features_{safe_name}.csv')
            
            df = RawVibrationDataProcessor.process_equipment_folder(equipment_path, output_file)
            if df is not None:
                all_datasets[equipment_dir] = df
        
        # Also create a consolidated dataset
        if all_datasets:
            consolidated_df = pd.concat(all_datasets.values(), ignore_index=True)
            consolidated_file = os.path.join(output_dir, 'features_consolidated.csv')
            consolidated_df.to_csv(consolidated_file, index=False)
            print(f"\n✓ Consolidated dataset saved: {consolidated_file}")
            print(f"  Total records: {len(consolidated_df)}")
            print(f"  Features per record: {len(consolidated_df.columns)}")
        
        return all_datasets


class TimeSeriesDataBuilder:
    """Build time-series history for each machine from multiple measurements."""
    
    @staticmethod
    def build_machine_history(csv_dataset, machine_pattern, date_col='date'):
        """
        Build chronological history for a specific machine.
        
        Parameters:
        - csv_dataset: DataFrame with processed features
        - machine_pattern: Pattern to match machine name (substring)
        - date_col: Column containing measurement date
        
        Returns:
        - history: DataFrame sorted by date with machine history
        """
        # Filter for matching machines
        mask = csv_dataset['filename'].str.contains(machine_pattern, case=False, na=False)
        history = csv_dataset[mask].copy()
        
        if len(history) == 0:
            return None
        
        # Sort by date
        try:
            history['date_parsed'] = pd.to_datetime(history['date'], format='%d/%m/%Y', errors='coerce')
            history = history.sort_values('date_parsed')
        except:
            pass
        
        return history[['date', 'rms_mm_s', 'peak_mm_s', 'peak_frequency_hz', 'crest_factor', 'kurtosis']]
    
    @staticmethod
    def resample_history(history_df, target_column='rms_mm_s'):
        """
        Convert history to time-indexed format for forecasting.
        
        Returns:
        - resampled_df: DataFrame with date index
        """
        history_df = history_df.copy()
        
        try:
            history_df['date'] = pd.to_datetime(history_df['date'], format='%d/%m/%Y')
            history_df.set_index('date', inplace=True)
            history_df = history_df[[target_column]]
            history_df.columns = ['amplitude']
            
            return history_df
        except:
            return None


# =============================================================================
# USAGE EXAMPLE
# =============================================================================

if __name__ == "__main__":
    print("Raw Data Processor initialized")
    print("Use RawVibrationDataProcessor to process vibration CSV files")
