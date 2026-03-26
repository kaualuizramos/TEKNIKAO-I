"""
BRAZILIAN DATA PROCESSOR
Specialized processor for Brazilian vibration data format (Dataset.txt).

Handles:
- Brazilian decimal format (comma separator)
- DD/MM/YYYY date format
- V (Vibration) and E (Envelope) sensor types
- Multiple measurements per machine
- Time-series reconstruction from discrete measurements
"""

import pandas as pd
import numpy as np
import os
import re
from datetime import datetime
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')


class BrazilianDataProcessor:
    """Process Brazilian vibration data format (Dataset.txt)."""

    @staticmethod
    def parse_dataset_txt(filepath):
        """
        Parse Dataset.txt file with Brazilian format.

        Format:
        V
        DD/MM/YYYY	0,18	0,00
        DD/MM/YYYY	3,60	0,00
        ...
        E
        DD/MM/YYYY	0,00	0,02
        ...

        Returns:
        - List of measurement blocks, each containing:
          - sensor_type: 'V' or 'E'
          - measurements: List of (date, val1, val2) tuples
        """
        blocks = []
        current_block = None

        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                lines = f.readlines()

            i = 0
            while i < len(lines):
                line = lines[i].strip()

                # Check for sensor type header (V or E)
                if line in ['V', 'E']:
                    # Save previous block if exists
                    if current_block and current_block['measurements']:
                        blocks.append(current_block)

                    # Start new block
                    current_block = {
                        'sensor_type': line,
                        'measurements': []
                    }
                    i += 1
                    continue

                # Parse measurement line: DD/MM/YYYY	val1,val2
                if current_block and line:
                    parts = line.split('\t')
                    if len(parts) >= 3:
                        date_str = parts[0].strip()
                        val1_str = parts[1].strip()
                        val2_str = parts[2].strip() if len(parts) > 2 else '0,00'

                        try:
                            # Parse Brazilian date format
                            date_obj = datetime.strptime(date_str, '%d/%m/%Y')

                            # Convert Brazilian decimal format (comma to dot)
                            val1 = float(val1_str.replace(',', '.'))
                            val2 = float(val2_str.replace(',', '.'))

                            current_block['measurements'].append({
                                'date': date_obj,
                                'val1': val1,
                                'val2': val2,
                                'date_str': date_str
                            })
                        except (ValueError, IndexError) as e:
                            print(f"⚠️ Skipping invalid line: {line} ({e})")

                i += 1

            # Add final block
            if current_block and current_block['measurements']:
                blocks.append(current_block)

        except Exception as e:
            print(f"❌ Error parsing {filepath}: {e}")
            return []

        return blocks

    @staticmethod
    def blocks_to_dataframe(blocks):
        """
        Convert parsed blocks to pandas DataFrame.

        Returns:
        - DataFrame with columns: date, sensor_type, amplitude_primary, amplitude_secondary
        """
        rows = []

        for block in blocks:
            sensor_type = block['sensor_type']

            for measurement in block['measurements']:
                rows.append({
                    'date': measurement['date'],
                    'date_str': measurement['date_str'],
                    'sensor_type': sensor_type,
                    'amplitude_primary': measurement['val1'],  # Usually RMS or peak
                    'amplitude_secondary': measurement['val2']  # Usually secondary measurement
                })

        if not rows:
            return pd.DataFrame()

        df = pd.DataFrame(rows)
        df = df.sort_values('date').reset_index(drop=True)

        return df

    @staticmethod
    def process_dataset_file(filepath, machine_name="Dataset_Machine"):
        """
        Process complete Dataset.txt file into machine history.

        Returns:
        - DataFrame ready for predictive maintenance analysis
        """
        print(f"📄 Processing Brazilian dataset: {filepath}")

        # Parse the file
        blocks = BrazilianDataProcessor.parse_dataset_txt(filepath)

        if not blocks:
            print("❌ No valid data blocks found")
            return None

        print(f"✓ Found {len(blocks)} measurement blocks")

        # Convert to DataFrame
        df = BrazilianDataProcessor.blocks_to_dataframe(blocks)

        if df.empty:
            print("❌ No valid measurements found")
            return None

        # Add machine identifier
        df['machine'] = machine_name
        df['filename'] = os.path.basename(filepath)

        # Calculate derived features
        df['amplitude_max'] = df[['amplitude_primary', 'amplitude_secondary']].max(axis=1)
        df['amplitude_avg'] = df[['amplitude_primary', 'amplitude_secondary']].mean(axis=1)

        print(f"✓ Processed {len(df)} measurements from {df['date'].min().date()} to {df['date'].max().date()}")
        print(f"  Sensor types: {', '.join(df['sensor_type'].unique())}")
        print(f"  Amplitude range: {df['amplitude_primary'].min():.2f} - {df['amplitude_primary'].max():.2f} mm/s")
        return df

    @staticmethod
    def create_time_series_for_prediction(df, target_column='amplitude_max'):
        """
        Create time-series format suitable for predictive maintenance.

        Returns:
        - DataFrame with date index and amplitude column
        """
        if df.empty or target_column not in df.columns:
            return None

        # Group by date and take maximum amplitude (worst case)
        time_series = df.groupby('date')[target_column].max().reset_index()

        # Sort by date
        time_series = time_series.sort_values('date')

        # Rename for consistency with existing system
        time_series = time_series.rename(columns={
            'date': 'date',
            target_column: 'amplitude'
        })

        return time_series

    @staticmethod
    def validate_brazilian_format(filepath):
        """
        Validate that file follows Brazilian data format.

        Returns:
        - True if valid, False otherwise
        - Error message if invalid
        """
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                lines = f.readlines()[:20]  # Check first 20 lines

            has_v_header = any('V' == line.strip() for line in lines)
            has_e_header = any('E' == line.strip() for line in lines)

            if not (has_v_header or has_e_header):
                return False, "No 'V' or 'E' sensor headers found"

            # Check for Brazilian date format
            date_pattern = re.compile(r'\d{2}/\d{2}/\d{4}')
            has_dates = any(date_pattern.search(line) for line in lines)

            if not has_dates:
                return False, "No DD/MM/YYYY dates found"

            # Check for comma decimals
            comma_pattern = re.compile(r'\d+,\d+')
            has_commas = any(comma_pattern.search(line) for line in lines)

            if not has_commas:
                return False, "No comma decimal separators found"

            return True, "Valid Brazilian format"

        except Exception as e:
            return False, f"Error reading file: {e}"


# =============================================================================
# USAGE EXAMPLE
# =============================================================================

if __name__ == "__main__":
    print("Brazilian Data Processor initialized")
    print("Use BrazilianDataProcessor to process Dataset.txt files")
