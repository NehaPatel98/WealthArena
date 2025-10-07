#!/usr/bin/env python3
"""
Excel to CSV Converter for Reddit Data
This script helps convert your Excel file to CSV format for import into WealthArena
"""

import pandas as pd
import sys
import os

def convert_excel_to_csv(excel_file_path, csv_file_path=None):
    """
    Convert Excel file to CSV format
    
    Args:
        excel_file_path: Path to the Excel file
        csv_file_path: Path for the output CSV file (optional)
    """
    try:
        # Read Excel file
        print(f"Reading Excel file: {excel_file_path}")
        df = pd.read_excel(excel_file_path)
        
        # If no output path specified, create one
        if csv_file_path is None:
            csv_file_path = excel_file_path.replace('.xlsx', '.csv').replace('.xls', '.csv')
        
        # Save as CSV
        print(f"Converting to CSV: {csv_file_path}")
        df.to_csv(csv_file_path, index=False)
        
        print(f"✅ Successfully converted to CSV!")
        print(f"📊 Rows: {len(df)}")
        print(f"📋 Columns: {list(df.columns)}")
        print(f"💾 Output file: {csv_file_path}")
        
        return csv_file_path
        
    except Exception as e:
        print(f"❌ Error converting file: {e}")
        return None

def main():
    if len(sys.argv) < 2:
        print("Usage: python convert_excel_to_csv.py <excel_file_path> [csv_file_path]")
        print("Example: python convert_excel_to_csv.py reddit_data.xlsx reddit_data.csv")
        return
    
    excel_file = sys.argv[1]
    csv_file = sys.argv[2] if len(sys.argv) > 2 else None
    
    if not os.path.exists(excel_file):
        print(f"❌ File not found: {excel_file}")
        return
    
    convert_excel_to_csv(excel_file, csv_file)

if __name__ == "__main__":
    main()
