#!/usr/bin/env python3
"""
Script to add metadata information to coverage CSV files.
Adds "Last Modified" and "DoubleML Version" columns by reading from corresponding metadata files.
"""

import os
import pandas as pd
from pathlib import Path
import re

def find_metadata_file(coverage_file_path):
    """Find the corresponding metadata file for a coverage CSV file."""
    coverage_path = Path(coverage_file_path)
    
    # Try different metadata file naming patterns
    possible_metadata_files = [
        # Pattern 1: replace _coverage.csv with _metadata.csv
        coverage_path.parent / coverage_path.name.replace('_coverage.csv', '_metadata.csv'),
        
        # Pattern 2: for files like did_cs_atte_coverage.csv -> did_cs_atte_coverage_metadata.csv
        coverage_path.parent / (coverage_path.stem + '_metadata.csv'),
        
        # Pattern 3: extract base name and add _metadata.csv
        # e.g., irm_ate_coverage.csv -> irm_ate_metadata.csv
        coverage_path.parent / (coverage_path.stem.replace('_coverage', '') + '_metadata.csv'),
    ]
    
    # Check which metadata file exists
    for metadata_file in possible_metadata_files:
        if metadata_file.exists():
            return metadata_file
    
    return None

def read_metadata(metadata_file_path):
    """Read metadata from CSV file and extract relevant information."""
    try:
        metadata_df = pd.read_csv(metadata_file_path)
        
        if len(metadata_df) == 0:
            return None, None
            
        # Extract DoubleML Version and Date
        doubleml_version = metadata_df.iloc[0]['DoubleML Version'] if 'DoubleML Version' in metadata_df.columns else 'Unknown'
        
        # Try to extract date from different possible columns
        date = None
        if 'Date' in metadata_df.columns:
            date = metadata_df.iloc[0]['Date']
        
        return doubleml_version, date
        
    except Exception as e:
        print(f"Error reading metadata file {metadata_file_path}: {e}")
        return None, None

def add_metadata_to_coverage_file(coverage_file_path):
    """Add metadata columns to a coverage CSV file."""
    print(f"Processing: {coverage_file_path}")
    
    # Find corresponding metadata file
    metadata_file = find_metadata_file(coverage_file_path)
    
    if metadata_file is None:
        print(f"  No metadata file found for {coverage_file_path}")
        return False
    
    print(f"  Found metadata file: {metadata_file}")
    
    # Read metadata
    doubleml_version, last_modified = read_metadata(metadata_file)
    
    if doubleml_version is None:
        print(f"  Could not extract metadata from {metadata_file}")
        return False
    
    # Read coverage file
    try:
        coverage_df = pd.read_csv(coverage_file_path)
    except Exception as e:
        print(f"  Error reading coverage file: {e}")
        return False
    
    # Check if metadata columns already exist
    if 'DoubleML Version' in coverage_df.columns and 'Last Modified' in coverage_df.columns:
        print(f"  Metadata columns already exist, skipping...")
        return True
    
    # Add metadata columns
    coverage_df['DoubleML Version'] = doubleml_version
    if last_modified:
        coverage_df['Last Modified'] = last_modified
    
    # Save the updated file
    try:
        coverage_df.to_csv(coverage_file_path, index=False)
        print(f"  Successfully added metadata columns")
        return True
    except Exception as e:
        print(f"  Error saving file: {e}")
        return False

def main():
    """Main function to process all coverage files."""
    # Find all coverage CSV files
    results_dir = Path("/home/svenklaassen/github/doubleml-coverage/results")
    
    coverage_files = []
    for pattern in ["**/*_coverage.csv", "**/did_*_coverage.csv"]:
        coverage_files.extend(results_dir.glob(pattern))
    
    # Remove duplicates
    coverage_files = list(set(coverage_files))
    
    print(f"Found {len(coverage_files)} coverage files to process\n")
    
    success_count = 0
    for coverage_file in sorted(coverage_files):
        if add_metadata_to_coverage_file(coverage_file):
            success_count += 1
        print()  # Empty line for readability
    
    print(f"Successfully processed {success_count}/{len(coverage_files)} files")

if __name__ == "__main__":
    main()