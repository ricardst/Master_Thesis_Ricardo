#!/usr/bin/env python3
"""
Script to correct timing in Final_Labels.csv using synchronization data from SyncTimesVideos.csv.

This script applies time shift and drift corrections to label timestamps based on video synchronization data.
The approach is similar to the one used in raw_data_processor.py for sensor data correction.
"""

import pandas as pd
import numpy as np
import os
import re
import logging
from datetime import timedelta

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def correct_timestamp_drift(timestamp_seconds, t0_seconds, t1_seconds, total_drift_seconds):
    """
    Adjusts timestamp for linear drift correction.
    
    Args:
        timestamp_seconds: Time in seconds to be corrected
        t0_seconds: Start time reference in seconds
        t1_seconds: End time reference in seconds  
        total_drift_seconds: Total drift to be applied over the interval
        
    Returns:
        Corrected timestamp in seconds
    """
    total_interval = t1_seconds - t0_seconds
    if total_interval == 0:
        return timestamp_seconds
    
    elapsed = timestamp_seconds - t0_seconds
    drift_offset = total_drift_seconds * (elapsed / total_interval)
    return timestamp_seconds + drift_offset


def extract_subject_id(video_filename):
    """
    Extract subject ID from video filename.
    
    Args:
        video_filename: Video filename (e.g., 'OutSense-036GX040233.MP4')
        
    Returns:
        Subject ID (e.g., '036')
    """
    # Extract subject ID from video filename pattern like OutSense-036GX040233.MP4
    match = re.search(r'OutSense-(\d+)', video_filename)
    if match:
        return match.group(1)
    else:
        logging.warning(f"Could not extract subject ID from filename: {video_filename}")
        return None


def load_sync_data(sync_file_path):
    """
    Load synchronization data from SyncTimesVideos.csv.
    
    Args:
        sync_file_path: Path to SyncTimesVideos.csv
        
    Returns:
        Dictionary mapping subject IDs to sync parameters
    """
    try:
        sync_df = pd.read_csv(sync_file_path)
        logging.info(f"Loaded sync data with {len(sync_df)} entries")
        
        sync_dict = {}
        for _, row in sync_df.iterrows():
            subject_id = str(row['Patient']).zfill(3)  # Ensure 3-digit format
            
            # Parse time shift (Delta Start [s])
            delta_start = row.get('Delta Start [s]', 0)
            if pd.isna(delta_start):
                delta_start = 0
            
            # Parse time drift (Delta End [s])
            delta_end = row.get('Delta End [s]', 0)
            if pd.isna(delta_end):
                delta_end = 0
            
            # Calculate net drift (subtract delta start as specified)
            net_drift = delta_end - delta_start
            
            # Parse start and end times for drift calculation
            start_time_str = str(row.get('Start Time', ''))
            end_time_str = str(row.get('End Time', ''))
            
            sync_dict[subject_id] = {
                'time_shift_seconds': float(delta_start),
                'net_drift_seconds': float(net_drift),
                'start_time_str': start_time_str,
                'end_time_str': end_time_str,
                'has_end_time': not pd.isna(row.get('End Time')) and str(row.get('End Time')) != 'nan'
            }
            
            logging.debug(f"Subject {subject_id}: shift={delta_start}s, net_drift={net_drift}s")
        
        return sync_dict
        
    except Exception as e:
        logging.error(f"Error loading sync data: {e}")
        return {}


def apply_time_corrections(labels_df, sync_dict):
    """
    Apply time shift and drift corrections to the labels dataframe.
    
    Args:
        labels_df: DataFrame with label data
        sync_dict: Dictionary with sync parameters per subject
        
    Returns:
        Corrected labels dataframe
    """
    corrected_df = labels_df.copy()
    
    # Add subject ID column
    corrected_df['Subject_ID'] = corrected_df['Video_File'].apply(extract_subject_id)
    
    # Convert time columns to datetime
    time_columns = ['Real_Start_Time', 'Real_End_Time']
    for col in time_columns:
        if col in corrected_df.columns:
            corrected_df[col] = pd.to_datetime(corrected_df[col], errors='coerce')
    
    corrections_applied = 0
    
    for subject_id in corrected_df['Subject_ID'].dropna().unique():
        if subject_id not in sync_dict:
            logging.warning(f"No sync data found for subject {subject_id}")
            continue
            
        subject_mask = corrected_df['Subject_ID'] == subject_id
        subject_sync = sync_dict[subject_id]
        
        # Apply time shift
        time_shift_seconds = subject_sync['time_shift_seconds']
        if time_shift_seconds != 0:
            shift_delta = pd.Timedelta(seconds=time_shift_seconds)
            for col in time_columns:
                if col in corrected_df.columns:
                    corrected_df.loc[subject_mask, col] += shift_delta
            
            logging.info(f"Applied {time_shift_seconds}s time shift to subject {subject_id}")
        
        # Apply drift correction if we have end time data
        net_drift_seconds = subject_sync['net_drift_seconds']
        if net_drift_seconds != 0 and subject_sync['has_end_time']:
            # Get the time range for this subject to apply drift correction
            subject_labels = corrected_df[subject_mask].copy()
            
            if not subject_labels.empty:
                # Find min and max times for drift calculation
                min_time = subject_labels['Real_Start_Time'].min()
                max_time = subject_labels['Real_End_Time'].max()
                
                if pd.notna(min_time) and pd.notna(max_time):
                    # Convert to seconds since epoch for drift calculation
                    t0_seconds = min_time.timestamp()
                    t1_seconds = max_time.timestamp()
                    
                    # Apply drift correction to each timestamp
                    for col in time_columns:
                        if col in corrected_df.columns:
                            timestamps = corrected_df.loc[subject_mask, col]
                            corrected_timestamps = timestamps.apply(
                                lambda ts: pd.Timestamp.fromtimestamp(
                                    correct_timestamp_drift(
                                        ts.timestamp(), t0_seconds, t1_seconds, net_drift_seconds
                                    )
                                ) if pd.notna(ts) else ts
                            )
                            corrected_df.loc[subject_mask, col] = corrected_timestamps
                    
                    logging.info(f"Applied {net_drift_seconds}s drift correction to subject {subject_id}")
                    corrections_applied += 1
        
        elif net_drift_seconds != 0:
            logging.warning(f"Subject {subject_id} has drift ({net_drift_seconds}s) but no end time data")
    
    logging.info(f"Applied corrections to {corrections_applied} subjects")
    return corrected_df


def main():
    """Main function to execute the label timing correction."""
    
    # File paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    labels_file = os.path.join(script_dir, 'Final_Labels.csv')
    sync_file = os.path.join(script_dir, 'SyncTimesVideos.csv')
    output_file = os.path.join(script_dir, 'Final_Labels_corrected.csv')
    
    # Check if input files exist
    if not os.path.exists(labels_file):
        logging.error(f"Labels file not found: {labels_file}")
        return
        
    if not os.path.exists(sync_file):
        logging.error(f"Sync file not found: {sync_file}")
        return
    
    # Load data
    logging.info("Loading label data...")
    try:
        labels_df = pd.read_csv(labels_file)
        logging.info(f"Loaded {len(labels_df)} label entries")
    except Exception as e:
        logging.error(f"Error loading labels file: {e}")
        return
    
    logging.info("Loading sync data...")
    sync_dict = load_sync_data(sync_file)
    
    if not sync_dict:
        logging.error("No sync data loaded, cannot proceed")
        return
    
    # Apply corrections
    logging.info("Applying time corrections...")
    corrected_df = apply_time_corrections(labels_df, sync_dict)
    
    # Remove the temporary Subject_ID column before saving
    if 'Subject_ID' in corrected_df.columns:
        corrected_df = corrected_df.drop('Subject_ID', axis=1)
    
    # Save corrected data
    try:
        corrected_df.to_csv(output_file, index=False)
        logging.info(f"Saved corrected labels to: {output_file}")
        
        # Print summary statistics
        logging.info("Summary:")
        logging.info(f"  Original entries: {len(labels_df)}")
        logging.info(f"  Corrected entries: {len(corrected_df)}")
        logging.info(f"  Unique subjects processed: {len(sync_dict)}")
        
    except Exception as e:
        logging.error(f"Error saving corrected labels: {e}")


if __name__ == '__main__':
    main()
