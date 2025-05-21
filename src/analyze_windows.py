import argparse
import os
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yaml
from datetime import datetime

def load_pickle_data(file_path):
    """Loads data from a pickle file."""
    if not os.path.exists(file_path):
        print(f"Error: File not found - {file_path}")
        return None
    try:
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        return data
    except Exception as e:
        print(f"Error loading pickle file {file_path}: {e}")
        return None

def load_npy_data(file_path):
    """Loads data from a .npy file."""
    if not os.path.exists(file_path):
        print(f"Error: File not found - {file_path}")
        return None
    try:
        data = np.load(file_path, allow_pickle=True)  # allow_pickle=True for object arrays like subject_ids
        return data
    except Exception as e:
        print(f"Error loading .npy file {file_path}: {e}")
        return None

def load_yaml_config(file_path):
    """Loads data from a YAML file."""
    if not os.path.exists(file_path):
        print(f"Error: File not found - {file_path}")
        return None
    try:
        with open(file_path, 'r') as f:
            config = yaml.safe_load(f)
        return config
    except Exception as e:
        print(f"Error loading YAML file {file_path}: {e}")
        return None

def find_target_window_index(subject_id_target, time_target_dt,
                             subject_ids_all, start_times_all, end_times_all):
    """Finds the index of the window matching the subject and time."""
    for i in range(len(subject_ids_all)):
        if subject_ids_all[i] == subject_id_target:
            # Ensure time_target_dt is timezone-naive if start/end times are, or vice-versa
            # Numpy datetime64 objects are typically timezone-naive
            current_start_time = pd.Timestamp(start_times_all[i])
            current_end_time = pd.Timestamp(end_times_all[i])
            
            if current_start_time.tzinfo is not None and time_target_dt.tzinfo is None:
                time_target_dt = time_target_dt.tz_localize(current_start_time.tzinfo)
            elif current_start_time.tzinfo is None and time_target_dt.tzinfo is not None:
                time_target_dt = time_target_dt.tz_localize(None)

            if current_start_time <= time_target_dt <= current_end_time:
                return i
    return None

def plot_window_data(window_data, selected_sensor_indices, all_sensor_names,
                     sampling_rate, window_start_time_str, subject_id,
                     window_label, label_colors,
                     output_plot_path=None):
    """Plots the selected sensor data for the given window with label visualization."""
    plt.figure(figsize=(15, 7))
    ax = plt.gca()  # Get current axes

    num_samples = window_data.shape[0]
    time_axis = np.arange(num_samples) / sampling_rate  # Time in seconds

    for sensor_idx in selected_sensor_indices:
        sensor_name = all_sensor_names[sensor_idx]
        plt.plot(time_axis, window_data[:, sensor_idx], label=sensor_name)

    plt.title(f"Sensor Data Window for Subject: {subject_id}\nStart Time: {window_start_time_str}")
    plt.xlabel("Time within window (seconds)")
    plt.ylabel("Sensor Value")

    # Add colored background for the window label
    activity_name = str(window_label)  # Ensure it's a string for dict lookup
    activity_color = 'gray'  # Default color if not found
    if label_colors and isinstance(label_colors, dict):
        activity_color = label_colors.get(activity_name, 'gray')
        if activity_name not in label_colors:
            print(f"Warning: Label '{activity_name}' not found in plot_label_colors from config. Using default color 'gray'.")
    else:
        print("Warning: plot_label_colors not found or not a dictionary in config. Using default color 'gray' for label background.")

    ax.axvspan(time_axis[0], time_axis[-1], color=activity_color, alpha=0.3, zorder=0, label=f'Activity: {activity_name}')

    plt.legend()
    plt.grid(True)

    if output_plot_path:
        try:
            plt.savefig(output_plot_path)
            print(f"Plot saved to {output_plot_path}")
        except Exception as e:
            print(f"Error saving plot to {output_plot_path}: {e}")
    else:
        plt.show()

def main():
    parser = argparse.ArgumentParser(description="Analyze and plot specific data windows from feature engineering output.")
    parser.add_argument('--data_dir', type=str, default='features',
                        help="Directory containing window data files (e.g., X_windows_raw.npy). Default: 'features'")
    parser.add_argument('--config_file', type=str, default='config.yaml',
                        help="Path to the configuration file (config.yaml). Default: 'config.yaml'")
    parser.add_argument('--subject_id', type=str, required=True,
                        help="Subject ID to analyze (e.g., 'OutSense-036').")
    parser.add_argument('--time', type=str, required=True,
                        help="Timestamp within the desired window (format: 'YYYY-MM-DD HH:MM:SS' or 'YYYY-MM-DD HH:MM:SS.ffffff').")
    parser.add_argument('--sensors', type=str, required=True,
                        help="Comma-separated list of sensor names to plot (e.g., 'wrist_acc_x,wrist_acc_y').")
    parser.add_argument('--output_plot', type=str, default=None,
                        help="Path to save the plot image. If not provided, plot will be shown interactively.")
    
    args = parser.parse_args()

    # --- Load Configuration ---
    config = load_yaml_config(args.config_file)
    if config is None:
        return

    try:
        # Stage 1 for sensor_columns_original, Stage 0 for downsample_freq, Stage 2 for window_size
        all_sensor_names = config['sensor_columns_original']
        sampling_rate = config.get('downsample_freq', 25) # Default from config if available
        # window_size = config['window_size'] # Not directly needed for plotting if X_windows_raw has fixed size
    except KeyError as e:
        print(f"Error: Missing required key in config file: {e}. Needed: 'sensor_columns_original', 'downsample_freq'.")
        return

    # --- Load Window Data ---
    x_windows_raw_path = os.path.join(args.data_dir, 'X_windows_raw.npy')
    subject_ids_path = os.path.join(args.data_dir, 'subject_ids_windows.npy')
    start_times_path = os.path.join(args.data_dir, 'window_start_times.npy')
    end_times_path = os.path.join(args.data_dir, 'window_end_times.npy')
    y_windows_path = os.path.join(args.data_dir, 'y_windows.npy')  # Path for labels

    X_windows_raw = load_npy_data(x_windows_raw_path)
    subject_ids_windows = load_npy_data(subject_ids_path)
    window_start_times = load_npy_data(start_times_path)
    window_end_times = load_npy_data(end_times_path)
    y_windows = load_npy_data(y_windows_path)  # Load labels

    if any(data is None for data in [X_windows_raw, subject_ids_windows, window_start_times, window_end_times, y_windows]):
        print("Failed to load one or more essential data files (including y_windows.npy). Exiting.")
        return

    # --- Parse User Inputs ---
    try:
        target_time_dt = pd.to_datetime(args.time)
    except ValueError as e:
        print(f"Error: Invalid time format for --time argument. Use 'YYYY-MM-DD HH:MM:SS'. Error: {e}")
        return

    selected_sensor_names_user = [s.strip() for s in args.sensors.split(',')]
    selected_sensor_indices = []
    invalid_sensors = []
    for sensor_name in selected_sensor_names_user:
        try:
            idx = all_sensor_names.index(sensor_name)
            selected_sensor_indices.append(idx)
        except ValueError:
            invalid_sensors.append(sensor_name)
    
    if invalid_sensors:
        print(f"Error: The following sensor names are invalid or not found in config's 'sensor_columns_original': {', '.join(invalid_sensors)}")
        print(f"Available sensors are: {', '.join(all_sensor_names)}")
        return
    if not selected_sensor_indices:
        print("Error: No valid sensors selected for plotting.")
        return

    # --- Find Target Window ---
    print(f"Searching for window for Subject ID: {args.subject_id} around time: {target_time_dt.strftime('%Y-%m-%d %H:%M:%S.%f')}")
    window_idx = find_target_window_index(args.subject_id, target_time_dt,
                                          subject_ids_windows, window_start_times, window_end_times)

    if window_idx is None:
        print(f"No window found for Subject ID '{args.subject_id}' containing the time '{args.time}'.")
        print("Please check subject ID, timestamp, and data availability.")
        return

    print(f"Found window at index {window_idx}.")
    target_window_data = X_windows_raw[window_idx]
    target_window_label = y_windows[window_idx]  # Get the label for the window
    window_actual_start_time = pd.Timestamp(window_start_times[window_idx])
    window_actual_end_time = pd.Timestamp(window_end_times[window_idx])
    
    print(f"Window Start: {window_actual_start_time.strftime('%Y-%m-%d %H:%M:%S.%f')}, End: {window_actual_end_time.strftime('%Y-%m-%d %H:%M:%S.%f')}, Label: {target_window_label}")

    # Retrieve plot_label_colors from config
    plot_label_colors = config.get('plot_label_colors', {})  # Get from config, default to empty dict

    # --- Plot Data ---
    plot_window_data(target_window_data, selected_sensor_indices, all_sensor_names,
                     sampling_rate, window_actual_start_time.strftime('%Y-%m-%d %H:%M:%S.%f'),
                     args.subject_id, 
                     target_window_label, plot_label_colors,
                     output_plot_path='output_plot.png' if args.output_plot is None else args.output_plot)

if __name__ == '__main__':
    main()
