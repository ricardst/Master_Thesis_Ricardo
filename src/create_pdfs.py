import os
import argparse
import numpy as np
import pandas as pd
import yaml
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from datetime import datetime

def load_npy_data(file_path):
    """Loads data from a .npy file."""
    if not os.path.exists(file_path):
        print(f"Error: File not found - {file_path}")
        return None
    try:
        data = np.load(file_path, allow_pickle=True)
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

def plot_window_to_pdf_page(fig, window_data, all_sensor_names,
                            sampling_rate, window_start_time_str, subject_id,
                            window_label, label_colors):
    """Plots the sensor data for a given window onto the provided figure."""
    ax = fig.gca()

    num_samples = window_data.shape[0]
    time_axis = np.arange(num_samples) / sampling_rate  # Time in seconds

    # Plot all sensor channels
    for sensor_idx in range(window_data.shape[1]):
        sensor_name = all_sensor_names[sensor_idx] if sensor_idx < len(all_sensor_names) else f"Sensor {sensor_idx+1}"
        ax.plot(time_axis, window_data[:, sensor_idx], label=sensor_name, linewidth=0.5)

    ax.set_title(f"Subject: {subject_id} | Window Start: {window_start_time_str} | Label: {window_label}", fontsize=10)
    ax.set_xlabel("Time within window (seconds)", fontsize=8)
    ax.set_ylabel("Sensor Value", fontsize=8)
    ax.tick_params(axis='both', which='major', labelsize=7)

    # Add colored background for the window label
    activity_name = str(window_label)  # Ensure it's a string for dict lookup
    activity_color = 'gray'  # Default color if not found
    if label_colors and isinstance(label_colors, dict):
        activity_color = label_colors.get(activity_name, 'gray')
        if activity_name not in label_colors:
            # This warning can be very verbose if many labels are missing from the config.
            # Consider logging to a file or a less frequent console print if it becomes an issue.
            # print(f"Warning: Label '{activity_name}' not found in plot_label_colors from config. Using default color 'gray'.")
            pass 
    elif not label_colors: # label_colors is None or empty
        # print("Warning: plot_label_colors not found or not a dictionary in config. Using default color 'gray' for label background.")
        pass


    ax.axvspan(time_axis[0], time_axis[-1], color=activity_color, alpha=0.3, zorder=0)

    # Add legend - make it compact if many sensors
    handles, labels = ax.get_legend_handles_labels()
    if handles: # Only show legend if there are labeled plots
        # Adjust ncol dynamically based on the number of sensors to prevent overly wide legends
        num_sensors = window_data.shape[1]
        legend_cols = max(1, num_sensors // 10) # Example: 1 col for <10 sensors, 2 for 10-19, etc.
        ax.legend(handles, labels, loc='upper right', fontsize='xx-small', ncol=legend_cols)
    ax.grid(True, linestyle='--', alpha=0.6)


def main():
    parser = argparse.ArgumentParser(description="Create PDFs of sensor data windows for each subject.")
    parser.add_argument('--data_dir', type=str, default='features',
                        help="Directory containing window data files (X_windows_raw.npy, etc.). Default: 'features'")
    parser.add_argument('--config_file', type=str, default='config.yaml',
                        help="Path to the configuration file (config.yaml). Default: 'config.yaml'")
    parser.add_argument('--output_dir', type=str, default='WindowPDFs',
                        help="Directory to save the output PDFs. Default: 'WindowPDFs'")
    
    args = parser.parse_args()

    # --- Load Configuration ---
    config = load_yaml_config(args.config_file)
    if config is None:
        print("Exiting due to config loading error.")
        return

    try:
        all_sensor_names = config['sensor_columns_original']
        # Use 'downsample_freq' as sampling_rate, default if not found
        sampling_rate = config.get('downsample_freq', 25) 
        plot_label_colors = config.get('plot_label_colors', {}) # Default to empty dict if not in config
    except KeyError as e:
        print(f"Error: Missing required key in config file: {e}. 'sensor_columns_original' is mandatory.")
        print("Optional keys: 'downsample_freq', 'plot_label_colors'.")
        return

    # --- Load Window Data ---
    # Construct absolute paths or ensure data_dir is correctly relative to script execution
    # Assuming script is run from the root directory /scai_data3/scratch/stirnimann_r/
    base_data_dir = args.data_dir 
    x_windows_raw_path = os.path.join(base_data_dir, 'X_windows_raw.npy')
    subject_ids_path = os.path.join(base_data_dir, 'subject_ids_windows.npy')
    start_times_path = os.path.join(base_data_dir, 'window_start_times.npy')
    y_windows_path = os.path.join(base_data_dir, 'y_windows.npy')

    print(f"Attempting to load data from directory: {os.path.abspath(base_data_dir)}")

    X_windows_raw = load_npy_data(x_windows_raw_path)
    subject_ids_windows = load_npy_data(subject_ids_path)
    window_start_times = load_npy_data(start_times_path)
    y_windows = load_npy_data(y_windows_path)

    if any(data is None for data in [X_windows_raw, subject_ids_windows, window_start_times, y_windows]):
        print("Failed to load one or more essential data files. Please check paths and file existence. Exiting.")
        return
    
    print(f"Data loaded successfully. X_windows_raw shape: {X_windows_raw.shape}, y_windows shape: {y_windows.shape}")
    print(f"Subject IDs array shape: {subject_ids_windows.shape}, Start times array shape: {window_start_times.shape}")


    # --- Create Output Directory ---
    # Assuming script is run from the root directory /scai_data3/scratch/stirnimann_r/
    output_pdf_dir = args.output_dir
    os.makedirs(output_pdf_dir, exist_ok=True)
    print(f"Output PDFs will be saved to: {os.path.abspath(output_pdf_dir)}")

    # --- Process by Subject ---
    # Flatten subject_ids_windows if it's not already 1D, and get unique subjects
    unique_subject_ids = pd.Series(subject_ids_windows.flatten()).unique() 
    unique_subject_ids = [sid for sid in unique_subject_ids if pd.notna(sid)] # Filter out None/NaN if any
    
    if not unique_subject_ids:
        print("No subject IDs found in 'subject_ids_windows.npy'. Cannot generate PDFs.")
        return

    print(f"Found {len(unique_subject_ids)} unique subjects to process: {unique_subject_ids}")

    for subject_id in unique_subject_ids:
        subject_id_str = str(subject_id).replace(' ', '_').replace('/', '_') # Sanitize subject_id for filename
        print(f"\\nProcessing subject: {subject_id_str}...")
        pdf_path = os.path.join(output_pdf_dir, f"{subject_id_str}_windows.pdf")
        
        # Find indices for the current subject
        s_ids_flat = subject_ids_windows.flatten()
        subject_window_indices = [i for i, sid_loop in enumerate(s_ids_flat) if sid_loop == subject_id]
        
        if not subject_window_indices:
            print(f"No windows found for subject {subject_id_str}. Skipping PDF generation for this subject.")
            continue

        print(f"Found {len(subject_window_indices)} windows for subject {subject_id_str}. Creating PDF...")

        # A4 landscape: 11.69 x 8.27 inches.
        # We will create one figure per page inside the loop.
        with PdfPages(pdf_path) as pdf:
            for count, window_idx in enumerate(subject_window_indices):
                if (count + 1) % 50 == 0: # Progress update every 50 windows
                    print(f"  Plotting window {count + 1}/{len(subject_window_indices)} for subject {subject_id_str}...")

                target_window_data = X_windows_raw[window_idx]
                
                # Handle cases where y_windows might have an extra dimension or be a single value
                current_label_raw = y_windows[window_idx]
                if isinstance(current_label_raw, np.ndarray):
                    target_window_label = current_label_raw[0] if current_label_raw.size > 0 else "N/A"
                else:
                    target_window_label = current_label_raw if pd.notna(current_label_raw) else "N/A"


                window_actual_start_time_ts = pd.Timestamp(window_start_times[window_idx])
                window_start_time_str = window_actual_start_time_ts.strftime('%Y-%m-%d %H:%M:%S.%f')

                # Create a new figure for each page, A4 landscape
                fig = plt.figure(figsize=(11.69, 8.27)) 

                plot_window_to_pdf_page(fig, target_window_data, all_sensor_names,
                                        sampling_rate, window_start_time_str, subject_id_str,
                                        target_window_label, plot_label_colors)
                
                pdf.savefig(fig, orientation='landscape', bbox_inches='tight') # Save the current figure to the PDF
                plt.close(fig) # Close the figure to free memory

        print(f"PDF saved for subject {subject_id_str} to {pdf_path}")

    print("\\nAll subjects processed.")

if __name__ == '__main__':
    # Ensure Matplotlib uses a non-interactive backend if running in an environment without a display
    import matplotlib
    matplotlib.use('Agg') # Use 'Agg' for non-interactive plotting to files
    main()
