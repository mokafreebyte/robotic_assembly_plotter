#!/usr/bin/env python3
# Plotter and analysis script for robotic insertion task logs
# Usage: python plotter.py [plot_pose_force|plot_trajectory|analyze_results] [--data_dir ...] [--results_dir ...] [--files ...]

import os
import sys
import argparse
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yaml
import logging
from datetime import datetime

# =====================
# Default Variables
# =====================
DEFAULT_CONFIG_FILE = os.path.join(os.path.dirname(__file__), 'config.yaml')

# Fallback defaults if config file is not available
FALLBACK_DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../data'))
FALLBACK_RESULTS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../results'))
FALLBACK_PLOTS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../plots'))

# Setup logging
def setup_logging(verbosity):
	"""Setup logging based on verbosity level"""
	levels = {
		0: logging.WARNING,
		1: logging.INFO, 
		2: logging.DEBUG
	}
	level = levels.get(verbosity, logging.DEBUG)
	
	logging.basicConfig(
		level=level,
		format='%(levelname)s: %(message)s',
		handlers=[logging.StreamHandler(sys.stdout)]
	)
	return logging.getLogger(__name__)

def load_config(config_file):
	"""Load configuration from YAML file"""
	try:
		with open(config_file, 'r') as f:
			config = yaml.safe_load(f)
		logging.info(f"Loaded configuration from {config_file}")
		return config
	except FileNotFoundError:
		logging.warning(f"Config file {config_file} not found, using defaults")
		return {}
	except yaml.YAMLError as e:
		logging.error(f"Error parsing config file {config_file}: {e}")
		return {}

def get_directories_from_config(config):
	"""Get directory paths from config, with fallbacks"""
	script_dir = os.path.dirname(__file__)
	dirs_config = config.get('directories', {})
	
	def resolve_path(path, fallback):
		if path and not os.path.isabs(path):
			# Relative path - make it relative to script directory
			return os.path.abspath(os.path.join(script_dir, path))
		elif path:
			# Absolute path
			return os.path.abspath(path)
		else:
			# Use fallback
			return fallback
	
	data_dir = resolve_path(dirs_config.get('data_dir'), FALLBACK_DATA_DIR)
	results_dir = resolve_path(dirs_config.get('results_dir'), FALLBACK_RESULTS_DIR)
	plots_dir = resolve_path(dirs_config.get('plots_dir'), FALLBACK_PLOTS_DIR)
	
	return data_dir, results_dir, plots_dir

def list_csv_files(data_dir):
	return sorted(glob.glob(os.path.join(data_dir, '*.csv')))

def list_yaml_files(results_dir):
	return sorted(glob.glob(os.path.join(results_dir, '*.yaml')))

def load_and_process_trial(file, sampling_freq):
	"""Load and process a single trial file."""
	logging.debug(f"Loading trial file: {file}")
	df = pd.read_csv(file)
	
	# Create proper time axis - handle duplicate timestamps
	if 'time' in df.columns:
		# Check if we have duplicate timestamps
		time_values = df['time'].values
		if len(np.unique(time_values)) < len(time_values):
			logging.debug(f"Detected duplicate timestamps in {file}, creating sequential time axis")
			# Create time axis assuming constant sampling rate
			time = np.arange(len(df)) / sampling_freq  # Convert to seconds
		else:
			time = df['time'] - df['time'].iloc[0]  # Start from 0
			logging.debug(f"Using original timestamps from {file}")
	else:
		# Fallback: create sequential time assuming default sampling frequency
		time = np.arange(len(df)) / sampling_freq
		logging.debug(f"No time column found, creating sequential time axis")
	
	# Extract required columns
	pose_x = df.get('ee_pose_lin_x', None)
	fx = df.get('fts_wrench_lin_x', None)
	fy = df.get('fts_wrench_lin_y', None)
	fz = df.get('fts_wrench_lin_z', None)
	
	if pose_x is None or fx is None or fy is None or fz is None:
		logging.warning(f"Missing columns in {file}, skipping.")
		return None
	
	force_norm = np.sqrt(fx**2 + fy**2 + fz**2)
	
	return {
		'file': file,
		'time': time,
		'pose_x': pose_x.values,
		'force_norm': force_norm.values
	}

def trim_trial_duration(trial_data, max_duration):
	"""Trim trial data to maximum duration."""
	if max_duration is None or max_duration <= 0:
		return trial_data
	
	time = trial_data['time']
	max_time = time[0] + max_duration
	
	# Find the index where time exceeds max_duration
	trim_indices = time <= max_time
	
	if not np.any(trim_indices):
		logging.warning(f"Max duration {max_duration}s is shorter than trial start time")
		return trial_data
	
	# Find the last valid index
	last_valid_idx = np.where(trim_indices)[0][-1] + 1
	
	if last_valid_idx < len(time):
		original_duration = time[-1] - time[0]
		new_duration = time[last_valid_idx-1] - time[0]
		logging.info(f"Trimming trial {os.path.basename(trial_data['file'])}: "
					f"{original_duration:.2f}s -> {new_duration:.2f}s")
	
	return {
		'file': trial_data['file'],
		'time': time[:last_valid_idx],
		'pose_x': trial_data['pose_x'][:last_valid_idx],
		'force_norm': trial_data['force_norm'][:last_valid_idx]
	}

def align_trials_by_time(trials, interpolation_points):
	"""Align trials by time using interpolation."""
	if not trials:
		return [], None
	
	# Find common time range
	max_duration = max(trial['time'].max() for trial in trials)
	common_time = np.linspace(0, max_duration, interpolation_points)
	
	aligned_trials = []
	for trial in trials:
		# Interpolate to common time grid
		pose_interp = np.interp(common_time, trial['time'], trial['pose_x'])
		force_interp = np.interp(common_time, trial['time'], trial['force_norm'])
		
		aligned_trials.append({
			'file': trial['file'],
			'time': common_time,
			'pose_x': pose_interp,
			'force_norm': force_interp
		})
	
	return aligned_trials, common_time

def align_trials_by_distance(trials, interpolation_points):
	"""Align trials by distance (pose position) using interpolation."""
	if not trials:
		return [], None
	
	# Find common pose range (intersection of all trials)
	min_pose = max(trial['pose_x'].min() for trial in trials)
	max_pose = min(trial['pose_x'].max() for trial in trials)
	
	if min_pose >= max_pose:
		logging.error("No overlapping pose range found between trials for distance alignment")
		return [], None
	
	common_distance = np.linspace(min_pose, max_pose, interpolation_points)
	
	aligned_trials = []
	for trial in trials:
		# Sort by pose_x for interpolation
		sort_indices = np.argsort(trial['pose_x'])
		sorted_pose = trial['pose_x'][sort_indices]
		sorted_force = trial['force_norm'][sort_indices]
		sorted_time = trial['time'][sort_indices]
		
		# Interpolate force and time based on pose position
		force_interp = np.interp(common_distance, sorted_pose, sorted_force)
		time_interp = np.interp(common_distance, sorted_pose, sorted_time)
		
		aligned_trials.append({
			'file': trial['file'],
			'distance': common_distance,
			'pose_x': common_distance,
			'force_norm': force_interp,
			'time': time_interp
		})
	
	return aligned_trials, common_distance

def calculate_trial_statistics(aligned_trials):
	"""Calculate mean and standard deviation across trials."""
	if not aligned_trials:
		return None, None, None, None
	
	# Stack data from all trials
	pose_data = np.array([trial['pose_x'] for trial in aligned_trials])
	force_data = np.array([trial['force_norm'] for trial in aligned_trials])
	
	# Calculate statistics
	mean_pose = np.mean(pose_data, axis=0)
	std_pose = np.std(pose_data, axis=0)
	mean_force = np.mean(force_data, axis=0)
	std_force = np.std(force_data, axis=0)
	
	return mean_pose, std_pose, mean_force, std_force

def plot_pose_force(files, config, plots_dir):
	"""
	Plot ee_pose_lin_x and force norm with multi-trial support, statistics, and alignment options.
	Supports both time-based and distance-based alignment of multiple trials.
	"""
	sampling_freq = config.get('data', {}).get('default_sampling_freq', 100.0)
	plot_config = config.get('plotting', {})
	pose_force_config = plot_config.get('pose_force', {})
	
	# Get configuration parameters
	alignment_method = pose_force_config.get('alignment', 'time')
	show_individual = pose_force_config.get('show_individual_trials', True)
	interpolation_points = pose_force_config.get('interpolation_points', 1000)
	show_std_bands = pose_force_config.get('std_dev_bands', True)
	individual_alpha = pose_force_config.get('individual_alpha', 0.3)
	max_duration = pose_force_config.get('max_duration', None)
	
	# Load all trials
	logging.info(f"Loading {len(files)} trial files for multi-trial analysis")
	all_trials = []
	for file in files:
		trial_data = load_and_process_trial(file, sampling_freq)
		if trial_data is not None:
			# Apply duration trimming if configured
			if max_duration is not None and max_duration > 0:
				trial_data = trim_trial_duration(trial_data, max_duration)
			all_trials.append(trial_data)
	
	if not all_trials:
		logging.error("No valid trial files found")
		return
	
	logging.info(f"Successfully loaded {len(all_trials)} trials")
	
	# Align trials based on selected method
	if alignment_method == 'distance':
		logging.info("Aligning trials by distance (pose position)")
		aligned_trials, x_axis = align_trials_by_distance(all_trials, interpolation_points)
		x_label = 'ee_pose_lin_x [m]'
		plot_title_suffix = "(aligned by distance)"
	else:  # default to time
		logging.info("Aligning trials by time")
		aligned_trials, x_axis = align_trials_by_time(all_trials, interpolation_points)
		x_label = 'Time [s]'
		plot_title_suffix = "(aligned by time)"
	
	if not aligned_trials:
		logging.error("Failed to align trials")
		return
	
	# Calculate statistics
	mean_pose, std_pose, mean_force, std_force = calculate_trial_statistics(aligned_trials)
	
	# Get plotting parameters from config
	fig_size = plot_config.get('figure_size', [12, 6])
	line_width = plot_config.get('line_width', 1.0)
	grid_alpha = plot_config.get('grid_alpha', 0.3)
	colors = plot_config.get('colors', {})
	pose_color = colors.get('pose', 'blue')
	force_color = colors.get('force', 'red')
	
	# Create the plot
	fig, ax1 = plt.subplots(figsize=fig_size)
	ax2 = ax1.twinx()
	
	# Plot individual trials (faded background)
	if show_individual and len(aligned_trials) > 1:
		for i, trial in enumerate(aligned_trials):
			trial_label = os.path.basename(trial['file']).replace('.csv', '')
			ax1.plot(x_axis, trial['pose_x'], color=pose_color, alpha=individual_alpha, 
					linewidth=line_width*0.7, linestyle='-', 
					label=f'Individual trials' if i == 0 else '')
			ax2.plot(x_axis, trial['force_norm'], color=force_color, alpha=individual_alpha, 
					linewidth=line_width*0.7, linestyle='-',
					label=f'Individual trials' if i == 0 else '')
	
	# Plot mean lines
	ax1.plot(x_axis, mean_pose, color=pose_color, linewidth=line_width*1.5, 
			label=f'Mean ee_pose_lin_x (n={len(aligned_trials)})')
	ax2.plot(x_axis, mean_force, color=force_color, linewidth=line_width*1.5, 
			label=f'Mean Force Norm (n={len(aligned_trials)})')
	
	# Plot standard deviation bands
	if show_std_bands and len(aligned_trials) > 1:
		ax1.fill_between(x_axis, mean_pose - std_pose, mean_pose + std_pose, 
						alpha=0.2, color=pose_color, label='±1 std dev (pose)')
		ax2.fill_between(x_axis, mean_force - std_force, mean_force + std_force, 
						alpha=0.2, color=force_color, label='±1 std dev (force)')
	
	# Configure axes
	ax1.set_xlabel(x_label)
	ax1.set_ylabel('ee_pose_lin_x [m]', color=pose_color)
	ax1.tick_params(axis='y', labelcolor=pose_color)
	ax1.grid(True, alpha=grid_alpha)
	
	ax2.set_ylabel('Force Norm [N]', color=force_color)
	ax2.tick_params(axis='y', labelcolor=force_color)
	
	# Title
	if len(aligned_trials) == 1:
		title = f"ee_pose_lin_x and Force Norm\n{os.path.basename(aligned_trials[0]['file'])}"
	else:
		title = f"ee_pose_lin_x and Force Norm {plot_title_suffix}\n{len(aligned_trials)} trials"
	ax1.set_title(title)
	
	# Combine legends
	lines1, labels1 = ax1.get_legend_handles_labels()
	lines2, labels2 = ax2.get_legend_handles_labels()
	ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right', fontsize='small')
	
	fig.tight_layout()
	
	# Handle plot display and saving
	file_config = config.get('files', {})
	try:
		plt.show()
	except Exception as e:
		logging.error(f"Failed to show multi-trial plot: {e}")

	# Save plot if enabled
	if file_config.get('auto_save_plots', True):
		if not os.path.exists(plots_dir):
			os.makedirs(plots_dir)
		
		if file_config.get('timestamp_plots', True):
			run_folder = os.path.join(plots_dir, datetime.now().strftime('%Y-%m-%d_%H-%M'))
			if not os.path.exists(run_folder):
				os.makedirs(run_folder)
		else:
			run_folder = plots_dir
			
		plot_format = file_config.get('plot_format', 'png')
		
		if len(aligned_trials) == 1:
			# Single trial naming
			plot_filename = os.path.join(run_folder, os.path.basename(aligned_trials[0]['file']).replace('.csv', f'_pose_force.{plot_format}'))
		else:
			# Multi-trial naming
			timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
			plot_filename = os.path.join(run_folder, f'pose_force_multitrials_{alignment_method}_{timestamp}.{plot_format}')
		
		dpi = plot_config.get('dpi', 100)
		fig.savefig(plot_filename, dpi=dpi, bbox_inches='tight')
		logging.info(f"Saved multi-trial plot to {plot_filename}")
	
	plt.close(fig)  # Free memory

def plot_trajectory(files, config, plots_dir):
	"""
	3D plot of trajectories using ee_pose_lin_x, ee_pose_lin_y, ee_pose_lin_z, colored by time.
	Supports plotting multiple trajectories (up to 5) with start markers.
	"""
	from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
	
	plot_config = config.get('plotting', {})
	traj_config = plot_config.get('trajectory', {})
	colors = plot_config.get('colors', {})
	sampling_freq = config.get('data', {}).get('default_sampling_freq', 100.0)
	
	# Check if file limiting is enabled (default to True for backward compatibility)
	limit_files = traj_config.get('limit_files', True)
	
	# Limit to maximum 5 files for readability if enabled
	if limit_files and len(files) > 5:
		files_to_plot = files[:5]
		logging.warning(f"Limiting to first 5 files for trajectory plotting. Total files: {len(files)}. Set 'limit_files: false' in config to plot all.")
	else:
		files_to_plot = files
		if len(files) > 5:
			logging.info(f"Plotting all {len(files)} trajectory files as limit_files is disabled.")
	
	# Create a single figure for all trajectories
	fig = plt.figure(figsize=plot_config.get('figure_size', [12, 8]))
	ax = fig.add_subplot(111, projection='3d')
	
	point_size = traj_config.get('point_size', 2)
	alpha = traj_config.get('alpha', 0.7)
	colormap = colors.get('trajectory', 'viridis')
	
	# Colors for start markers (distinct colors)
	start_marker_colors = ['red', 'green', 'blue', 'orange', 'purple']
	
	for idx, file in enumerate(files_to_plot):
		logging.info(f"Processing trajectory file: {file}")
		df = pd.read_csv(file)
		x = df.get('ee_pose_lin_x', None)
		y = df.get('ee_pose_lin_y', None)
		z = df.get('ee_pose_lin_z', None)
		if x is None or y is None or z is None:
			logging.warning(f"Missing columns in {file}, skipping.")
			continue
		
		# Create time axis
		if 'time' in df.columns:
			# Check if we have duplicate timestamps
			time_values = df['time'].values
			if len(np.unique(time_values)) < len(time_values):
				logging.info(f"Detected duplicate timestamps in {file}, creating sequential time axis")
				# Create time axis assuming constant sampling rate
				time = np.arange(len(df)) / sampling_freq  # Convert to seconds
			else:
				time = df['time'] - df['time'].iloc[0]  # Start from 0
				logging.debug(f"Using original timestamps from {file}")
		else:
			# Fallback: create sequential time assuming default sampling frequency
			time = np.arange(len(df)) / sampling_freq
			logging.debug(f"No time column found, creating sequential time axis")

		# Plot trajectory colored by time (no label to avoid cluttering legend)
		file_label = os.path.basename(file).replace('.csv', '')
		p = ax.scatter(x, y, z, c=time, cmap=colormap, 
					  s=point_size, alpha=alpha)
		
		# Add start marker (triangle) - only these will appear in legend
		marker_color = start_marker_colors[idx % len(start_marker_colors)]
		ax.scatter(x.iloc[0], y.iloc[0], z.iloc[0], 
				  c=marker_color, marker='^', s=50, alpha=1.0, 
				  edgecolors='black', linewidth=1,
				  label=f'{file_label}')
	
	ax.set_xlabel('ee_pose_lin_x [m]')
	ax.set_ylabel('ee_pose_lin_y [m]')
	ax.set_zlabel('ee_pose_lin_z [m]')
	
	# Add colorbar for time - position it to avoid overlap with legend
	if len(files_to_plot) > 0:
		cbar = fig.colorbar(p, ax=ax, label='Time [s]', shrink=0.6, pad=0.1)
	
	# Add legend - position it below the plot to avoid overlap with colorbar
	ax.legend(bbox_to_anchor=(0.5, -0.05), loc='upper center', ncol=3)
	
	if len(files_to_plot) == 1:
		ax.set_title(f'3D Trajectory (colored by time)\n{os.path.basename(files_to_plot[0])}')
	else:
		ax.set_title(f'3D Trajectories (colored by time)\n{len(files_to_plot)} trajectories')

	# Handle plot display and saving
	file_config = config.get('files', {})
	try:
		plt.show()
	except Exception as e:
		logging.error(f"Failed to show 3D trajectory plot: {e}")

	# Save plot if enabled
	if file_config.get('auto_save_plots', True):
		if not os.path.exists(plots_dir):
			os.makedirs(plots_dir)

		if file_config.get('timestamp_plots', True):
			run_folder = os.path.join(plots_dir, datetime.now().strftime('%Y-%m-%d_%H-%M'))
			if not os.path.exists(run_folder):
				os.makedirs(run_folder)
		else:
			run_folder = plots_dir

		plot_format = file_config.get('plot_format', 'png')
		
		if len(files_to_plot) == 1:
			# Single file naming
			plot_filename = os.path.join(run_folder, os.path.basename(files_to_plot[0]).replace('.csv', f'_trajectory3d.{plot_format}'))
		else:
			# Multiple files naming
			timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
			plot_filename = os.path.join(run_folder, f'trajectories_3d_{timestamp}.{plot_format}')

		dpi = plot_config.get('dpi', 100)
		fig.savefig(plot_filename, dpi=dpi, bbox_inches='tight')
		logging.info(f"Saved 3D trajectory plot to {plot_filename}")

	plt.close(fig)  # Free memory

def analyze_results(files, config):
	"""
	Compute success rate and average time from YAML result files.
	"""
	analysis_config = config.get('analysis', {})
	success_keywords = analysis_config.get('success_keywords', ['success', 'completed', 'finished'])
	time_keywords = analysis_config.get('time_keywords', ['required_time', 'duration', 'time'])
	
	n_success = 0
	n_total = 0
	times = []
	
	for file in files:
		logging.debug(f"Analyzing result file: {file}")
		try:
			with open(file, 'r') as f:
				data = yaml.safe_load(f)
		except Exception as e:
			logging.error(f"Failed to load YAML file {file}: {e}")
			continue
			
		# Check for outcome/success
		outcome = None
		for key in ['outcome', 'result', 'status', 'result_case']:
			if key in data:
				outcome = str(data[key]).lower()
				break
				
		if outcome is not None:
			n_total += 1
			if any(keyword in outcome for keyword in success_keywords):
				n_success += 1
				logging.debug(f"File {file}: SUCCESS")
			else:
				logging.debug(f"File {file}: FAILURE ({outcome})")
		
		# Check for time
		time_value = None
		for key in time_keywords:
			if key in data:
				time_value = data[key]
				break
				
		if time_value is not None:
			try:
				times.append(float(time_value))
				logging.debug(f"File {file}: time = {time_value}")
			except (ValueError, TypeError):
				logging.warning(f"Invalid time value in {file}: {time_value}")
	
	success_rate = (n_success / n_total * 100) if n_total > 0 else 0
	avg_time = np.mean(times) if times else float('nan')
	
	print("=== ANALYSIS RESULTS ===")
	print(f"Success rate: {success_rate:.1f}% ({n_success}/{n_total})")
	print(f"Average time: {avg_time:.2f} s")
	if times:
		print(f"Min time: {np.min(times):.2f} s")
		print(f"Max time: {np.max(times):.2f} s")
		print(f"Std time: {np.std(times):.2f} s")

def main():
	parser = argparse.ArgumentParser(description='Plotter and analysis for robotic insertion logs')
	parser.add_argument('mode', choices=['plot_pose_force', 'plot_trajectory', 'analyze_results'], help='Which function to run')
	parser.add_argument('--data_dir', type=str, default=None, help='Directory with CSV data files (overrides config)')
	parser.add_argument('--results_dir', type=str, default=None, help='Directory with YAML result files (overrides config)')
	parser.add_argument('--plots_dir', type=str, default=None, help='Directory where plots will be saved (overrides config)')
	parser.add_argument('--files', nargs='*', default=None, help='Specific files to use (overrides data_dir/results_dir)')
	parser.add_argument('--config', type=str, default=DEFAULT_CONFIG_FILE, help='Configuration YAML file')
	parser.add_argument('-v', '--verbose', action='count', default=0, help='Increase verbosity level (use -v, -vv, or -vvv)')
	args = parser.parse_args()
	
	# Setup logging based on verbosity
	logger = setup_logging(args.verbose)
	
	# Load configuration
	config = load_config(args.config)
	
	# Get directories from config or use command line overrides
	config_data_dir, config_results_dir, config_plots_dir = get_directories_from_config(config)
	
	data_dir = args.data_dir if args.data_dir else config_data_dir
	results_dir = args.results_dir if args.results_dir else config_results_dir
	plots_dir = args.plots_dir if args.plots_dir else config_plots_dir
	
	logging.debug(f"Using directories: data={data_dir}, results={results_dir}, plots={plots_dir}")
	
	if args.mode == 'plot_pose_force':
		files = args.files if args.files else list_csv_files(data_dir)
		if not files:
			logging.error(f"No CSV files found in {data_dir}")
			return
		plot_pose_force(files, config, plots_dir)
	elif args.mode == 'plot_trajectory':
		files = args.files if args.files else list_csv_files(data_dir)
		if not files:
			logging.error(f"No CSV files found in {data_dir}")
			return
		plot_trajectory(files, config, plots_dir)
	elif args.mode == 'analyze_results':
		files = args.files if args.files else list_yaml_files(results_dir)
		if not files:
			logging.error(f"No YAML files found in {results_dir}")
			return
		analyze_results(files, config)

if __name__ == '__main__':
	main()
