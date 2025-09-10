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

def plot_pose_force(files, config, plots_dir):
	"""
	Plot ee_pose_lin_x over time and norm of measured forces on a secondary y-axis.
	"""
	sampling_freq = config.get('data', {}).get('default_sampling_freq', 100.0)
	plot_config = config.get('plotting', {})
	
	for file in files:
		logging.info(f"Processing file: {file}")
		df = pd.read_csv(file)
		
		# Create proper time axis - handle duplicate timestamps
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
			
		pose_x = df.get('ee_pose_lin_x', None)
		fx = df.get('fts_wrench_lin_x', None)
		fy = df.get('fts_wrench_lin_y', None)
		fz = df.get('fts_wrench_lin_z', None)
		if pose_x is None or fx is None or fy is None or fz is None:
			logging.warning(f"Missing columns in {file}, skipping.")
			continue
		force_norm = np.sqrt(fx**2 + fy**2 + fz**2)
		
		# Get plotting parameters from config
		fig_size = plot_config.get('figure_size', [12, 6])
		line_width = plot_config.get('line_width', 1.0)
		grid_alpha = plot_config.get('grid_alpha', 0.3)
		colors = plot_config.get('colors', {})
		pose_color = colors.get('pose', 'blue')
		force_color = colors.get('force', 'red')
		
		fig, ax1 = plt.subplots(figsize=fig_size)
		ax1.set_title(f"ee_pose_lin_x and Force Norm\n{os.path.basename(file)}")
		ax1.plot(time, pose_x, color=pose_color, label='ee_pose_lin_x', linewidth=line_width)
		ax1.set_xlabel('Time [s]')
		ax1.set_ylabel('ee_pose_lin_x [m]', color=pose_color)
		ax1.tick_params(axis='y', labelcolor=pose_color)
		ax1.grid(True, alpha=grid_alpha)
		
		ax2 = ax1.twinx()
		ax2.plot(time, force_norm, color=force_color, label='Force Norm', linewidth=line_width)
		ax2.set_ylabel('Force Norm [N]', color=force_color)
		ax2.tick_params(axis='y', labelcolor=force_color)
		
		# Add legends
		lines1, labels1 = ax1.get_legend_handles_labels()
		lines2, labels2 = ax2.get_legend_handles_labels()
		ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
		
		fig.tight_layout()
		
		# Handle plot display and saving
		file_config = config.get('files', {})
		try:
			plt.show()
		except Exception as e:
			logging.error(f"Failed to show plot for {file}: {e}")

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
			plot_filename = os.path.join(run_folder, os.path.basename(file).replace('.csv', f'_pose_force.{plot_format}'))
			
			dpi = plot_config.get('dpi', 100)
			fig.savefig(plot_filename, dpi=dpi, bbox_inches='tight')
			logging.info(f"Saved plot to {plot_filename}")
		
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
	
	# Limit to maximum 5 files for readability
	files_to_plot = files[:5] if len(files) > 5 else files
	if len(files) > 5:
		logging.warning(f"Limiting to first 5 files for trajectory plotting. Total files: {len(files)}")
	
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

		# Plot trajectory colored by time
		file_label = os.path.basename(file).replace('.csv', '')
		p = ax.scatter(x, y, z, c=time, cmap=colormap, label=file_label, 
					  s=point_size, alpha=alpha)
		
		# Add start marker (triangle)
		marker_color = start_marker_colors[idx % len(start_marker_colors)]
		ax.scatter(x.iloc[0], y.iloc[0], z.iloc[0], 
				  c=marker_color, marker='^', s=50, alpha=1.0, 
				  edgecolors='black', linewidth=1,
				  label=f'{file_label} start')
	
	ax.set_xlabel('ee_pose_lin_x [m]')
	ax.set_ylabel('ee_pose_lin_y [m]')
	ax.set_zlabel('ee_pose_lin_z [m]')
	
	# Add colorbar for time
	if len(files_to_plot) > 0:
		cbar = fig.colorbar(p, ax=ax, label='Time [s]', shrink=0.8)
	
	# Add legend and title
	ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
	
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
