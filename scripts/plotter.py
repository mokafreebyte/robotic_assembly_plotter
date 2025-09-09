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
from datetime import datetime

# =====================
# Default Variables
# =====================
DEFAULT_DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../data'))
DEFAULT_RESULTS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../results'))
DEFAULT_PLOTS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../plots'))
DEFAULT_SAMPLING_FREQ = 100.0  # Hz, adjust based on your data collection rate

def list_csv_files(data_dir):
	return sorted(glob.glob(os.path.join(data_dir, '*.csv')))

def list_yaml_files(results_dir):
	return sorted(glob.glob(os.path.join(results_dir, '*.yaml')))

def plot_pose_force(files):
	"""
	Plot ee_pose_lin_x over time and norm of measured forces on a secondary y-axis.
	"""
	for file in files:
		df = pd.read_csv(file)
		
		# Create proper time axis - handle duplicate timestamps
		if 'time' in df.columns:
			# Check if we have duplicate timestamps
			time_values = df['time'].values
			if len(np.unique(time_values)) < len(time_values):
				print(f"[INFO] Detected duplicate timestamps in {file}, creating sequential time axis")
				# Create time axis assuming constant sampling rate
				time = np.arange(len(df)) / DEFAULT_SAMPLING_FREQ  # Convert to seconds
			else:
				time = df['time'] - df['time'].iloc[0]  # Start from 0
		else:
			# Fallback: create sequential time assuming default sampling frequency
			time = np.arange(len(df)) / DEFAULT_SAMPLING_FREQ
			
		pose_x = df.get('ee_pose_lin_x', None)
		fx = df.get('fts_wrench_lin_x', None)
		fy = df.get('fts_wrench_lin_y', None)
		fz = df.get('fts_wrench_lin_z', None)
		if pose_x is None or fx is None or fy is None or fz is None:
			print(f"[WARN] Missing columns in {file}, skipping.")
			continue
		force_norm = np.sqrt(fx**2 + fy**2 + fz**2)
		fig, ax1 = plt.subplots(figsize=(12, 6))
		ax1.set_title(f"ee_pose_lin_x and Force Norm\n{os.path.basename(file)}")
		ax1.plot(time, pose_x, 'b-', label='ee_pose_lin_x', linewidth=1)
		ax1.set_xlabel('Time [s]')
		ax1.set_ylabel('ee_pose_lin_x [m]', color='b')
		ax1.tick_params(axis='y', labelcolor='b')
		ax1.grid(True, alpha=0.3)
		
		ax2 = ax1.twinx()
		ax2.plot(time, force_norm, 'r-', label='Force Norm', linewidth=1)
		ax2.set_ylabel('Force Norm [N]', color='r')
		ax2.tick_params(axis='y', labelcolor='r')
		
		# Add legends
		lines1, labels1 = ax1.get_legend_handles_labels()
		lines2, labels2 = ax2.get_legend_handles_labels()
		ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
		
		fig.tight_layout()
		try:
			plt.show()
		except Exception as e:
			print(f"[ERROR] Failed to show plot for {file}: {e}. Saving to {DEFAULT_PLOTS_DIR}.")

		# Save da plot
		if not os.path.exists(DEFAULT_PLOTS_DIR):
			os.makedirs(DEFAULT_PLOTS_DIR)
		run_folder = os.path.join(DEFAULT_PLOTS_DIR, datetime.now().strftime('%Y-%m-%d_%H-%M'))
		if not os.path.exists(run_folder):
			os.makedirs(run_folder)
		plot_filename = os.path.join(run_folder, os.path.basename(file).replace('.csv', '_pose_force.png'))
		fig.savefig(plot_filename)

def plot_trajectory(files):
	"""
	3D plot of trajectories using ee_pose_lin_x, ee_pose_lin_y, ee_pose_lin_z, colored by position norm.
	"""
	from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
	fig = plt.figure()
	ax = fig.add_subplot(111, projection='3d')
	for file in files:
		df = pd.read_csv(file)
		x = df.get('ee_pose_lin_x', None)
		y = df.get('ee_pose_lin_y', None)
		z = df.get('ee_pose_lin_z', None)
		if x is None or y is None or z is None:
			print(f"[WARN] Missing columns in {file}, skipping.")
			continue
		pos_norm = np.sqrt(x**2 + y**2 + z**2)
		p = ax.scatter(x, y, z, c=pos_norm, cmap='viridis', label=os.path.basename(file), s=2)
	ax.set_xlabel('ee_pose_lin_x')
	ax.set_ylabel('ee_pose_lin_y')
	ax.set_zlabel('ee_pose_lin_z')
	fig.colorbar(p, ax=ax, label='Position Norm')
	ax.set_title('3D Trajectories (colored by position norm)')
	plt.show()

def analyze_results(files):
	"""
	Compute success rate and average time from YAML result files.
	"""
	n_success = 0
	n_total = 0
	times = []
	for file in files:
		with open(file, 'r') as f:
			data = yaml.safe_load(f)
		outcome = data.get('outcome', None)
		time = data.get('required_time', None)
		if outcome is not None:
			n_total += 1
			if outcome == 'success':
				n_success += 1
		if time is not None:
			times.append(time)
	success_rate = (n_success / n_total * 100) if n_total > 0 else 0
	avg_time = np.mean(times) if times else float('nan')
	print(f"Success rate: {success_rate:.1f}% ({n_success}/{n_total})")
	print(f"Average time: {avg_time:.2f} s")

def main():
	parser = argparse.ArgumentParser(description='Plotter and analysis for robotic insertion logs')
	parser.add_argument('mode', choices=['plot_pose_force', 'plot_trajectory', 'analyze_results'], help='Which function to run')
	parser.add_argument('--data_dir', type=str, default=DEFAULT_DATA_DIR, help='Directory with CSV data files')
	parser.add_argument('--results_dir', type=str, default=DEFAULT_RESULTS_DIR, help='Directory with YAML result files')
	parser.add_argument('--files', nargs='*', default=None, help='Specific files to use (overrides data_dir/results_dir)')
	parser.add_argument('--sampling_freq', type=float, default=DEFAULT_SAMPLING_FREQ, help='Sampling frequency in Hz (for time axis when timestamps are duplicate)')
	args = parser.parse_args()
	
	# Update global sampling frequency
	global DEFAULT_SAMPLING_FREQ
	DEFAULT_SAMPLING_FREQ = args.sampling_freq

	if args.mode == 'plot_pose_force':
		files = args.files if args.files else list_csv_files(args.data_dir)
		if not files:
			print(f"No CSV files found in {args.data_dir}")
			return
		plot_pose_force(files)
	elif args.mode == 'plot_trajectory':
		files = args.files if args.files else list_csv_files(args.data_dir)
		if not files:
			print(f"No CSV files found in {args.data_dir}")
			return
		plot_trajectory(files)
	elif args.mode == 'analyze_results':
		files = args.files if args.files else list_yaml_files(args.results_dir)
		if not files:
			print(f"No YAML files found in {args.results_dir}")
			return
		analyze_results(files)

if __name__ == '__main__':
	main()
