# Robotic Assembly Plotter

Simple plotting tool for robotic insertion task analysis.

## Quick Start

```bash
# Install dependencies
pip install pandas matplotlib numpy pyyaml

# Run plots
python scripts/plotter.py plot_pose_force    # Force/pose over time with statistics
python scripts/plotter.py plot_trajectory    # 3D trajectory visualization  
python scripts/plotter.py analyze_results    # Success rate analysis
```

## Configuration

Edit `scripts/config.yaml` to customize:

- **Data directories**: `data_dir`, `results_dir`, `plots_dir`
- **Plot alignment**: `time` or `distance` based
- **Plot Trimming**: `max_duration` (seconds) or `max_distance` (meters)
- **Trajectory markers**: Start/end triangle markers
- **Multi-trial stats**: Show individual trials + mean ± std dev

## Output

Plots are automatically saved to the `plots/` directory with timestamps.
