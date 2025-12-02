# Widefield Analysis Usage Guide

This guide provides step-by-step instructions for using the `WidefieldAnalysis` class to analyze widefield imaging data from body core temperature experiments.

## Overview

The `WidefieldAnalysis` class provides a comprehensive interface for:
- Loading imaging data (TIFF stacks)
- Processing and analyzing fluorescence signals
- Calculating delta F over F (ΔF/F)
- Extracting region of interest (ROI) traces
- Creating visualization movies
- Saving processed data

## Trial Folder Structure

A trial folder must contain:
- `recording.tiff`: Multi-page TIFF stack with imaging data
- `recording.csv`: Timestamp metadata for frames (optional)
- `data.h5`: Sensor data (temperature, camera triggers)
- `green.tiff`: Reference/green channel image (optional)

## Quick Start

### 1. Basic Setup

```python
from pathlib import Path
from analysis import WidefieldAnalysis

trial_path = Path("data/raw/session_name/trials/trial_name")
wf = WidefieldAnalysis(trial_path)
wf.load_data()
```

### 2. Get Recording Information

```python
fps = wf.get_fps()
duration = wf.get_recording_duration()
print(f"Frame rate: {fps} Hz, Duration: {duration:.2f} s")
```

## Step-by-Step Workflows

### Workflow 1: View Reference Image

To inspect the green reference image:

```python
wf = WidefieldAnalysis(trial_path)
wf.load_data()
wf.view_reference(cmap="gray")
```

### Workflow 2: Create and Apply Mask

**Step 1: Create mask interactively**
- Click to set center
- Press 'B' to increase radius
- Press 'S' to decrease radius
- Press 'Enter' to confirm

```python
mask_data = wf.create_mask(initial_radius=100.0)
wf.save_mask(mask_data, name="mask")
```

**Step 2: Apply mask to data**
```python
masked_data = wf.apply_mask()
wf.save_array(masked_data, "masked_data")
```

### Workflow 3: Calculate ΔF/F

**Step 1: Detect stimulus onset**
```python
from helpers import detect_stimulus_frames

temperature_key = 'data/in/temperature'
temp_trace = wf.sensor_data[temperature_key]
temp_attributes = wf.sensor_attrs.get(temperature_key, {})
sampling_rate = temp_attributes.get('sr', 1000)
fps = wf.get_fps()

stimulus_result = detect_stimulus_frames(
    temp_trace, sampling_rate, camera_fps=fps, plot=False
)
onset_frame = stimulus_result.get('onset_frame')
```

**Step 2: Calculate baseline**
```python
baseline_map = wf.calculate_baseline(
    stimulus_start_frame=onset_frame,
    baseline_ms=500.0,
    fps=fps
)
```

**Step 3: Calculate ΔF/F**
```python
dff_data = wf.calculate_deltaff(baseline=baseline_map)
wf.save_array(dff_data, "dff_data")
```

### Workflow 4: ROI Analysis

**Step 1: Find ROI centroid**
```python
mean_dff = np.mean(dff_data, axis=0)
roi = wf.calculate_percentile_centroid_roi(mean_dff, percentile=95.0)
wf.set_roi(roi)
```

**Step 2: Extract trace from ROI**
```python
roi_diameter = 150.0
wf_trace = wf.calculate_trace_within_roi(
    data=dff_data,
    diameter=roi_diameter
)
```

**Step 3: Visualize trace**
```python
import matplotlib.pyplot as plt
import numpy as np

time_axis = np.arange(len(wf_trace)) / fps
plt.plot(time_axis, wf_trace)
plt.xlabel('Time (s)')
plt.ylabel('ΔF/F')
plt.show()
```

### Workflow 5: Create Movies

**Raw data movie:**
```python
movie_path = wf.create_movie(
    data=None,
    fps=fps,
    cmap="gray"
)
```

**ΔF/F movie:**
```python
dff_movie_path = wf.create_movie(
    data=dff_data,
    fps=fps,
    cmap='RdBu_r',
    vmin=-0.5,
    vmax=0.5
)
```

### Workflow 6: Downscale Data

To reduce memory usage and speed up processing:

```python
downscaled_data = wf.downscale(factor=2)
wf.save_array(downscaled_data, "downscaled_data")
```

Or specify target resolution:

```python
downscaled_data = wf.downscale(target_resolution=(512, 512))
```

## Complete Example

See `example_usage.py` for complete, runnable examples of all workflows.

## Key Methods Reference

### Data Loading
- `load_data()`: Load all trial data (imaging, timestamps, sensors)
- `get_fps()`: Get frame rate from file attributes
- `get_recording_duration()`: Calculate recording duration

### Data Processing
- `downscale()`: Reduce image resolution
- `apply_mask()`: Apply circular mask to data
- `calculate_baseline()`: Calculate baseline from pre-stimulus period
- `calculate_deltaff()`: Calculate ΔF/F
- `calculate_percentile()`: Calculate percentile map

### ROI Analysis
- `create_mask()`: Interactive mask creation tool
- `save_mask()`: Save mask to file
- `load_mask()`: Load saved mask
- `set_roi()`: Set ROI center coordinates
- `calculate_percentile_centroid_roi()`: Find ROI from percentile
- `calculate_trace_within_roi()`: Extract mean trace from ROI

### Visualization
- `view_reference()`: Display green reference image
- `create_movie()`: Create MP4 movie from data

### Data I/O
- `save_array()`: Save numpy array to processed folder
- `to_numpy()`: Convert TIFF or return numpy array

## Environment Setup

Before running analysis code, activate the environment:

```bash
micromamba activate imaging-probe
```

## File Organization

Processed data is automatically saved to:
- Session-level: `data/processed/[session]/`
- Trial-level: `data/processed/[session]/trials/[trial]/`

Masks are saved at the session level (shared across trials).
Processed arrays are saved at the trial level.

## Troubleshooting

**Issue: File not found**
- Ensure trial path is correct
- Check that `recording.tiff` exists in trial folder

**Issue: FPS not available**
- Check `data.h5` file attributes for `camera_fps`
- Manually provide `fps` parameter to methods

**Issue: Mask coordinates out of bounds**
- Mask coordinates are automatically scaled if data dimensions differ from reference image
- Check that mask was created using the same reference image

**Issue: Memory errors**
- Use `downscale()` to reduce data size
- Process data in chunks if needed

## See Also

- `example_usage.py`: Complete runnable examples
- `analysis.py`: Full class documentation
- `batch/process_trials.py`: Batch processing example

