"""
Step-by-step example of using WidefieldAnalysis class.

This script demonstrates the complete workflow for analyzing widefield
imaging data from body core temperature experiments.

A trial folder should contain:
- recording.tiff: Multi-page TIFF stack with imaging data
- recording.csv: Timestamp metadata for frames
- data.h5: Sensor data (temperature, camera triggers)
- green.tiff: Reference/green channel image (optional)
"""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from analysis import WidefieldAnalysis
from helpers import get_condition_from_attributes, detect_stimulus_frames


def example_basic_usage():
    """
    Example 1: Basic data loading and inspection.
    
    This example shows how to:
    1. Initialize the analyzer
    2. Load all trial data
    3. View basic information about the recording
    """
    trial_path = Path("data/raw/session_name/trials/trial_name")
    
    wf = WidefieldAnalysis(trial_path)
    wf.load_data()
    
    fps = wf.get_fps()
    duration = wf.get_recording_duration()
    
    print(f"Frame rate: {fps} Hz")
    print(f"Duration: {duration:.2f} seconds")


def example_view_reference():
    """
    Example 2: Viewing the green reference image.
    
    Displays the green reference image for visual inspection.
    Useful for identifying brain regions and setting up masks.
    """
    trial_path = Path("data/raw/session_name/trials/trial_name")
    
    wf = WidefieldAnalysis(trial_path)
    wf.load_data()
    
    wf.view_reference(cmap="gray")


def example_create_and_save_mask():
    """
    Example 3: Creating and saving a circular mask.
    
    Interactive tool to create a circular mask:
    - Click to set center
    - Press 'B' to increase radius
    - Press 'S' to decrease radius
    - Press 'Enter' to confirm
    
    The mask is saved to the session-level processed folder.
    """
    trial_path = Path("data/raw/session_name/trials/trial_name")
    
    wf = WidefieldAnalysis(trial_path)
    wf.load_data()
    
    mask_data = wf.create_mask(initial_radius=100.0)
    
    if mask_data is not None:
        saved_path = wf.save_mask(mask_data, name="mask")
        print(f"Mask saved to: {saved_path}")


def example_apply_mask():
    """
    Example 4: Applying a saved mask to imaging data.
    
    Loads a previously saved mask and applies it to the imaging data,
    setting pixels outside the mask to zero.
    """
    trial_path = Path("data/raw/session_name/trials/trial_name")
    
    wf = WidefieldAnalysis(trial_path)
    wf.load_data()
    
    masked_data = wf.apply_mask()
    
    if masked_data is not None:
        print(f"Masked data shape: {masked_data.shape}")
        saved_path = wf.save_array(masked_data, "masked_data")
        print(f"Saved masked data to: {saved_path}")


def example_calculate_dff():
    """
    Example 5: Calculating delta F over F (ΔF/F).
    
    This example shows the complete workflow:
    1. Detect stimulus onset from temperature trace
    2. Calculate baseline from pre-stimulus period
    3. Calculate ΔF/F using the baseline
    """
    trial_path = Path("data/raw/session_name/trials/trial_name")
    
    wf = WidefieldAnalysis(trial_path)
    wf.load_data()
    
    temperature_key = 'data/in/temperature'
    temp_trace = wf.sensor_data[temperature_key]
    temp_attributes = wf.sensor_attrs.get(temperature_key, {})
    sampling_rate = temp_attributes.get('sr', 1000)
    fps = wf.get_fps()
    
    stimulus_result = detect_stimulus_frames(
        temp_trace, sampling_rate, camera_fps=fps, plot=False
    )
    
    onset_frame = stimulus_result.get('onset_frame')
    
    baseline_map = wf.calculate_baseline(
        stimulus_start_frame=onset_frame,
        baseline_ms=500.0,
        fps=fps
    )
    
    dff_data = wf.calculate_deltaff(baseline=baseline_map)
    
    if dff_data is not None:
        print(f"ΔF/F data shape: {dff_data.shape}")
        saved_path = wf.save_array(dff_data, "dff_data")
        print(f"Saved ΔF/F data to: {saved_path}")


def example_roi_analysis():
    """
    Example 6: ROI-based trace extraction.
    
    This example shows how to:
    1. Calculate mean ΔF/F map
    2. Find ROI centroid from percentile threshold
    3. Extract mean trace within circular ROI
    """
    trial_path = Path("data/raw/session_name/trials/trial_name")
    
    wf = WidefieldAnalysis(trial_path)
    wf.load_data()
    
    temperature_key = 'data/in/temperature'
    temp_trace = wf.sensor_data[temperature_key]
    temp_attributes = wf.sensor_attrs.get(temperature_key, {})
    sampling_rate = temp_attributes.get('sr', 1000)
    fps = wf.get_fps()
    
    stimulus_result = detect_stimulus_frames(
        temp_trace, sampling_rate, camera_fps=fps, plot=False
    )
    
    onset_frame = stimulus_result.get('onset_frame')
    
    baseline_map = wf.calculate_baseline(
        stimulus_start_frame=onset_frame,
        baseline_ms=500.0,
        fps=fps
    )
    
    dff_data = wf.calculate_deltaff(baseline=baseline_map)
    
    if dff_data is not None:
        mean_dff = np.mean(dff_data, axis=0)
        
        roi = wf.calculate_percentile_centroid_roi(mean_dff, percentile=95.0)
        wf.set_roi(roi)
        
        roi_diameter = 150.0
        wf_trace = wf.calculate_trace_within_roi(
            data=dff_data,
            diameter=roi_diameter
        )
        
        if wf_trace is not None:
            time_axis = np.arange(len(wf_trace)) / fps
            
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.plot(time_axis, wf_trace, 'k-', linewidth=1.5)
            ax.set_xlabel('Time (s)', fontsize=12)
            ax.set_ylabel('ΔF/F', fontsize=12)
            ax.set_title('Widefield Trace (ROI mean)', fontsize=12)
            ax.spines[['top', 'right']].set_visible(False)
            plt.tight_layout()
            plt.show()
            
            print(f"ROI center: ({roi[0]}, {roi[1]})")
            print(f"Trace length: {len(wf_trace)} frames")
            print(f"Trace mean: {wf_trace.mean():.4f}")


def example_create_movie():
    """
    Example 7: Creating MP4 movies from imaging data.
    
    Creates a movie from the imaging data or processed data (e.g., ΔF/F).
    Can customize colormap, value range, and frame rate.
    """
    trial_path = Path("data/raw/session_name/trials/trial_name")
    
    wf = WidefieldAnalysis(trial_path)
    wf.load_data()
    
    fps = wf.get_fps()
    
    movie_path = wf.create_movie(
        data=None,
        output_path=None,
        fps=fps,
        cmap="gray"
    )
    
    print(f"Movie saved to: {movie_path}")


def example_create_dff_movie():
    """
    Example 8: Creating a movie from ΔF/F data.
    
    Creates a movie with custom colormap (RdBu_r) suitable for
    visualizing ΔF/F changes.
    """
    trial_path = Path("data/raw/session_name/trials/trial_name")
    
    wf = WidefieldAnalysis(trial_path)
    wf.load_data()
    
    temperature_key = 'data/in/temperature'
    temp_trace = wf.sensor_data[temperature_key]
    temp_attributes = wf.sensor_attrs.get(temperature_key, {})
    sampling_rate = temp_attributes.get('sr', 1000)
    fps = wf.get_fps()
    
    stimulus_result = detect_stimulus_frames(
        temp_trace, sampling_rate, camera_fps=fps, plot=False
    )
    
    onset_frame = stimulus_result.get('onset_frame')
    
    baseline_map = wf.calculate_baseline(
        stimulus_start_frame=onset_frame,
        baseline_ms=500.0,
        fps=fps
    )
    
    dff_data = wf.calculate_deltaff(baseline=baseline_map)
    
    if dff_data is not None:
        dff_movie_path = wf.create_movie(
            data=dff_data,
            output_path=None,
            fps=fps,
            cmap='RdBu_r',
            vmin=-0.5,
            vmax=0.5
        )
        
        print(f"ΔF/F movie saved to: {dff_movie_path}")


def example_downscale_data():
    """
    Example 9: Downscaling imaging data.
    
    Reduces image resolution to save memory and speed up processing.
    Can specify target resolution or downscale factor.
    """
    trial_path = Path("data/raw/session_name/trials/trial_name")
    
    wf = WidefieldAnalysis(trial_path)
    wf.load_data()
    
    downscaled_data = wf.downscale(factor=2)
    
    if downscaled_data is not None:
        print(f"Original shape: {wf.imaging_data.shape}")
        print(f"Downscaled shape: {downscaled_data.shape}")
        
        saved_path = wf.save_array(downscaled_data, "downscaled_data")
        print(f"Saved downscaled data to: {saved_path}")


def example_complete_workflow():
    """
    Example 10: Complete analysis workflow.
    
    This example combines all steps into a complete workflow:
    1. Load data
    2. Set condition information
    3. Detect stimulus
    4. Calculate baseline and ΔF/F
    5. Find ROI and extract trace
    6. Save processed data
    """
    trial_path = Path("data/raw/session_name/trials/trial_name")
    
    wf = WidefieldAnalysis(trial_path)
    wf.load_data()
    
    condition_dict = get_condition_from_attributes(wf.file_attrs)
    wf.set_condition(condition_dict)
    
    temperature_key = 'data/in/temperature'
    temp_trace = wf.sensor_data[temperature_key]
    temp_attributes = wf.sensor_attrs.get(temperature_key, {})
    sampling_rate = temp_attributes.get('sr', 1000)
    fps = wf.get_fps()
    
    stimulus_result = detect_stimulus_frames(
        temp_trace, sampling_rate, camera_fps=fps, plot=False
    )
    
    onset_frame = stimulus_result.get('onset_frame')
    
    baseline_map = wf.calculate_baseline(
        stimulus_start_frame=onset_frame,
        baseline_ms=500.0,
        fps=fps
    )
    
    dff_data = wf.calculate_deltaff(baseline=baseline_map)
    
    if dff_data is not None:
        mean_dff = np.mean(dff_data, axis=0)
        roi = wf.calculate_percentile_centroid_roi(mean_dff, percentile=95.0)
        wf.set_roi(roi)
        
        roi_diameter = 150.0
        wf_trace = wf.calculate_trace_within_roi(
            data=dff_data,
            diameter=roi_diameter
        )
        
        if wf_trace is not None:
            wf.save_array(dff_data, "dff_data")
            wf.save_array(baseline_map, "baseline_map")
            
            print("Analysis complete!")
            print(f"ROI center: ({roi[0]}, {roi[1]})")
            print(f"Trace statistics: mean={wf_trace.mean():.4f}, "
                  f"std={wf_trace.std():.4f}")


if __name__ == "__main__":
    """
    To run a specific example, uncomment the function call below.
    
    Make sure to:
    1. Activate the correct environment: micromamba activate imaging-probe
    2. Update the trial_path in the example function
    3. Ensure the trial folder contains required files
    """
    
    example_basic_usage()
    
    # example_view_reference()
    # example_create_and_save_mask()
    # example_apply_mask()
    # example_calculate_dff()
    # example_roi_analysis()
    # example_create_movie()
    # example_create_dff_movie()
    # example_downscale_data()
    # example_complete_workflow()

