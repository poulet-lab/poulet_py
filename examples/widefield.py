from pathlib import Path

from poulet_py.utils.analisys.widefield import WidefieldAnalysis


TRIAL_PATH = Path(
    r"Y:\experiments\corebodytemp_imaging_IC\data\raw"
    r"\20260504_JPCM_09695_widefieldimaging_precno\trials\260504_115040"
)


analysis = WidefieldAnalysis(path=TRIAL_PATH)
analysis.load()
trial = analysis.active_trial

print(f"Loaded trial: {trial.path.name}")
print(f"Trial folder: {trial.path}")
print(f"Imaging shape: {trial.imaging_data.shape}")
print(f"Reference image shape: {trial.reference_image.shape}")
print(f"Timestamps rows: {len(trial.timestamps)}")

print("\nMetadata:")
for key, value in trial.analog_output_data_file_attrs.items():
    print(f"  {key}: {value}")

analysis.session.close()
