"""Unchanged BladeGL output naming; unrelated matrix routes omitted."""
from pathlib import Path
EXTENSIONS = {'metrics': 'json', 'calibration': 'json', 'image_scores': 'csv', 'curves': 'npz', 'cost_rgb': 'json'}

def readout_artifact(folder: Path, kind: str,
                     mode: str = 'combined', weight: float = .5) -> Path:
    """A BladeGL file for one explicitly named image readout."""
    return folder/f'{kind}_{mode}_{weight}.{EXTENSIONS[kind]}'
