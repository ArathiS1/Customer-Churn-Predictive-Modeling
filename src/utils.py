import pandas as pd
import numpy as np
import yaml
import pickle
import json
from typing import Dict, Any
import matplotlib.pyplot as plt

def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def save_model(model, filepath: str):
    """Save trained model to file."""
    with open(filepath, 'wb') as file:
        pickle.dump(model, file)

def load_model(filepath: str):
    """Load trained model from file."""
    with open(filepath, 'rb') as file:
        return pickle.load(file)

def save_results(results: Dict[str, Any], filepath: str):
    """Save results to JSON file."""
    with open(filepath, 'w') as file:
        json.dump(results, file, indent=4)

def save_plot(fig, filepath: str, dpi: int = 300):
    """Save matplotlib figure to file."""
    fig.savefig(filepath, dpi=dpi, bbox_inches='tight')
    plt.close(fig)
