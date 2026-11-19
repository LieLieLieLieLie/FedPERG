"""Locations shared by the GPU experiment runners and reporting commands."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / 'results'
MODELS = RESULTS / 'models'
PROTOCOLS = MODELS / 'protocols'
LOGS = MODELS / 'logs'
DEVELOPMENT = MODELS / 'development'
FORMAL = MODELS / 'formal'


def prepare_output_directories():
    for path in [PROTOCOLS, LOGS, DEVELOPMENT, FORMAL, RESULTS / 'figures', RESULTS / 'tables']:
        path.mkdir(parents=True, exist_ok=True)
