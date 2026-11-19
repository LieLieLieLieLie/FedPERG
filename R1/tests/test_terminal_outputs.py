"""Terminal averaging preserves training and stores a reproducible output model."""
import torch

from fedperg.config import ExperimentConfig
from fedperg.trainer import FederatedExperiment


def test_terminal_average_matches_saved_iterates_and_preserves_raw_model(tmp_path):
    torch.set_num_threads(1)

    def run(rounds, window, name):
        config = ExperimentConfig(
            dataset='synthetic_complementarity', regime='label_skew', method='FedAvg',
            seed=191, clients=3, clients_per_round=2, hidden_dim=8,
            rounds=rounds, terminal_average_rounds=window, max_local_batches=1,
            device='cpu', results_dir=str(tmp_path / name),
        )
        experiment = FederatedExperiment(config)
        result = experiment.run(False)
        checkpoint = next((tmp_path / name / 'models').glob('*.pth'))
        return result, torch.load(checkpoint, map_location='cpu')

    _, before = run(3, 0, 'before')
    raw_result, raw = run(4, 0, 'raw')
    averaged_result, averaged = run(4, 2, 'averaged')
    assert averaged_result['terminal_average']['rounds'] == 2
    assert raw_result['final']['accuracy'] == averaged_result['final']['accuracy']
    for name, tensor in raw['model_state'].items():
        torch.testing.assert_close(averaged['model_state'][name], tensor, rtol=0, atol=0)
        expected = (before['model_state'][name] + tensor) / 2
        torch.testing.assert_close(averaged['terminal_average_state'][name], expected)
