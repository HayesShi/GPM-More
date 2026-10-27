"""Exercise the public prepare/train commands, including fresh walk sampling."""
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np


def test_cli_prepares_and_trains_without_external_assets(tmp_path):
    rng = np.random.default_rng(42)
    x = rng.normal(size=(8, 3)).astype(np.float32)
    graph = tmp_path / 'graph.npz'
    np.savez_compressed(graph, x=x,
                        edge_index=np.array([[0, 1, 2, 2, 3, 4, 5],
                                             [1, 2, 0, 3, 4, 5, 6]]),
                        train_ids=np.arange(5), val_ids=np.arange(5, 8),
                        train_y=x[:5, 0], val_y=x[5:, 0])
    config = {'steps': 2, 'warmup': 0, 'batch_size': 4,
              'micro_batch_size': 2, 'train_budgets': [4, 16],
              'eval_budget': 4, 'eval_every': 1,
              'feedback': {'mode': 'late_stop', 'stop_after': 1},
              'model': {'hidden_dim': 16}}
    config_path = tmp_path / 'config.json'
    config_path.write_text(json.dumps(config), encoding='utf8')
    prepared, output = tmp_path / 'prepared', tmp_path / 'run'
    environment = {**os.environ, 'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1'}
    repository = Path(__file__).resolve().parents[1]
    for arguments in [
        ['prepare', '--graph', str(graph), '--output', str(prepared), '--seed', '0'],
        ['train', '--data', str(prepared), '--config', str(config_path),
         '--output', str(output), '--device', 'cpu'],
    ]:
        subprocess.run([sys.executable, '-m', 'gpm_more.cli', *arguments],
                       cwd=repository, env=environment, check=True,
                       capture_output=True, text=True, timeout=120)
    assert np.load(prepared / 'context_paths.npy').shape == (8, 16, 9)
    metrics = json.loads((output / 'metrics.json').read_text(encoding='utf8'))
    assert metrics['complete'] and metrics['completed_update'] == 2
    assert metrics['test_labels_read'] is False
    assert metrics['execution']['deterministic_algorithms'] is True
    assert set(metrics['last_validation']) == {'R2', 'MSE'}
    assert (output / 'latest.pt').exists() and (output / 'best.pt').exists()
