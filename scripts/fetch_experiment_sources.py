"""Fetch pinned upstream source; never install dependencies or launch training."""
import argparse
import json
from pathlib import Path
import subprocess
import urllib.request

REPO = Path(__file__).resolve().parents[1]
GPM_REV = '2ed3ebf7df8d8401520e6ec3bdbd1c2b291b5784'
MM_REV = '0769fa670c69a6dc0756a2d7fc697b980e494775'


def fetch_gpm():
    destination = REPO / 'external/GPM'
    if destination.exists():
        actual = subprocess.check_output(['git', '-C', str(destination), 'rev-parse', 'HEAD'], text=True).strip()
        if actual != GPM_REV:
            raise RuntimeError(f'Existing GPM checkout is {actual}; expected {GPM_REV}. No reset performed.')
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(['git', 'clone', '--no-checkout', 'https://github.com/Zehong-Wang/GPM.git', str(destination)], check=True)
    subprocess.run(['git', '-C', str(destination), 'checkout', '--detach', GPM_REV], check=True)
    return destination


def fetch_mm():
    destination = REPO / 'external/mm_graph_sports_snapshot'
    mappings = {name: 'lp/' + name for name in ['models.py', 'main.py', 'utils.py', 'lp_dataset.py',
                                               'configs/defaults.yaml', 'configs/optuna.yaml']}
    mappings['README.md'] = 'README.md'
    for local, remote in mappings.items():
        url = f'https://raw.githubusercontent.com/mm-graph-benchmark/mm-graph-benchmark/{MM_REV}/{remote}'
        body = urllib.request.urlopen(url, timeout=60).read()
        target = destination / local
        if target.exists() and target.read_bytes() != body:
            raise RuntimeError(f'Existing upstream file differs: {target}; no overwrite performed.')
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            target.write_bytes(body)
    (destination / 'source_commit.json').write_text(json.dumps({'repository': 'https://github.com/mm-graph-benchmark/mm-graph-benchmark',
                                                               'commit': MM_REV}, indent=2), encoding='utf8')
    return destination


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', choices=['gpm', 'mm-graph', 'all'])
    args = parser.parse_args()
    if args.source in ('gpm', 'all'):
        print(fetch_gpm())
    if args.source in ('mm-graph', 'all'):
        print(fetch_mm())
