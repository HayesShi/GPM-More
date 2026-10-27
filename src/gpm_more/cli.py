"""Command line entry points for data preparation and training."""
from __future__ import annotations
import argparse
import json


def main(argv=None):
    parser = argparse.ArgumentParser(prog='gpm-more', description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('prepare', help='Create train/validation data and an observation cache')
    p.add_argument('--graph', required=True, help='Preprocessed graph NPZ; see docs/USAGE.md')
    p.add_argument('--output', required=True)
    p.add_argument('--task', choices=['regression','binary_classification','multiclass_classification'], default='regression')
    p.add_argument('--name', default='custom')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--bank', help='Reuse an existing context/extra .npy cache instead of resampling')
    p.add_argument('--chunk-size', type=int, default=512)
    p.add_argument('--standardize-targets', action='store_true')
    p = sub.add_parser('train', help='Run a fresh or resumed supervised experiment')
    p.add_argument('--data', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--config', required=True)
    p.add_argument('--device', default='cpu')
    p.add_argument('--resume', help='Complete checkpoint produced by this trainer')
    p.add_argument('--init-weights', help='Optional model weights; historical active weights supported')
    args = parser.parse_args(argv)
    if args.command == 'prepare':
        from .data import prepare
        result = prepare(args.graph, args.output, task=args.task, name=args.name, seed=args.seed,
                         bank=args.bank, chunk_size=args.chunk_size,
                         standardize_targets=args.standardize_targets)
    else:
        from .train import train
        result = train(args.data, args.output, args.config, device=args.device,
                       resume=args.resume, init_weights=args.init_weights)
    print(json.dumps(result, indent=2, allow_nan=False), flush=True)


if __name__ == '__main__':
    main()
