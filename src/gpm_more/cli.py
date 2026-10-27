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
    p.add_argument('--split-index', type=int, default=0, help='Column for a multi-split train/validation mask')
    p = sub.add_parser('train', help='Run a fresh or resumed supervised experiment')
    p.add_argument('--data', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--config', required=True)
    p.add_argument('--device', default='cpu')
    p.add_argument('--resume', help='Complete checkpoint produced by this trainer')
    p.add_argument('--init-weights', help='Optional model weights; historical active weights supported')
    p.add_argument('--reset-head', action='store_true', help='Reset task head after loading weights')
    p.add_argument('--max-updates', type=int, help='Absolute resumable update boundary')
    p.add_argument('--stop-file', help='Stop at the next validation checkpoint if this file exists')
    p.add_argument('--max-wall-seconds', type=float, help='Stop at a validation checkpoint after this time')
    p = sub.add_parser('autotune', help='Run a resumable per-dataset validation search')
    p.add_argument('--plan', required=True, help='Autotune JSON plan; see docs/AUTOTUNE.md')
    p.add_argument('--output', required=True)
    p = sub.add_parser('dispatch', help='Wait for earlier campaigns and run a finite campaign list')
    p.add_argument('--plan', required=True, help='Dispatcher JSON plan; see docs/AUTOTUNE.md')
    p.add_argument('--output', required=True)
    args = parser.parse_args(argv)
    if args.command == 'prepare':
        from .data import prepare
        result = prepare(args.graph, args.output, task=args.task, name=args.name, seed=args.seed,
                         bank=args.bank, chunk_size=args.chunk_size,
                         standardize_targets=args.standardize_targets, split_index=args.split_index)
    elif args.command == 'train':
        from .train import train
        result = train(args.data, args.output, args.config, device=args.device,
                       resume=args.resume, init_weights=args.init_weights,
                       reset_head=args.reset_head, max_updates=args.max_updates,
                       stop_file=args.stop_file, max_wall_seconds=args.max_wall_seconds)
    elif args.command == 'autotune':
        from .autotune import run_queue
        result = run_queue(args.plan, args.output)
    else:
        from .dispatch import run_dispatch
        result = run_dispatch(args.plan, args.output)
    print(json.dumps(result, indent=2, allow_nan=False), flush=True)


if __name__ == '__main__':
    main()
