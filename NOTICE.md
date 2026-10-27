# Attribution

The active node backbone in `src/gpm_more/backbone.py` adapts the official
[GPM](https://github.com/Zehong-Wang/GPM) implementation at revision
`2ed3ebf7df8d8401520e6ec3bdbd1c2b291b5784` (MIT, © 2025 Zehong Wang).
Its original notice is preserved in `third_party/GPM-LICENSE`.

This repository retains the active Transformer Pattern encoder, anonymous-path
GRU, multiscale masks, outer residual Transformer and readout used in the local
raw_sum experiments. Disabled VQ, unused template layers, inactive legacy phase
modules and unused pretraining heads are omitted from this standalone package.
See `docs/METHOD.md` for the exact supported architecture and result scope.

The fixed cross-walk sum-phase operator, input-level feedback control, cached
observation interface, experiment aggregation and documentation are supplied
with GPM-More. GraphLand datasets are obtained separately from their publishers;
this repository does not redistribute datasets or pretrained checkpoints.
