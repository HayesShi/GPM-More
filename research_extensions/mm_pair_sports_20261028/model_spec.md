# Sports pair pilot model specification

This directory is independent research code. It does not modify production GPM
or launch a training job. All arms instantiate the same `PairSportsModel` and
load exactly the same initial state. The ordinary arm calls `use_joint=False`;
the joint arm calls `use_joint=True`.

## Input and retained backbone

The model consumes the complete frozen CLIP feature vector `[N,1024]` as a
single fused input. The modality coordinate order has not been verified, so
there is no unsupported 512/512 modality split. The hidden width is 256.

`RelationalPatternModel` and `GPMBackbone` are imported directly from
`delivery/GPM-More/src/gpm_more`. This retains its input projection, contextual
pattern Transformer, multiscale `(2,4,6,8)` observation mask, original all-query
position mean, anonymous-identity GRU and projection, learned 93-dimensional
observed-relation adapter, action embedding (SRW=0), original sum-phase closure
token, outer residual Transformer and root readout. The NC output head is
replaced with the common LP head. A LayerNorm is applied to z in both arms.
Pattern dropout is 0.1, outer dropout 0.0, with one layer in each Transformer,
one pattern head and four outer heads, as in the raw_sum256 defaults.

The original closure implementation is imported unchanged. In particular,
`cos((t+s)*omega)` is retained, with position zero excluded and all same-walk
terms subtracted. The new score does not reinterpret this as relative RoPE.

## API

```python
model = PairSportsModel(input_dim=1024, hidden_dim=256, rank=16)
observed = sample_walks(train_csr, unique_roots, num_walks=16, seed=seed,
                        excluded_edges=batch_training_positive_edges_2_by_E)
encoded = model.encode(x_global, paths, relations, occurrence_valid=None)
logits = model.score(encoded, src_idx, dst_idx, use_joint=True)
```

Paths have `[B,K,9]`, relations `[B,K,93]`, and the optional validity mask has
`[B,K,9]` or `[B,K,8]`. Returns `z[B,256]`, `ids[B,K*8]`, `left/right[B,K*8,16]`
and boolean `valid[B,K*8]`. IDs are sorted, unique among valid slots, and padded
by the non-node sentinel N. There are no trainable identity embeddings.
`score` takes endpoint row indexes into this common encoded root batch.

## Depth-preserving low-rank interaction

For each depth t=1..8, independent learned U_t and V_t have shape `[256,16]`.
They are initialized with independent normal entries of standard deviation
`1/sqrt(256)`. For each valid occurrence, apply its depth-specific projection
before aggregation by exact global witness ID:

```
c = 1/sqrt(8*16)
A_u(w) = c/K * sum_{p,t: paths[u,p,t]=w, t>=1} h[u,p,t] @ U[t-1]
B_u(w) = c/K * sum_{p,t: paths[u,p,t]=w, t>=1} h[u,p,t] @ V[t-1]
F(u,v) = 0.5 * sum_{w in support(u) intersection support(v)}
                  (A_u(w) dot B_v(w) + B_u(w) dot A_v(w))
s(u,v) = MLP(LayerNorm(z_u) * LayerNorm(z_v)) + joint_scale * F(u,v)
```

The MLP is Linear(256,256), ReLU, Linear(256,1), identical for all arms. The
learned scalar `joint_scale` starts at **0.1**, so gradients reach U/V on the
first backward. Fixed c stabilizes the depth/rank contraction; F is multiplied
by `1/(8*16)` relative to unscaled occurrence projections. No normalization by
observed support overlap is applied. Witness vectors use sum/K, without an
additional division by occurrences or positions. The factorization permits
all t-by-s depth interactions while avoiding explicit concatenated depth
vectors, the full W matrix, or a K²L² tensor.

Sorting and scatter-add combine repeated visits. Batched `torch.searchsorted`
joins sorted witness IDs at score time. Pair chunks default to 2048. At K=16,
M=128, padded float32 left/right storage is 16,384 bytes per root, plus 1024
bytes of int64 IDs, 128 bytes of boolean validity and 1024 bytes of float32 z
(about 18.1 KiB per root total). This is the encoder cache, not the path
Transformer activation budget. Projection storage can be reduced using a
validated lower-precision cache in a later run, but the initial interface is
float32.

Two projection factors cost O(B*K*8*256*16) multiply-accumulate operations;
per-root sorting costs O(B*M*log M). Each pair join costs O(M*log M) comparisons
plus O(M*16) masked contraction work in this padded implementation. Numeric
work is performed for padded slots and then masked; it is not claimed to be
strictly proportional to actual matches. The pair chunk limit bounds the
temporary per-pair witness tensor. No claim excludes path encoding, graph
sampling, relation probing, or sorting from the total runtime.

## Graph context and isolated vertices

Sampling uses only the caller-supplied train-only `gpm_more.sampling.CSRGraph`.
For each training batch, pass **all positive target edges together**, in `[2,E]`
form (preferred; a 2x2 input is interpreted as `[2,E]`). Both orientations are
removed. Exclusion is an exact sorted overlay over existing CSR entries,
queried inside the neighbor-choice kernel and every induced-adjacency probe.
Allowed neighbor ranks are sampled uniformly without a rejection loop.

The original descriptor routine then produces the true 36 equality bits, 36
induced-edge bits, and 21 rooted graphlet/sentinel bits on the same effective
graph. It is never generated from the unmodified graph and masked afterward.
The declared primitive budget remains K*(8+36)=704 per root at K=16; overlay
construction and lookup also consume CPU time and must be recorded by the
runner. Sampling cannot verify that its caller has not supplied val/test edges
in the input CSR; the data/runner layer owns that split guarantee.

When no allowed edge exists, a walk fills remaining positions with the same
node. These positions have `occurrence_valid=False` and contribute no witness
value. If the optional mask is omitted, encode recognizes a repeated
consecutive ID as padding because this CSR is loopless. Ordinary root encoding
and the original closure retain their historical repeated-node behavior.
An isolated endpoint therefore has **F=0**, including for itself, and cannot
invent cross-root same-witness evidence from padding.

Evaluation uses a fixed K=16 or K=32 bank on the unchanged train-only CSR, with
no query-specific deletion. Cache a root once and reuse its bank/encoding for
all candidate pairs; scatter repeated endpoint roots from that cache. The
sampler permits repeated roots but does not itself impose this cache policy.

## Critical-path verification

Run `model_check.py --output PATH` with the configured Python. It samples an
actual eight-node graph on CPU and compares the exclusion overlay exactly
with a physically edge-deleted CSR, including all 93 descriptors. It checks
the absence of excluded transitions/probes, sorted support, isolated padding,
the base-plus-joint equation, undirected symmetry, common global ID-renaming
invariance, repeated-root cache scatter, sensitivity to broken cross-end
identity, and one finite backward through both low-rank factors and the active
ordinary encoder modules. There is no optimizer step or training launch. The
JSON result records actual numerical errors and gradient norms. Full GPU
memory/runtime validation remains the parent runner's responsibility.
