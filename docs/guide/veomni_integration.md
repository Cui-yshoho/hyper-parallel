# Optional VeOmni integration

This package supplies VeOmni's opt-in Hyper execution capabilities. VeOmni owns
trainer selection, runtime construction/composition, native training loops,
data, callbacks and model/Omni checkpoint managers.
Install compatible VeOmni and HyperParallel revisions together. VeOmni is an
optional framework dependency of this adapter, not of Hyper's core/root.

## Execution ownership

- `parallelize.py`: Hyper FSDP/HSDP, EP/FP32 groups and device-mesh adaptation.
- `prefetch.py`: declared compute-stage wrapping and prefetch policy.
- `module_utils.py`: materialization/loading and tensor-layout interoperability,
  reusing native readers/converters and initialization helpers.
- `muon.py` and `optimizer.py`: Hyper Muon, compatible native Muon, optional
  Gram/QuACK callbacks and scheduler/state ownership.
- `clip_grad_norm.py` and `checkpoint.py`: clipping and low-level Hyper DCP
  protocol. Lifecycle, orchestration and paths stay in VeOmni's managers.

The adapter uses explicit `veomni.*` imports for native helpers. Import it only
after selecting Hyper; it does not install global patches to select runtimes or
models. Modality-specific optional dependencies remain lazy. The existing
reentrant checkpoint replacement is preserved, not introduced by this relocation.
The default non-reentrant path performs no such replacement.

The independent FSDP and Muon flags remain independent: optimizer-only Hyper uses
native wrapping and clipping; Hyper FSDP can retain native Muon. Checkpoint
selection is independent too: VeOmni's existing `train.checkpoint.manager: dcp`
is the default. This package's DCP protocol runs only with explicit
`train.checkpoint.manager: hyper`, currently requiring Hyper FSDP.
Native VeOmni execution algorithms are unchanged. Hyper's Muon changes are
limited to the separately reviewed overlap, momentum and Gram-NS feature set.
Supported persistence remains DCP/state_dict-based. Relocating private classes
does not preserve whole-object Python pickle paths for runtimes or optimizers;
the adapter does not install module aliases to emulate those private paths.

## Compute-stage policy

`hyper_parallel.integration.veomni.StageWrapPolicy` is used by the Hyper-owned
VeOmni FSDP builder in this package's `parallelize.py`. Importing the package root
or this lightweight policy does not import VeOmni. Explicit execution adapters
reuse VeOmni helpers; the default native Torch path must not import the adapter.

The adapter declares known Qwen3-MoE and Qwen3-VL-MoE text/vision attention and
MLP boundaries. It does not infer arbitrary execution order from the module
tree. CUDA and NPU use these declarations; compiled, unknown, shared-parameter,
empty-stage and persistent ExtraParallel blocks retain whole-block wrapping.
Explicit nested wrapping targets also keep their existing whole-block flow,
instead of adding a second parent stage chain for the same units.

## Builder contract

VeOmni supplies configuration and native process groups. This package's Hyper
builder owns the ordered target-block list, wrapping, precision policy,
offload and loading. The builder calls:

1. `StageWrapPolicy(model, target_modules, backend, device_type, compiled)`
   before wrapping. `backend` supplies `fully_shard` and `module_type`.
2. `exclude_persistent(params)` after identifying FSDP-ignored parameters.
3. `wrap_submodules(block, fsdp_kwargs)` after building nested EP/FP32 units
   in `block._fsdp_modules`. `needs_parent_wrap(block)` preserves an enclosing
   output-cast boundary when needed; call `record_parent(block)` after wrapping it.
   Other residual parameters are claimed by root.
4. `wrap_model_io(fsdp_kwargs)` before root wrapping selects independent input
   embeddings and the eligible patched Qwen3-MoE fused-loss head.
5. `configure_prefetch()` after wrapping, only when explicit prefetch is enabled.
   False means no split blocks were selected and the builder retains its old
   whole-block prefetch path.

The eligible Qwen3-MoE head changes locally to VeOmni's `FusedCrossEntropy`
subclass without replacing the module or its weight. The generated model
dispatches loss through that head's forward so FSDP hooks execute. Its gather
is retained until backward. Shared, custom and already-hooked heads stay on root;
root keeps `reshard_after_forward=False` for direct-weight consumers.
No global class or forward method is patched. Frozen models disable explicit
prefetch; output precision boundaries are preserved by CUDA parent wrapping or
NPU whole-block fallback.

## Scheduling

Forward prefetch requests the following two declared stages:
`attention[i] -> mlp[i], attention[i+1]` and
`mlp[i] -> attention[i+1], mlp[i+1]`.
Nested EP/FP32 units follow their enclosing stage, preserving their previously
selected meshes and precision policies.

Backward targets account for whole-block checkpoint replay: both stages of a
split block request the preceding block's ordered stages. A split block after
an unsplit block also connects its final stage to its own previous stage.
The adapter changes neither checkpointing nor gradient reduction intervals.

This policy requires hardware validation for overlap, memory and throughput.
CPU schedule and ownership tests alone cannot establish a performance benefit,
nor validate all model-specific direct-weight consumers.

## Optional Muon features

VeOmni's `hyper_muon_fsdp_overlap` (default false) and positive optional
`hyper_muon_fsdp_overlap_memory_budget_bytes` forward to Hyper's bounded
gather/NS/update pipeline. `hyper_muon_hsdp_replicate_dedup` defaults true;
false disables replica ownership deduplication. `hyper_muon_momentum_mode`
defaults to `legacy`; `vanilla` selects EMA-style momentum and changes the
optimization algorithm.
The VeOmni adapter records the momentum formula in checkpoints and rejects
restoring into another formula. Unmarked older checkpoints imply `legacy`.
Direct core Muon users must keep the constructor's momentum mode unchanged
across restore; replica ownership settings and topology must also remain fixed.

`hyper_muon_ns_backend` defaults to `veomni`, preserving VeOmni's existing
NS implementation selection. `native` uses Hyper NS; `gram` uses the optional
Gram adapter through `get_hyper_optimizer(ns_backend="gram")`, requiring CUDA
SM90+, CUDA 12.9+ and `gram-newton-schulz`. A Gram backend and an explicit
`zeropower_fn` are mutually exclusive. Custom callbacks declaring `ns_config`
receive each group's coefficient schedule and epsilon.

## Optional expert gradient precision

Torch `MixedPrecisionPolicy(grad_dtype_fn=selector)` evaluates the selector on
each original parameter before sharding. Return a Torch dtype to set the stable
unsharded autograd leaf's gradient dtype, or None to preserve the default.
The parameter compute dtype and reduction dtype remain independently controlled.
Explicit selection requires a Torch version supporting `Tensor.grad_dtype`;
This option uses the current Torch-only FSDP core and leaves its existing
per-parameter precision overrides unchanged.

VeOmni exposes `hyper_expert_grad_dtype_fp32` (default false) for CUDA Qwen3-MoE,
EP=1, no LoRA, fused Triton MoE, BF16 parameters and FP32 gradient reduction.
Only expert `gate_up_proj` and `down_proj` leaves are selected. Unsupported
combinations fail instead of silently changing unrelated parameters.
