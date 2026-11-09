# Optional framework integrations

## VeOmni with genuine Torch DTensors

This experimental entry keeps parameters and gradients as real
`torch.distributed.tensor.DTensor` objects, while Hyper owns FSDP/HSDP
communication and scheduling. It does not replace Torch types globally.

With VeOmni's adapter startup entry, set:

```yaml
train:
  use_hyper_parallel: true
```

Then launch the original task:

```bash
torchrun --standalone --nproc_per_node=2 tasks/train_text.py config.yaml
```

Use the task's normal Torch/FSDP2 configuration; do not also select VeOmni's
older Hyper runtime branch. For application code, call
`hyper_parallel.integration.veomni.apply_veomni_patch()` before constructing the original
trainer, or use `VeOmniAdapter()` as a context manager. Importing the module alone
does not activate it or import VeOmni. Startup/shutdown binding changes are not
thread-safe.

VeOmni's FSDP constructor, FSDP-module recognition, Muon factory and clipping
entry are bound. Managed Hyper models use Hyper's native gradient clipping;
unmanaged models retain VeOmni's original clipping. The global Torch clipping
function is unchanged. Native parameter loading, runtime, training loop and
DCP functions stay unchanged. AdamW uses the native factory. Muon uses Hyper's
original distributed implementation and native VeOmni parameter grouping,
scheduler and checkpoint serialization. Transient layout-correct optimizer
templates avoid the checkpoint API's synthetic zero-LR optimizer step.

For direct use without VeOmni:

```python
from hyper_parallel import fully_shard

fully_shard(model, mesh=hyper_mesh, use_torch_dtensor=True)
```

`use_torch_dtensor=True` selects genuine Torch storage in the existing Hyper
builder. The default `False` preserves native behavior: ordinary and Hyper
parameters use Hyper DTensors; existing Torch inputs retain Torch DTensors.
VeOmni's adapter supplies `True`; the application does not need another entry.

Plain parameters enter Hyper partitioning directly; only the resulting shards are
wrapped as Torch DTensors. Existing Torch DTensors keep their
source mesh, placements and global metadata. Hyper composes supported disjoint
source and FSDP axes using the existing communication algorithm. Partial source
placements and source sharding overlapping the FSDP domain are rejected. No
conversion to Hyper DTensor is required for Hyper clipping or native DCP.
Composing source and FSDP sharding on the same tensor dimension currently needs
an unsupported strided Torch placement and is rejected as well.

Existing NPU bring-up covers small Qwen3/Qwen3-MoE, FSDP/HSDP, accumulation,
non-reentrant activation checkpointing, broadcast loading, Hyper Muon and
checkpoint continuation. This is not full-size performance or all-model
acceptance. CUDA, CPU offload, complete TP/EP model execution and arbitrary modality paths
remain unverified. HSDP Muon's existing checkpoint limitation for never-updated
owner parameters is unchanged. Custom Muon groups and head splitting are
explicitly unsupported in this entry.

The explicit storage flag has CPU/Gloo regression coverage; its NPU comparison
against the earlier input-preparation path is pending.

Hyper clipping reads Torch DTensor placements and local shards, then uses its
existing norm reduction and clipping algorithm. It does not convert training
parameters or substitute Torch's norm algorithm. Cross-representation numerical
checks and same-path checkpoint continuation are separate acceptance gates.
