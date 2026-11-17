# Optimizer Guide

**English** | [简体中文](optimizer.zh.md)

HyperParallel provides chained Muon+AdamW optimizers and learning rate schedulers, with sharded optimizer support and FSDP/HSDP integration.

## Core Concepts

| Optimizer | Description | Use case |
|--------|------|----------|
| AdamW | Standard AdamW optimizer | General training |
| Muon | Momentum-based optimizer | Large model training |
| ChainedOptimizer | Chained Muon+AdamW combination | Training with mixed parameter groups |

Muon applies momentum-based updates to selected parameter groups, such as embedding and projection layers, while AdamW handles the remaining parameters for hybrid optimization.

## API Overview

| Interface | Description |
|------|------|
| `AdamW` | Standard AdamW optimizer |
| `Muon` | Momentum-based optimizer |
| `ChainedOptimizer` | Chained optimizer combining Muon and AdamW |
| `get_hyper_optimizer` | Optimizer factory function |

---

## Basic Usage

### 1. Create a Chained Optimizer with get_hyper_optimizer

```python
from hyper_parallel.core.optimizer import get_hyper_optimizer

optimizer = get_hyper_optimizer(
    model=model,
    muon_params=muon_param_groups,   # Muon parameter groups
    adamw_params=adamw_param_groups,  # AdamW parameter groups
    muon_kwargs={"lr": 0.02, "momentum": 0.95},
    adamw_kwargs={"lr": 3e-4, "weight_decay": 0.1},
)
```

`get_hyper_optimizer` creates Muon and AdamW instances internally and combines them with `ChainedOptimizer`. Parameter grouping:

- `muon_params`: Parameter groups optimized by Muon, typically large projection parameters.
- `adamw_params`: Parameter groups optimized by AdamW, typically embeddings and biases.
- If either parameter group is an empty list, its optimizer is not created.

### 2. Parameter Grouping Example

```python
# Split model parameters into Muon and AdamW groups.
muon_params = []
adamw_params = []

for name, param in model.named_parameters():
    if not param.requires_grad:
        continue
    # Use Muon for large projection layers.
    if any(key in name for key in ["q_proj", "k_proj", "v_proj", "o_proj",
                                    "gate_proj", "up_proj", "down_proj"]):
        muon_params.append(param)
    else:
        # Use AdamW for embeddings, biases, norms, and other parameters.
        adamw_params.append(param)

optimizer = get_hyper_optimizer(
    model=model,
    muon_params=muon_params,
    adamw_params=adamw_params,
    muon_kwargs={"lr": 0.02, "momentum": 0.95},
    adamw_kwargs={"lr": 3e-4, "weight_decay": 0.1},
)
```

### 3. Training Loop

```python
for epoch in range(num_epochs):
    for batch in dataloader:
        output = model(batch)
        loss = criterion(output)
        loss.backward()

        # ChainedOptimizer handles the step for both optimizers.
        optimizer.step()
        optimizer.zero_grad()
        scheduler.step()
```

---

## Sharded Optimizers (FSDP/HSDP Integration)

Optimizer states are sharded automatically when used with FSDP/HSDP:

```python
from hyper_parallel import fully_shard, init_device_mesh
from hyper_parallel.core.optimizer import get_hyper_optimizer

mesh = init_device_mesh("npu", (dp_size,), mesh_dim_names=("dp",))
model = fully_shard(model, mesh=mesh)

# After FSDP sharding, optimizer states are also sharded automatically.
optimizer = get_hyper_optimizer(
    model=model,
    muon_params=muon_param_groups,
    adamw_params=adamw_param_groups,
    ...
)
```

---

## gradient scaling factor + clip_grad

```python
from hyper_parallel import fully_shard, init_device_mesh

mesh = init_device_mesh("npu", (dp_size,), mesh_dim_names=("dp",))
model = fully_shard(model, mesh=mesh)

# Set the gradient scaling factor.
model.set_gradient_scaling_factor(scale_factor=0.5)

# clip_grad enhancement: align the clip_grad_norm_ reduction with each gradient's process group.
# This ensures correct clip_grad behavior in hybrid parallelism.
```

---

## Performance Recommendations

1. **Muon parameters**: Typically assign large projection parameters to Muon and embeddings, biases, and norms to AdamW.
2. **Learning rates**: Muon's learning rate is typically larger than AdamW's, for example 0.02 versus 3e-4.
3. **Warmup**: Use warmup_steps to increase the learning rate linearly from zero to its target.
4. **Weight decay**: The recommended AdamW weight_decay is 0.1.
5. **FSDP integration**: Optimizer states are sharded automatically under FSDP, without additional configuration.

## Sinkhorn and DeepSeek-V4.1 head-wise Muon

`hyper_parallel.core.optimizer.sinkhorn.Sinkhorn` is an independent
`torch.optim.Optimizer`. It maintains FP32 momentum and balances dense 2D
updates using row/column statistics. It has no weight decay. `steps` must be
positive and odd; `correction` multiplies the configured learning rate.

For distributed parameters, Sinkhorn reduces statistics along the mesh axes
that shard the corresponding matrix dimension. Replicated axes do not
participate in these reductions. Each HSDP replica retains its own momentum;
there is no replica ownership or parameter broadcast. Gradients must already
be reduced, and all shards of a matrix must agree on gradient presence.
Sparse gradients and Partial parameter/gradient layouts are unsupported.

The existing factory accepts optional keyword-only `sinkhorn_params` and
`sinkhorn_kwargs`, alongside the existing Muon and AdamW arguments:

```python
optimizer = get_hyper_optimizer(
    model,
    muon_params=matrix_params,
    adamw_params=scalar_params,
    sinkhorn_params=embedding_params,
    sinkhorn_kwargs={"lr": 2.6e-4},
)
```

The YAML builder `hyper_parallel.components.optim.Sinkhorn` partitions
trainable parameters by identity: embeddings and output heads go to Sinkhorn,
other matrices to Muon, and remaining parameters to AdamW. Normalization
parameters use AdamW regardless of dimensionality. Each optimizer config may
supply `param_patterns` to override routing; conflicting selectors across tied
aliases are rejected. `sinkhorn_config.param_groups` supplies nonoverlapping
pattern-selected overrides, such as a different embedding learning rate.
Frozen parameters and `excluded_param_ids` are excluded. Host sparse tables
must remain excluded and managed by their existing sparse optimizer.

This new builder applies AdamW decay to normalization weights by default;
biases, ordinary scalar/vector parameters and explicitly named scales do not
decay. `no_decay_params` overrides those exclusions. Existing AdamW and Muon
builders retain their original defaults.

DeepSeek-V4.1 supplies its own callback through
`hyper_parallel.models.deepseek_v41.adapter.optim.headwise_muon.build_reshape_fn`.
It takes `attention_head_dim` and `indexer_head_dim` and returns a callable for
Muon's existing `reshape_fn` option. The callback produces shared-storage head
views for attention `q_b_proj`/`kv_proj` and indexer `q_b_proj`/`wk`; other
parameters use the original matrix shape. Widths must match the actual model,
including cropped models. No head-wise options are added to the core Muon API.

See the [text example](../../examples/training_demo/deepseek_v41/train_deepseek_v41_online.yaml)
and [VLM example](../../examples/training_demo/deepseek_v41/train_deepseek_v41_vlm_online.yaml)
for nested callback configuration and Engram parameter groups. The examples
configure optimizer hyperparameters; they do not reproduce the report's full
training schedule. Resume with the same optimizer partition and head layout.

## Optional Muon Gram Newton-Schulz

```python
from hyper_parallel.core.optimizer.muon import Muon

optimizer = Muon(parameters, use_gram=True, ns_backend="auto")
```

Mixed optimizer `muon_kwargs` or YAML `muon_config` also accept `muon_use_gram: true`.
Gram is disabled by default, preserving the original NS behavior. The option is
stored in parameter groups and optimizer checkpoints. An explicit `zeropower_fn`
continues to take precedence.

Configure the algorithm and symmetric GEMM backend independently:

| Configuration | `use_gram` | `ns_backend` |
| --- | --- | --- |
| Standard NS + native matmul (default) | `False` | `"cann"` or `None` |
| Static Gram selection + native matmul | `True` | `"cann"` or `None` |
| Standard NS + static Ascend C selection | `False` | `"auto"` |
| Static Gram selection + static Ascend C selection | `True` | `"auto"` |

`ns_backend="ascendc"` forces the Ascend C symmetric operator for validation.
Both options are stored in parameter groups. Enabling Gram alone keeps native
matmul. On non-NPU devices, `"cann"` refers to the framework's native matmul.

The configured coefficient schedule reaches Gram independently of coefficient
family, iteration count and floating-point input dtype. The algorithm is selected
from the actual NS input after reshaping and batching. With batch B, short side n and long side L, Gram
requires `n > 256` and `6 * B * n**2 * (2*L - 3*n) >= 50_000_000_000`, and
restarts at `gram_reset_iterations` (default `(2,)`). Other shapes use standard NS.
The static cost rule was calibrated for five steps with the default restart;
it remains a performance heuristic when configuring other schedules.
With `ns_backend="auto"`, `symmetric_matmul` selects Ascend C or native matmul
independently for each symmetric product. When the static rule rejects Gram,
standard NS retains its coefficient rounding order and buffer layout.
Rectangular products continue to use ordinary matmul.

Selection uses an offline heuristic without runtime timing or tuning; it does
not guarantee optimal performance for every shape. The optimizer retains its
existing BF16 NS preparation; direct NS calls retain the input dtype. Gram reassociation and fused epilogues change
rounding, so exact-arithmetic equivalence does not imply bitwise equality or
validated training convergence. Gram selection preserves the existing momentum,
parameter update and communication paths.

Muon also accepts `momentum_mode="vanilla"` for EMA momentum or `"legacy"`
for the existing formula, independently of Gram and the GEMM backend. The default
is `"vanilla"`. See the [VeOmni adapter](../../hyper_parallel/integration/veomni/README.md)
for EMA + Gram + Ascend C configuration through VeOmni.
