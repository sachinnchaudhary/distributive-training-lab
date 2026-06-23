# distributive-training-lab

## Purpose and Essence

distributive-training-lab(reproduced Megatron) made for intuitive understanding over distributive training which has real-end parallelism across DP, PP, TP, CP, EP and provides training engine for all of the parallelism. It not just have standard parallelism but optimized one like DP has ZeRO:1, 2, 3 and FSDP while TP has linear, mlp and attention. pipeline parallelism also contain Gpipe, 1f1b, interleaved-1f1b similarly CP has both ring attention and causal ring attention.

This repo is for someone who want to understand how distributive training works but dont want to understand just conceptually and in limited boundaries. i think this repo is "simple" and "useful" every part made for keeping in mind that it just becomes easy to grasp while making sure its do not become lossy. it has usefullness.

this repo is not competititon for Megatron but more of like bridiging step which will make you competent for such production grade distributive training code.

Pairing this blog with repo will help you a lot: [Distributive Training](https://sachinnchaudhary.github.io/publications/distributive_training.html).

## What This Repo Is

Its PyTorch-first distributed training lab. It implements the core mechanisms behind modern distributed LLM training in small files.The goal is to make each parallelism strategy understandable and testable.

The repo currently includes:

```text
DP   data parallelism
ZeRO optimizer/gradient/parameter sharding ideas
FSDP flat parameter sharding
TP   tensor parallel linear, MLP, attention, transformer blocks
PP   pipeline stages, P2P, GPipe, 1F1B, interleaved schedules
CP   context parallel ring causal attention
EP   expert parallel token dispatch and combine
Engine config, builder, trainer, checkpoint, precision, activation checkpointing
```

As i mentione above this is a lab, not a production replacement for Megatron-LM. It is intended to make the mechanisms legible before moving to production-grade systems.

## Why It Is Simple And Useful

The repo is simple because each file has one narrow job:

```text
data/dataset.py                  reads token shards
data/packing.py                  creates next-token examples
data/sampler.py                  shards samples across DP ranks
data/dataloader.py               returns a local batch
training/loss.py                 returns loss_sum and num_tokens
parallelism/data_parallel.py     syncs parameters, loss stats, gradients
tensor_parallel/linear.py        column and row parallel linear layers
pipeline_parallel/p2p.py         sends activations and gradients between stages
context_parallelism.py           ring attention over sequence shards
expert_parallelism.py            routes tokens to expert-owning ranks
engine/trainer.py                wires model, data, loss, optimizer, and parallelism
```

The most useful example of this simplicity is `meshtrain.core.distributed`. It separates physical runtime, logical geometry, real process groups, and ownership placement:

```text
meshtrain/core/distributed/
  runtime.py          who am I physically? rank, local_rank, world_size, device
  mesh.py             what is my logical coordinate? dp, pp, tp, cp, ep
  groups.py           who do I communicate with on each axis?
  placement.py        what data/layers/context/experts/tensor shard do I own?
  collectives.py      what named communication operation am I running?
  collective_trace.py what happened before and after communication?
```

That makes the geometry explicit. A flat rank is not just rank `5`; it can be read as a coordinate:

```text
rank 5 -> dp=0, pp=1, tp=1, cp=0, ep=0
```

Then the same geometry gives the communication groups:

```text
dp group    -> ranks that share a data-parallel replica
pp group    -> ranks that form one pipeline lane
tp group    -> ranks that split tensor dimensions
cp group    -> ranks that split sequence/context
ep group    -> ranks that own different experts
stage group -> ranks inside one pipeline stage across TP/CP/EP
```

also files are not toy-only pseudocode. The implementation uses real PyTorch distributed process groups and real correctness scripts. Most modules have a script that compares the distributed result against a single-process reference or validates the expected distributed invariant.

One example is collective tracing. Distributed training can feel invisible because an `all_reduce` or `all_gather` happens inside the runtime and then disappears. `core/distributed/collective_trace.py` makes the communication visible as structured JSONL. A trace event can show:

```text
rank
collective name
parallel axis
input tensor shape
output tensor shape
dtype
device
before summary
after summary
```

So instead of only saying "tensor parallel all-gather happened", the repo can record a readable event like:

```json
{
  "name": "all_gather_tensor_parallel",
  "axis": "tp",
  "rank": 1,
  "coord": {"dp": 0, "pp": 0, "tp": 1, "cp": 0, "ep": 0},
  "before": {"shape": [2, 4, 16], "dtype": "torch.float32"},
  "after": {"shape": [2, 4, 32], "dtype": "torch.float32"}
}
```

That makes the repo easier to debug and easier to learn from. You can connect the code, the process group, the tensor shape, and the actual communication event without guessing what happened.

The design principle is:

```text
one concept.
one lossless compressed file.
one correctness script for falsifiability.
```

## Repository Architecture

```text
meshtrain/
  core/
    distributed/
      runtime.py              process identity, device, backend
      mesh.py                 logical DP/PP/TP/CP/EP rank coordinates
      groups.py               PyTorch process group construction
      placement.py            layer, batch, context, expert, tensor ownership
      collectives.py          named collective wrappers
      collective_trace.py     optional JSONL communication tracing

  data/
    tokenizer.py              SentencePiece tokenizer loading
    dataset.py                token shard reader
    packing.py                causal LM example packing
    sampler.py                DP sample sharding
    batch.py                  batch dataclass and collation
    dataloader.py             packer + sampler integration

  model/
    standard_transformer.py   compact dense decoder-only transformer
    moe_transformer.py        MoE transformer used for CP/EP engine slices

  training/
    loss.py                   next-token cross entropy with loss_sum/num_tokens
    reference_training.py     single-process reference training step

  parallelism/
    data_parallel/
      data_parallelism.py     simple DP parameter/loss/gradient sync
      bucketed_data_parallel.py
                              bucketed, async, and overlap DP
      zero.py                 ZeRO-1/2/3 correctness implementation
      fsdp.py                 educational FSDP-style flat sharding

    tensor_parallel/
      linear.py               ColumnParallelLinear and RowParallelLinear
      mlp.py                  tensor-parallel SwiGLU MLP
      attention.py            tensor-parallel causal self-attention
      transformer.py          tensor-parallel transformer integration

    pipeline_parallel/
      p2p.py                  pipeline activation/gradient P2P
      stage.py                pipeline stage construction
      schedules.py            forward/backward, GPipe, 1F1B, interleaved schedules

    context_parallelism.py    ring causal attention over sequence shards
    expert_parallelism.py     all-to-all expert routing and combine

  engine/
    config.py                 validated engine configuration
    builder.py                runtime, mesh, groups, placement construction
    precision.py              dtype/autocast helpers
    activation_checkpoint.py  block activation checkpointing
    checkpoint.py             model/optimizer checkpoint save/load
    trainer.py                training loop for DP, PP, TP, CP, EP slices

scripts/
  download_parameter_golf.py
  inspect_distributed_world.py
  check_dp_correctness.py
  check_real_dp_correctness.py
  check_zero_correctness.py
  check_fsdp_correctness.py
  check_tensor_parallel_correctness.py
  check_pipeline_parallel_correctness.py
  check_context_parallelism_correctness.py
  check_expert_parallelism_correctness.py
  check_engine_training.py
  check_engine_5d_training.py
```

## Setup

Create an environment with Python and PyTorch, then install the repo dependencies:

```bash
pip install -r requirements.txt
```

Download the small tokenizer and one training shard:

```bash
python scripts/download_parameter_golf.py --train-shards 1
```

Quick tokenizer check:

```bash
python -c "from meshtrain.data.tokenizer import load_tokenizer, tokenizer_summary; t=load_tokenizer(); print(tokenizer_summary(t)); print(t.encode('hello world')[:10])"
```



## Correctness Commands

Use `torchrun` for distributed tests. If your environment does not expose `torchrun`, use:

```bash
python -m torch.distributed.run ...
```

### Data Pipeline

```bash
python -m meshtrain.data.dataset
python -m meshtrain.data.packing
python -m meshtrain.data.batch
python -m meshtrain.data.sampler
python -m meshtrain.data.dataloader
```

### Loss

```bash
python -m meshtrain.training.loss
```

### Simulated Data Parallel Correctness

This runs on one process and compares a manually sharded DP update against a full-batch reference update:

```bash
python scripts/check_dp_correctness.py
```

### Real Data Parallel Correctness

Simple DP:

```bash
torchrun --nproc_per_node=2 scripts/check_real_dp_correctness.py \
  --train-path data/datasets/fineweb10B_sp1024/fineweb_train_000000.bin \
  --dp-mode simple
```

Bucketed sync DP:

```bash
torchrun --nproc_per_node=2 scripts/check_real_dp_correctness.py \
  --train-path data/datasets/fineweb10B_sp1024/fineweb_train_000000.bin \
  --dp-mode bucketed-sync
```

Bucketed async DP:

```bash
torchrun --nproc_per_node=2 scripts/check_real_dp_correctness.py \
  --train-path data/datasets/fineweb10B_sp1024/fineweb_train_000000.bin \
  --dp-mode bucketed-async
```

Backward-overlapped bucketed DP:

```bash
torchrun --nproc_per_node=2 scripts/check_real_dp_correctness.py \
  --train-path data/datasets/fineweb10B_sp1024/fineweb_train_000000.bin \
  --dp-mode bucketed-overlap
```


```

### ZeRO Correctness

```bash
torchrun --nproc_per_node=2 scripts/check_zero_correctness.py --stage 1
torchrun --nproc_per_node=2 scripts/check_zero_correctness.py --stage 2
torchrun --nproc_per_node=2 scripts/check_zero_correctness.py --stage 3
```

### FSDP Correctness

```bash
torchrun --nproc_per_node=2 scripts/check_fsdp_correctness.py \
  --global-batch-size 8 \
  --in-features 16 \
  --out-features 32
```

### Tensor Parallel Correctness

Test column parallel linear, row parallel linear, tensor-parallel MLP, and tensor-parallel attention:

```bash
torchrun --nproc_per_node=2 scripts/check_tensor_parallel_correctness.py \
  --case all \
  --batch-size 2 \
  --seq-len 4 \
  --in-features 16 \
  --out-features 32 \
  --hidden-dim 64 \
  --n-heads 4
```

Individual cases:

```bash
torchrun --nproc_per_node=2 scripts/check_tensor_parallel_correctness.py --case linear
torchrun --nproc_per_node=2 scripts/check_tensor_parallel_correctness.py --case mlp
torchrun --nproc_per_node=2 scripts/check_tensor_parallel_correctness.py --case attention
```

### Pipeline Parallel Correctness

```bash
torchrun --nproc_per_node=4 scripts/check_pipeline_parallel_correctness.py \
  --case all
```

Common individual cases:

```bash
torchrun --nproc_per_node=4 scripts/check_pipeline_parallel_correctness.py --case forward
torchrun --nproc_per_node=4 scripts/check_pipeline_parallel_correctness.py --case forward-backward
torchrun --nproc_per_node=4 scripts/check_pipeline_parallel_correctness.py --case gpipe
torchrun --nproc_per_node=4 scripts/check_pipeline_parallel_correctness.py --case 1f1b
torchrun --nproc_per_node=4 scripts/check_pipeline_parallel_correctness.py --case interleaved
```

### Context Parallel Correctness

Ring causal attention over sequence shards:

```bash
torchrun --nproc_per_node=4 scripts/check_context_parallelism_correctness.py \
  --batch-size 2 \
  --seq-len 16 \
  --n-heads 4 \
  --head-dim 8
```

### Expert Parallel Correctness

All-to-all token dispatch, local experts, and combine:

```bash
torchrun --nproc_per_node=4 scripts/check_expert_parallelism_correctness.py \
  --num-tokens 64 \
  --hidden-dim 32 \
  --num-experts 8
```

### Engine Training: DP And 3D Parallelism

DP-only engine training:

```bash
torchrun --nproc_per_node=8 scripts/check_engine_training.py \
  --mode dp \
  --steps 1 \
  --global-batch-size 8 \
  --microbatch-size 1 \
  --progress
```

DP + PP + TP engine training:

```bash
torchrun --nproc_per_node=8 scripts/check_engine_training.py \
  --mode dp-pp-tp \
  --steps 1 \
  --global-batch-size 8 \
  --microbatch-size 2 \
  --progress
```

### Engine Training: 5D Slices

Full `dp * pp * tp * cp * ep` must equal the number of launched processes.

PP + TP + EP slice:

```bash
torchrun --nproc_per_node=8 scripts/check_engine_5d_training.py \
  --dp 1 --pp 2 --tp 2 --cp 1 --ep 2 \
  --steps 1 \
  --global-batch-size 8 \
  --microbatch-size 4 \
  --num-experts 4 \
  --n-layers 2 \
  --progress
```

PP + TP + CP slice:

```bash
torchrun --nproc_per_node=8 scripts/check_engine_5d_training.py \
  --dp 1 --pp 2 --tp 2 --cp 2 --ep 1 \
  --steps 1 \
  --global-batch-size 8 \
  --microbatch-size 4 \
  --num-experts 4 \
  --n-layers 2 \
  --progress
```

DP + TP + EP slice:

```bash
torchrun --nproc_per_node=8 scripts/check_engine_5d_training.py \
  --dp 2 --pp 1 --tp 2 --cp 1 --ep 2 \
  --steps 1 \
  --global-batch-size 8 \
  --microbatch-size 1 \
  --num-experts 4 \
  --progress
```

DP + PP + CP slice:

```bash
torchrun --nproc_per_node=8 scripts/check_engine_5d_training.py \
  --dp 2 --pp 2 --tp 1 --cp 2 --ep 1 \
  --steps 1 \
  --global-batch-size 8 \
  --microbatch-size 2 \
  --num-experts 4 \
  --progress
```

DP + CP + EP slice:

```bash
torchrun --nproc_per_node=8 scripts/check_engine_5d_training.py \
  --dp 2 --pp 1 --tp 1 --cp 2 --ep 2 \
  --steps 1 \
  --global-batch-size 8 \
  --microbatch-size 1 \
  --num-experts 4 \
  --progress
```

## Known Limitations

This repo is intentionally educational. There is ton of optimization needed but i think current state of repo itself is enough for understanding lot of foundation on which serious production can be thought of.
 Current limitations are bounded by compute too. Due to very scarce compute I need to run all things very conservatively:

```text
Full PP + TP + EP works for one transformer block per PP stage.
Multiple MoE/TP blocks per PP stage need stronger block-level collective scheduling.
ZeRO and FSDP are implemented as correctness mechanisms but are not fully wired into EngineTrainer.
Pipeline checkpoint resume is not wired yet.
The engine is suitable for small controlled runs, not production-scale fault-tolerant training.
```
 

## Current Status

Implemented and tested:

```text
distributed runtime identity
rank mesh and parallel coordinates
process group construction
stage-local groups for 5D scheduling
rank ownership placement
collective wrappers
JSONL communication tracing
Parameter Golf/FineWeb token shard data path
standard dense transformer
MoE transformer for CP/EP engine slices
next-token loss with loss_sum / num_tokens accounting
reference trainer
simple DP
bucketed DP
async bucketed DP
backward-overlapped bucketed DP
ZeRO-1, ZeRO-2, ZeRO-3 correctness implementations
FSDP-style flat parameter sharding
tensor parallel linear, MLP, attention, transformer block
pipeline forward/backward, GPipe, 1F1B, interleaved schedule
context parallel ring causal attention
expert parallel token routing
engine DP, PP, TP, CP, EP slice training
```
