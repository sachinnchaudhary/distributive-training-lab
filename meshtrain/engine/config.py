from __future__ import annotations

from dataclasses import dataclass

from meshtrain.model.standard_transformer import TransformerConfig


PP_SCHEDULES = {"none", "gpipe", "1f1b", "interleaved_1f1b"}
OPTIMIZERS = {"adamw"}
PRECISION_DTYPES = {"fp32", "bf16", "fp16"}
CHECKPOINT_GRANULARITIES = {"block"}


@dataclass(frozen=True)
class ParallelismConfig:
    dp: int = 1
    tp: int = 1
    pp: int = 1
    cp: int = 1
    ep: int = 1

    pp_schedule: str = "none"
    virtual_stages_per_rank: int = 1

    zero_stage: int = 0
    fsdp: bool = False

    @property
    def world_size(self) -> int:
        return self.dp * self.tp * self.pp * self.cp * self.ep

    def __post_init__(self) -> None:
        for name in ("dp", "tp", "pp", "cp", "ep"):
            value = getattr(self, name)
            if value < 1:
                raise ValueError(f"{name} must be >= 1")

        if self.pp_schedule not in PP_SCHEDULES:
            raise ValueError(
                f"pp_schedule must be one of {sorted(PP_SCHEDULES)}, "
                f"got {self.pp_schedule!r}"
            )

        if self.virtual_stages_per_rank < 1:
            raise ValueError("virtual_stages_per_rank must be >= 1")

        if self.pp == 1 and self.pp_schedule != "none":
            raise ValueError("pp_schedule must be 'none' when pp=1")

        if self.pp > 1 and self.pp_schedule == "none":
            raise ValueError("pp_schedule must be set when pp>1")

        if self.pp_schedule != "interleaved_1f1b" and self.virtual_stages_per_rank != 1:
            raise ValueError(
                "virtual_stages_per_rank must be 1 unless pp_schedule='interleaved_1f1b'"
            )

        if self.zero_stage not in {0, 1, 2, 3}:
            raise ValueError("zero_stage must be one of {0, 1, 2, 3}")

        if self.fsdp and self.zero_stage != 0:
            raise ValueError("fsdp and ZeRO are mutually exclusive for now")


@dataclass(frozen=True)
class TrainingConfig:
    max_steps: int = 1000
    global_batch_size: int = 8
    microbatch_size: int = 1
    grad_accum_steps: int = 1
    seed: int = 0
    log_every: int = 10
    checkpoint_every: int = 100

    def __post_init__(self) -> None:
        for name in (
            "max_steps",
            "global_batch_size",
            "microbatch_size",
            "grad_accum_steps",
            "log_every",
            "checkpoint_every",
        ):
            value = getattr(self, name)
            if value < 1:
                raise ValueError(f"{name} must be >= 1")


@dataclass(frozen=True)
class OptimizerConfig:
    name: str = "adamw"
    lr: float = 3e-4
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    eps: float = 1e-8

    @property
    def betas(self) -> tuple[float, float]:
        return (self.beta1, self.beta2)

    def __post_init__(self) -> None:
        if self.name not in OPTIMIZERS:
            raise ValueError(f"optimizer must be one of {sorted(OPTIMIZERS)}")
        if self.lr <= 0:
            raise ValueError("lr must be > 0")
        if self.weight_decay < 0:
            raise ValueError("weight_decay must be >= 0")
        if not 0 <= self.beta1 < 1:
            raise ValueError("beta1 must be in [0, 1)")
        if not 0 <= self.beta2 < 1:
            raise ValueError("beta2 must be in [0, 1)")
        if self.eps <= 0:
            raise ValueError("eps must be > 0")


@dataclass(frozen=True)
class DataConfig:
    train_path: str
    tokenizer_path: str = "data/tokenizers/fineweb_1024_bpe.model"
    seq_len: int = 1024
    global_batch_id_start: int = 0

    def __post_init__(self) -> None:
        if not self.train_path:
            raise ValueError("train_path must be non-empty")
        if not self.tokenizer_path:
            raise ValueError("tokenizer_path must be non-empty")
        if self.seq_len < 1:
            raise ValueError("seq_len must be >= 1")
        if self.global_batch_id_start < 0:
            raise ValueError("global_batch_id_start must be >= 0")


@dataclass(frozen=True)
class PrecisionConfig:
    dtype: str = "bf16"
    autocast: bool = True
    grad_scaler: bool = False

    def __post_init__(self) -> None:
        if self.dtype not in PRECISION_DTYPES:
            raise ValueError(f"dtype must be one of {sorted(PRECISION_DTYPES)}")
        if self.dtype == "bf16" and self.grad_scaler:
            raise ValueError("grad_scaler is not needed for bf16")


@dataclass(frozen=True)
class ActivationCheckpointConfig:
    enabled: bool = False
    granularity: str = "block"
    every_n_layers: int = 1

    def __post_init__(self) -> None:
        if self.granularity not in CHECKPOINT_GRANULARITIES:
            raise ValueError(
                f"granularity must be one of {sorted(CHECKPOINT_GRANULARITIES)}"
            )
        if self.every_n_layers < 1:
            raise ValueError("every_n_layers must be >= 1")


@dataclass(frozen=True)
class CheckpointConfig:
    output_dir: str = "checkpoints"
    resume_from: str | None = None
    save_optimizer: bool = True

    def __post_init__(self) -> None:
        if not self.output_dir:
            raise ValueError("output_dir must be non-empty")


@dataclass(frozen=True)
class MoEConfig:
    enabled: bool = False
    num_experts: int = 1

    def __post_init__(self) -> None:
        if self.num_experts < 1:
            raise ValueError("num_experts must be >= 1")
        if self.enabled and self.num_experts < 2:
            raise ValueError("enabled MoE requires num_experts >= 2")


@dataclass(frozen=True)
class EngineConfig:
    parallelism: ParallelismConfig
    model: TransformerConfig
    training: TrainingConfig
    optimizer: OptimizerConfig
    data: DataConfig
    precision: PrecisionConfig = PrecisionConfig()
    activation_checkpointing: ActivationCheckpointConfig = ActivationCheckpointConfig()
    checkpoint: CheckpointConfig = CheckpointConfig()
    moe: MoEConfig = MoEConfig()

    def validate(self, *, world_size: int | None = None) -> None:
        parallelism = self.parallelism
        model = self.model
        training = self.training

        if world_size is not None and world_size != parallelism.world_size:
            raise ValueError(
                f"parallelism requires world_size={parallelism.world_size}, "
                f"got runtime world_size={world_size}"
            )

        if model.dim % parallelism.tp != 0:
            raise ValueError("model dim must be divisible by tp")

        if model.n_heads % parallelism.tp != 0:
            raise ValueError("model n_heads must be divisible by tp")

        if model.seq_len % parallelism.cp != 0:
            raise ValueError("model seq_len must be divisible by cp")

        if self.moe.enabled and self.moe.num_experts % parallelism.ep != 0:
            raise ValueError("num_experts must be divisible by ep")

        if parallelism.ep > 1 and not self.moe.enabled:
            raise ValueError("ep > 1 requires moe.enabled=True")

        if self.data.seq_len != model.seq_len:
            raise ValueError("data seq_len must match model seq_len")

        pipeline_parts = parallelism.pp * parallelism.virtual_stages_per_rank
        if model.n_layers % pipeline_parts != 0:
            raise ValueError(
                "model n_layers must be divisible by "
                "pp * virtual_stages_per_rank"
            )

        if training.global_batch_size % parallelism.dp != 0:
            raise ValueError("global_batch_size must be divisible by dp")

        if training.global_batch_size % training.microbatch_size != 0:
            raise ValueError("global_batch_size must be divisible by microbatch_size")

        if parallelism.pp > 1 and training.microbatch_size >= training.global_batch_size:
            raise ValueError(
                "pipeline parallel training should use at least two microbatches"
            )
