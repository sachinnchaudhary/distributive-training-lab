from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, Sequence


DEFAULT_REPO_ID = "willdepueoai/parameter-golf"
DEFAULT_VARIANT = "sp1024"
DEFAULT_TOKENIZER_MODEL = "data/tokenizers/fineweb_1024_bpe.model"

class TextTokenizer(Protocol):

    @property
    def name_or_path(self) -> str:
        ...

    @property
    def vocab_size(self) -> int:
        ...

    @property
    def eos_token_id(self) -> int | None:
        ...

    @property
    def bos_token_id(self) -> int | None:
        ...

    def encode(self, text: str, *, add_eos: bool = False) -> list[int]:
        ...

    def decode(self, ids: Sequence[int]) -> str:
        ...


@dataclass(frozen=True)
class HuggingFaceTokenizer:
   
    name_or_path: str
    tokenizer: Any

    @classmethod
    def from_pretrained(
        cls,
        name_or_path: str,
        **kwargs: Any,
    ) -> "HuggingFaceTokenizer":
      
        try:
            from transformers import AutoTokenizer
        except ImportError as exc:
            raise ImportError(
                "Loading a Hugging Face tokenizer requires `transformers`."
            ) from exc

        tokenizer = AutoTokenizer.from_pretrained(name_or_path, **kwargs)
        return cls(name_or_path=name_or_path, tokenizer=tokenizer)

    @property
    def vocab_size(self) -> int:
        return len(self.tokenizer)

    @property
    def eos_token_id(self) -> int | None:
        return self.tokenizer.eos_token_id

    @property
    def bos_token_id(self) -> int | None:
        return self.tokenizer.bos_token_id

    def encode(self, text: str, *, add_eos: bool = False) -> list[int]:
        ids = self.tokenizer.encode(text, add_special_tokens=False)
        ids = list(ids)

        if add_eos:
            eos = self.eos_token_id
            if eos is None:
                raise ValueError("tokenizer has no eos_token_id")
            ids.append(eos)

        return ids

    def decode(self, ids: Sequence[int]) -> str:
        return self.tokenizer.decode(list(ids), skip_special_tokens=False)


@dataclass(frozen=True)
class ParameterGolfTokenizer:
    model_path: Path
    processor: Any

    @classmethod
    def from_file(cls, model_path: str | Path = DEFAULT_TOKENIZER_MODEL):
       
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(
                f"Parameter Golf tokenizer model not found at {model_path}. "
                f"Download {DEFAULT_REPO_ID}:{DEFAULT_VARIANT} tokenizer files first."
            )

        import sentencepiece as spm

        processor = spm.SentencePieceProcessor()
        processor.Load(str(model_path))
        return cls(model_path=model_path, processor=processor)

    @property
    def name_or_path(self) -> str:
        return str(self.model_path)

    @property
    def vocab_size(self):
        return self.processor.GetPieceSize()

    @property
    def eos_token_id(self):
        eos = self.processor.eos_id()
        return eos if eos >= 0 else None

    @property
    def bos_token_id(self):
        bos = self.processor.bos_id()
        return bos if bos >= 0 else None

    def encode(self, text: str, *, add_eos: bool = False) -> list[int]:
        ids = list(self.processor.EncodeAsIds(text))
        if add_eos:
            eos = self.eos_token_id
            if eos is None:
                raise ValueError("tokenizer has no eos_token_id")
            ids.append(eos)
        return ids

    def decode(self, ids: Sequence[int]) -> str:
        return self.processor.DecodeIds(list(ids))


def load_tokenizer(
    model_path: str | Path = DEFAULT_TOKENIZER_MODEL,
) -> TextTokenizer:
    return ParameterGolfTokenizer.from_file(model_path)


def load_hf_tokenizer(name_or_path: str, **kwargs: Any) -> TextTokenizer:
    return HuggingFaceTokenizer.from_pretrained(name_or_path, **kwargs)


def tokenizer_summary(tokenizer: TextTokenizer) -> dict[str, int | str | None]:
    return {
        "name_or_path": tokenizer.name_or_path,
        "vocab_size": tokenizer.vocab_size,
        "eos_token_id": tokenizer.eos_token_id,
        "bos_token_id": tokenizer.bos_token_id,
    }
