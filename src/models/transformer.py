"""Fine-tuned transformer baseline.

A DistilBERT encoder with a six-unit sigmoid head trained under binary
cross-entropy -- the standard formulation for multi-label text classification,
and the approach that tops the Jigsaw leaderboard.  Unlike the linear models it
sees word order and sub-word context, which is what lets it separate "this
article is garbage" from "you are garbage".

The module is imported lazily by :mod:`src.models.registry` so that PyTorch
stays an optional dependency:

    pip install -r requirements-dl.txt --index-url https://download.pytorch.org/whl/cpu
"""

from __future__ import annotations

import logging
from collections.abc import Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.models.base import BaseTextClassifier

LOGGER = logging.getLogger(__name__)


class TransformerClassifier(BaseTextClassifier):
    """Fine-tune a pre-trained encoder for multi-label toxicity detection.

    Args:
        model_name: any Hugging Face sequence-classification checkpoint.
        max_length: tokens kept per comment.  The corpus is long-tailed; 192
            covers roughly 95% of the comments in full and truncation beyond
            that costs little.
        batch_size: training and inference batch size.
        epochs: passes over the training fold.  Two is enough; more overfits the
            frequent labels without helping the rare ones.
        learning_rate: AdamW learning rate.
        weight_decay: AdamW weight decay.
        device: ``"cuda"``, ``"cpu"``, or ``None`` to pick automatically.
        amp: train under automatic mixed precision.  ``None`` enables it
            whenever the run is on CUDA, where it is roughly 1.5x faster and
            uses less memory than fp32; it is always off on CPU, which has no
            fp16 path worth taking.
        seed: seed for the classification head's initialisation.
    """

    name = "distilbert"

    def __init__(
        self,
        model_name: str = "distilbert-base-uncased",
        max_length: int = 192,
        batch_size: int = 32,
        epochs: int = 2,
        learning_rate: float = 3e-5,
        weight_decay: float = 0.01,
        device: str | None = None,
        amp: bool | None = None,
        seed: int = 42,
        label_names: Sequence[str] | None = None,
    ) -> None:
        super().__init__(label_names=label_names)
        self.model_name = model_name
        self.max_length = max_length
        self.batch_size = batch_size
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.amp = self.device.startswith("cuda") if amp is None else amp
        self.seed = seed
        self.tokenizer_ = None
        self.model_ = None

    def _encode(self, texts: Sequence[str]) -> tuple[torch.Tensor, torch.Tensor]:
        """Tokenise ``texts`` into padded input ids and attention masks."""
        encoded = self.tokenizer_(
            list(texts),
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        return encoded["input_ids"], encoded["attention_mask"]

    def fit(self, texts: Sequence[str], y: np.ndarray) -> TransformerClassifier:
        """Fine-tune the encoder together with a fresh six-unit sigmoid head."""
        y = np.asarray(y, dtype=np.float32)
        torch.manual_seed(self.seed)

        self.tokenizer_ = AutoTokenizer.from_pretrained(self.model_name)
        self.model_ = AutoModelForSequenceClassification.from_pretrained(
            self.model_name,
            num_labels=y.shape[1],
            # Tells the head to apply a sigmoid per label instead of a softmax
            # over labels: the six categories are not mutually exclusive.
            problem_type="multi_label_classification",
            # A checkpoint that already carries a head (any fine-tuned model)
            # has the wrong output width for six labels; discard that head and
            # start from a freshly initialised one rather than failing to load.
            ignore_mismatched_sizes=True,
        ).to(self.device)

        input_ids, attention_mask = self._encode(texts)
        loader = DataLoader(
            TensorDataset(input_ids, attention_mask, torch.from_numpy(y)),
            batch_size=self.batch_size,
            shuffle=True,
        )
        optimiser = torch.optim.AdamW(
            self.model_.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        # fp16 gradients underflow without loss scaling; the scaler is a no-op
        # when disabled, so the fp32 path below stays exactly as it was.
        scaler = torch.amp.GradScaler(self.device, enabled=self.amp)

        self.model_.train()
        for epoch in range(1, self.epochs + 1):
            running_loss = 0.0
            for batch_ids, batch_mask, batch_labels in loader:
                optimiser.zero_grad()
                with torch.amp.autocast(
                    self.device, dtype=torch.float16, enabled=self.amp
                ):
                    outputs = self.model_(
                        input_ids=batch_ids.to(self.device),
                        attention_mask=batch_mask.to(self.device),
                        labels=batch_labels.to(self.device),
                    )
                scaler.scale(outputs.loss).backward()
                # Clipping reads the true gradients, so undo the scaling first.
                # Long comments occasionally produce large gradients early on.
                scaler.unscale_(optimiser)
                torch.nn.utils.clip_grad_norm_(self.model_.parameters(), 1.0)
                scaler.step(optimiser)
                scaler.update()
                running_loss += outputs.loss.detach().item()
            LOGGER.info(
                "epoch %d/%d  mean BCE loss %.4f",
                epoch,
                self.epochs,
                running_loss / max(1, len(loader)),
            )
        return self

    @torch.no_grad()
    def predict_proba(self, texts: Sequence[str]) -> np.ndarray:
        """Return the sigmoid of the logits, batch by batch."""
        if self.model_ is None:
            raise RuntimeError("Call fit() before predict_proba().")

        self.model_.eval()
        input_ids, attention_mask = self._encode(texts)
        loader = DataLoader(
            TensorDataset(input_ids, attention_mask), batch_size=self.batch_size
        )

        chunks = []
        for batch_ids, batch_mask in loader:
            with torch.amp.autocast(
                self.device, dtype=torch.float16, enabled=self.amp
            ):
                logits = self.model_(
                    input_ids=batch_ids.to(self.device),
                    attention_mask=batch_mask.to(self.device),
                ).logits
            # Back to fp32 before the sigmoid: probabilities are compared
            # against tuned thresholds, and fp16 resolution near 0 and 1 is
            # coarse enough to matter there.
            chunks.append(torch.sigmoid(logits.float()).cpu().numpy())
        return np.vstack(chunks)

    def get_params(self) -> dict[str, object]:
        """Return the hyper-parameters recorded alongside the metrics."""
        return {
            "name": self.name,
            "model_name": self.model_name,
            "max_length": self.max_length,
            "batch_size": self.batch_size,
            "epochs": self.epochs,
            "learning_rate": self.learning_rate,
            "device": self.device,
            # Recorded because it changes the arithmetic, not just the speed:
            # a run reproduced without it will not match to the last digit.
            "amp": self.amp,
        }
