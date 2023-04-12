"""Provide Clinical Context Attention MIL."""
from functools import cache, lru_cache
from typing import Callable, Union

import torch
import torch.nn.functional as F
from torch import nn
from transformers import AutoModel, AutoTokenizer

from dpat.mil.models.varmil import VarAttention
from dpat.types import LLMOutput


def preprocess_text(sentence: str) -> str:
    """Preprocess text."""
    sentence = sentence.replace("+", " and ")
    return sentence


class CCMIL(VarAttention):
    """Compute the Clinical Context (CC) attention MIL."""

    def _compute_cc_embedding(self, sentence: str) -> torch.Tensor:
        """Calculate the Clinical Context (CC) embedding.

        Embeddings are cached without maximum size.
        Make sure the embeddings of different sentences presented fit on the device.

        sentence : str
            Sentence to calculate embedding of.
        """
        sentence = preprocess_text(sentence)

        input = self.tokenizer(sentence, padding=True, return_tensors="pt")

        # Dis-/enable requires_grad and dis-/enable dropout etc. on layers depending
        # on self.trainable_llm.
        output: LLMOutput
        self.llm.train(self.trainable_llm)
        if self.trainable_llm:
            output = self.llm(**input)
        else:
            with torch.no_grad():
                output = self.llm(**input)

        # 0 holds the [CLS] token, which attempts to classify the sentence.
        features = output.last_hidden_state[:, 0, :]

        return features

    def __init__(
        self,
        trainable_llm: bool = False,
        text_embedding_cache_maxsize: Union[int, None] = None,
        *args,
        **kwargs,
    ) -> None:
        """Overwrite classifier attribute of `VarAttention`.

        `VarAttention` only concatenates mean and variance poolings,
        while `CCMIL` accepts a clinical input sentence and concatenates this to the
        vector right before the classifier.

        Parameters
        ----------
        trainable_llm : bool, default=False
            If the large language model should be trainable, do not cache the output.
        text_embedding_cache_maxsize : int, optional, default=None
            Maximum size of the text embedding cache. Only applicable if
            `trainable_llm=True`.
        See `Attention`.
        """
        super(VarAttention, self).__init__(*args, **kwargs)

        self.tokenizer = AutoTokenizer.from_pretrained(
            "emilyalsentzer/Bio_ClinicalBERT"
        )
        self.llm: nn.Module = AutoModel.from_pretrained(
            "emilyalsentzer/Bio_ClinicalBERT"
        )
        self.classifier = nn.Sequential(
            nn.Linear(
                768 + 2 * self.L * self.K, self.num_classes
            )  # +768, because that is what BERT outputs.
        )

        self.trainable_llm = trainable_llm
        self.compute_cc_embedding: Callable[[str], torch.Tensor]
        if trainable_llm:
            self.compute_cc_embedding = self._compute_cc_embedding
        else:
            if text_embedding_cache_maxsize is not None:
                # Cache the text embeddings if the LLM is not trained.
                cache_fn = lru_cache(text_embedding_cache_maxsize)
            else:
                # Use the faster cache function.
                cache_fn = cache
            self.compute_cc_embedding = cache_fn(self._compute_cc_embedding)

    def _common_step(self, batch):
        """Perform a common step that is used in train/val/test.

        Overwrites the parent _common_step. The clinical context is
        passed to the forward function.
        """
        x, y, cc = batch["data"], batch["target"], batch["cc"]
        y_hat, A = self(x, cc)
        loss = self.loss_fn(y_hat, y)
        return loss, y_hat, y, A

    def forward(  # type: ignore[override] # TODO make this strongly typed.
        self, x: torch.Tensor, sentence: str
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Calculate prediction and attention.

        Parameters
        ----------
        x : `torch.Tensor`
            Batch of size 1 with N feature vectors.
        sentence : str
            A sentence to calculate embeddings for.
        """
        # Since we have batch_size = 1,
        # squeezes from (1,num,features) to (num, features)
        H = x.squeeze(0)

        A = self.attention(H)  # NxK
        A = torch.transpose(A, 1, 0)  # KxN
        A = F.softmax(A, dim=1)  # softmax over N

        M = torch.mm(A, H)  # KxL

        S = self.compute_weighted_var(A, H)

        CC = self.compute_cc_embedding(sentence)

        # Concatenate the two tensors among the feature dimension,
        # giving a twice as big feature.
        MSCC = torch.cat((M, S, CC), dim=1)

        Y_hat = self.classifier(MSCC)
        return Y_hat, A
