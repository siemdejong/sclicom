"""Provide Clinical Context Attention MIL."""
from functools import cache, lru_cache
from typing import Callable, Literal, Union

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

        inputs = self.tokenizer(sentence, padding=True, return_tensors="pt").to(
            self.device
        )

        # Dis-/enable requires_grad and dis-/enable dropout etc. on layers depending
        # on self.trainable_llm.
        output: LLMOutput
        self.llm.train(self.trainable_llm)
        if self.trainable_llm:
            output = self.llm(**inputs)
        else:
            with torch.no_grad():
                output = self.llm(**inputs)

        # 0 holds the [CLS] token, which attempts to classify the sentence.
        features = output.last_hidden_state[:, 0, :]

        return features

    @cache
    def _calculate_llm_hidden_dim_size(self) -> int:
        """Calculate the LLM hidden dimension size."""
        example = "test"
        inputs = self.tokenizer(example, padding=True, return_tensors="pt").to(
            self.device
        )
        outputs: LLMOutput = self.llm(**inputs)
        features = outputs.last_hidden_state[:, 0, :]
        return features.numel()

    def __init__(
        self,
        llm_model: Literal[
            "emilyalsentzer/Bio_ClinicalBERT", "nlpie/tiny-clinicalbert"
        ] = "nlpie/tiny-clinicalbert",
        llm_tokenizer: Union[str, None] = None,
        trainable_llm: bool = False,
        text_embedding_cache_maxsize: Union[int, None] = None,
        in_features: int = 1024,
        *args,
        **kwargs,
    ) -> None:
        """Overwrite classifier attribute of `VarAttention`.

        `VarAttention` only concatenates mean and variance poolings,
        while `CCMIL` accepts a clinical input sentence and concatenates this to the
        vector right before the classifier.

        Parameters
        ----------
        llm_model : str, default=nlpie/tiny-clinicalbert
            Large language model from HuggingFace to use.
        llm_tokenizer : str, default=`llm_model`
            Tokenizer from HuggingFace to use. Defaults to 'llm_model' corresponding
            tokenizer.
        trainable_llm : bool, default=False
            If the large language model should be trainable, do not cache the output.
        text_embedding_cache_maxsize : int, optional, default=None
            Maximum size of the text embedding cache. Only applicable if
            `trainable_llm=True`.
        See `Attention`.
        """
        super(CCMIL, self).__init__(in_features=in_features, *args, **kwargs)

        self.example_input_array = (torch.randn((1, 1000, in_features)), "test")

        self.llm: nn.Module = AutoModel.from_pretrained(llm_model)

        if llm_tokenizer is None:
            llm_tokenizer = llm_model
        self.tokenizer = AutoTokenizer.from_pretrained(llm_model)

        self.classifier = nn.Sequential(
            nn.Linear(
                self._calculate_llm_hidden_dim_size() + 2 * self.L * self.K,
                self.num_classes,
            )  # + llm_hidden_dim, because that is what BERT outputs.
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
        # [0] to convert list to string,
        # because all tiles in a bag have the same clinical context.
        x, y, cc = batch["data"], batch["target"], batch["cc"][0]
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
