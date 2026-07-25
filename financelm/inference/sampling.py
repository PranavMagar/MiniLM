from enum import Enum

import torch
import torch.nn.functional as F


class SamplingStrategy(Enum):
    GREEDY = "greedy"
    TOP_K = "top_k"
    TOP_P = "top_p"


class Sampler:
    """
    Token sampler for autoregressive language models.

    Input:
        logits: (batch_size, vocab_size)

    Output:
        next_token: (batch_size, 1)
    """

    @staticmethod
    def _apply_temperature(
        logits: torch.Tensor,
        temperature: float,
    ) -> torch.Tensor:
        """
        Apply temperature scaling.

        Lower temperature -> more confident predictions.
        Higher temperature -> more random predictions.
        """

        if temperature <= 0:
            raise ValueError(
                "temperature must be greater than 0."
            )

        return logits / temperature

    @staticmethod
    def _greedy(
        logits: torch.Tensor,
    ) -> torch.Tensor:
        """
        Greedy decoding.

        Always selects the token with the highest logit.
        """

        next_token = torch.argmax(
            logits,
            dim=-1,
            keepdim=True,
        )

        return next_token

    @staticmethod
    def _top_k(
        logits: torch.Tensor,
        top_k: int,
    ) -> torch.Tensor:
        """
        Top-K sampling.

        Samples only from the K most likely tokens.
        """

        vocab_size = logits.size(-1)

        if top_k <= 0:
            raise ValueError(
                "top_k must be greater than 0."
            )

        top_k = min(top_k, vocab_size)

        values, indices = torch.topk(
            logits,
            k=top_k,
            dim=-1,
        )

        probabilities = F.softmax(
            values,
            dim=-1,
        )

        sampled_index = torch.multinomial(
            probabilities,
            num_samples=1,
        )

        next_token = torch.gather(
            indices,
            dim=-1,
            index=sampled_index,
        )

        return next_token

    @staticmethod
    def _top_p(
        logits: torch.Tensor,
        top_p: float,
    ) -> torch.Tensor:
        """
        Nucleus (Top-P) sampling.
        """

        if not (0.0 < top_p <= 1.0):
            raise ValueError(
                "top_p must be in the range (0, 1]."
            )

        sorted_logits, sorted_indices = torch.sort(
            logits,
            descending=True,
            dim=-1,
        )

        sorted_probabilities = F.softmax(
            sorted_logits,
            dim=-1,
        )

        cumulative_probabilities = torch.cumsum(
            sorted_probabilities,
            dim=-1,
        )

        remove_mask = cumulative_probabilities > top_p

        remove_mask[..., 1:] = remove_mask[..., :-1].clone()
        remove_mask[..., 0] = False

        sorted_logits = sorted_logits.masked_fill(
            remove_mask,
            float("-inf"),
        )

        filtered_probabilities = F.softmax(
            sorted_logits,
            dim=-1,
        )

        sampled_index = torch.multinomial(
            filtered_probabilities,
            num_samples=1,
        )

        next_token = torch.gather(
            sorted_indices,
            dim=-1,
            index=sampled_index,
        )

        return next_token

    def sample(
        self,
        logits: torch.Tensor,
        strategy: SamplingStrategy = SamplingStrategy.TOP_P,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 0.9,
    ) -> torch.Tensor:
        """
        Sample the next token.

        Parameters
        ----------
        logits : Tensor
            Shape (batch_size, vocab_size)

        Returns
        -------
        Tensor
            Shape (batch_size, 1)
        """

        if logits.dim() != 2:
            raise ValueError(
                "Expected logits with shape "
                "(batch_size, vocab_size)."
            )

        logits = self._apply_temperature(
            logits,
            temperature,
        )

        if strategy == SamplingStrategy.GREEDY:
            return self._greedy(logits)

        if strategy == SamplingStrategy.TOP_K:
            return self._top_k(
                logits,
                top_k,
            )

        if strategy == SamplingStrategy.TOP_P:
            return self._top_p(
                logits,
                top_p,
            )

        raise ValueError(
            f"Unsupported sampling strategy: {strategy}"
        )