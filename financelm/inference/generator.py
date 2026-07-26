from pathlib import Path

import torch
from tokenizers import Tokenizer

from financelm.model.model import FinanceLM
from financelm.inference.sampling import (
    Sampler,
    SamplingStrategy,
)


class Generator:
    """
    Autoregressive text generator for FinanceLM.
    """

    def __init__(
        self,
        model: FinanceLM,
        tokenizer_path: Path | str,
        device: torch.device,
        max_seq_length: int = 256,
    ):
        self.model = model
        self.device = device
        self.max_seq_length = max_seq_length

        self.tokenizer = Tokenizer.from_file(
            str(tokenizer_path)
        )

        self.sampler = Sampler()

        self.model.eval()

        self.bos_token = "[BOS]"
        self.eos_token = "[EOS]"

        self.bos_id = self.tokenizer.token_to_id(self.bos_token)
        self.eos_id = self.tokenizer.token_to_id(self.eos_token)

    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 50,
        strategy: SamplingStrategy = SamplingStrategy.TOP_P,
        temperature: float = 1.0,
        top_k: int = 40,
        top_p: float = 0.9,
    ) -> str:

        if not prompt.strip():
            raise ValueError("Prompt cannot be empty.")

        encoding = self.tokenizer.encode(prompt)
        input_ids = encoding.ids

        if self.bos_id is not None:
            input_ids = [self.bos_id] + input_ids

        # Truncate prompt to fit within context window
        input_ids = input_ids[-self.max_seq_length:]

        input_ids = torch.tensor(
            input_ids,
            dtype=torch.long,
            device=self.device,
        ).unsqueeze(0)

        for _ in range(max_new_tokens):

            # Always feed at most max_seq_length tokens
            context = input_ids[:, -self.max_seq_length:]

            logits = self.model(context)

            next_logits = logits[:, -1, :]

            next_token = self.sampler.sample(
                logits=next_logits,
                strategy=strategy,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
            )

            input_ids = torch.cat(
                [input_ids, next_token],
                dim=1,
            )

            if (
                self.eos_id is not None
                and next_token.item() == self.eos_id
            ):
                break

        generated_ids = input_ids[0].detach().cpu().tolist()

        generated_text = self.tokenizer.decode(
            generated_ids,
            skip_special_tokens=True,
        )

        return generated_text
