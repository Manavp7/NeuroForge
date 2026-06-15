"""Optional SMILES variational autoencoder generator (torch).

A compact char-level VAE trained on the seed library (with randomized-SMILES augmentation),
exposing the same ``.design(target, ...)`` interface as the GA generator. It samples from the
latent prior, decodes candidate SMILES, keeps the valid ones, and ranks them with
:func:`~neuroforge.design.objectives.design_score`.

This is a demonstration of the deep-generative upgrade path; it is only used when
``generator_engine="vae"`` and torch is installed (otherwise the GA is used). Quality on such a
tiny corpus is modest by design — the point is the pluggable architecture.
"""

from __future__ import annotations

import numpy as np
from rdkit import Chem

from ..config import SETTINGS
from ..models import TargetProfile
from .generator import GAResult, _valid
from .library import SEED_SMILES
from .objectives import design_score

_PAD, _START, _END = "<", "^", "$"


class VAEGenerator:
    def __init__(self, seed: int | None = None, latent_dim: int = 32, epochs: int = 200):
        import torch  # raises if unavailable -> factory falls back to GA
        import torch.nn as nn

        self.torch = torch
        self.nn = nn
        self.seed = SETTINGS.default_seed if seed is None else seed
        torch.manual_seed(self.seed)

        corpus = self._augmented_corpus()
        self.charset = [_PAD, _START, _END] + sorted({c for s in corpus for c in s})
        self.stoi = {c: i for i, c in enumerate(self.charset)}
        self.itos = {i: c for c, i in self.stoi.items()}
        self.max_len = max(len(s) for s in corpus) + 2
        self.latent_dim = latent_dim
        self.vocab = len(self.charset)

        self._build(latent_dim)
        self._train(corpus, epochs)

    # ------------------------------------------------------------------ #
    def _augmented_corpus(self, n_aug: int = 20) -> list[str]:
        out: set[str] = set()
        for s in SEED_SMILES:
            mol = Chem.MolFromSmiles(s)
            if mol is None:
                continue
            out.add(Chem.MolToSmiles(mol))
            for _ in range(n_aug):
                try:
                    out.add(Chem.MolToSmiles(mol, doRandom=True, canonical=False))
                except Exception:
                    pass
        # Keep lengths reasonable for the tiny model.
        return [s for s in out if len(s) <= 60] or [
            Chem.MolToSmiles(Chem.MolFromSmiles(s)) for s in SEED_SMILES
        ]

    def _encode_str(self, s: str) -> list[int]:
        seq = [self.stoi[_START]] + [self.stoi[c] for c in s] + [self.stoi[_END]]
        seq += [self.stoi[_PAD]] * (self.max_len - len(seq))
        return seq[: self.max_len]

    def _build(self, latent_dim: int) -> None:
        nn = self.nn
        emb = 32
        hid = 128
        self.embed = nn.Embedding(self.vocab, emb)
        self.enc = nn.GRU(emb, hid, batch_first=True)
        self.to_mu = nn.Linear(hid, latent_dim)
        self.to_lv = nn.Linear(hid, latent_dim)
        self.z2h = nn.Linear(latent_dim, hid)
        self.dec = nn.GRU(emb, hid, batch_first=True)
        self.out = nn.Linear(hid, self.vocab)
        self.params = (
            list(self.embed.parameters())
            + list(self.enc.parameters())
            + list(self.to_mu.parameters())
            + list(self.to_lv.parameters())
            + list(self.z2h.parameters())
            + list(self.dec.parameters())
            + list(self.out.parameters())
        )

    def _train(self, corpus: list[str], epochs: int) -> None:
        torch = self.torch
        data = torch.tensor([self._encode_str(s) for s in corpus], dtype=torch.long)
        opt = torch.optim.Adam(self.params, lr=2e-3)
        ce = self.nn.CrossEntropyLoss(ignore_index=self.stoi[_PAD])
        for _ in range(epochs):
            opt.zero_grad()
            x = self.embed(data)
            _, h = self.enc(x)
            h = h.squeeze(0)
            mu, lv = self.to_mu(h), self.to_lv(h)
            z = mu + torch.randn_like(mu) * torch.exp(0.5 * lv)
            h0 = torch.tanh(self.z2h(z)).unsqueeze(0)
            dec_in = self.embed(data[:, :-1])
            dout, _ = self.dec(dec_in, h0)
            logits = self.out(dout)
            recon = ce(logits.reshape(-1, self.vocab), data[:, 1:].reshape(-1))
            kld = -0.5 * torch.mean(1 + lv - mu.pow(2) - lv.exp())
            (recon + 0.05 * kld).backward()
            opt.step()

    # ------------------------------------------------------------------ #
    def _sample_one(self, temperature: float = 0.9) -> str:
        torch = self.torch
        with torch.no_grad():
            z = torch.randn(1, self.latent_dim)
            h = torch.tanh(self.z2h(z)).unsqueeze(0)
            tok = torch.tensor([[self.stoi[_START]]], dtype=torch.long)
            chars: list[str] = []
            for _ in range(self.max_len):
                x = self.embed(tok)
                dout, h = self.dec(x, h)
                logits = self.out(dout[:, -1, :]) / temperature
                probs = torch.softmax(logits, dim=-1).squeeze(0).numpy()
                idx = int(np.random.choice(len(probs), p=probs / probs.sum()))
                ch = self.itos[idx]
                if ch in (_END, _PAD):
                    break
                chars.append(ch)
                tok = torch.tensor([[idx]], dtype=torch.long)
        return "".join(chars)

    def design(
        self,
        target: TargetProfile,
        population: int | None = None,
        generations: int | None = None,
        top_k: int | None = None,
        constraints=None,
    ) -> list[GAResult]:
        n_samples = (population or SETTINGS.ga_population) * (generations or 4)
        k = top_k or SETTINGS.ga_top_k
        seen: dict[str, float] = {}
        for _ in range(n_samples):
            smi = self._sample_one()
            mol = _valid(Chem.MolFromSmiles(smi)) if smi else None
            if mol is None:
                continue
            canon = Chem.MolToSmiles(mol)
            if canon not in seen:
                seen[canon] = design_score(mol, target, constraints)
        if not seen:
            # Decoder produced nothing valid -> let the caller fall back.
            from .generator import MoleculeGenerator

            return MoleculeGenerator(seed=self.seed).design(target, population, generations, top_k)
        ranked = sorted(seen.items(), key=lambda kv: kv[1], reverse=True)[:k]
        return [GAResult(smiles=s, score=sc, provenance={"engine": "vae"}) for s, sc in ranked]
