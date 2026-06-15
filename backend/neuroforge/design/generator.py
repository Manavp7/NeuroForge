"""Genetic-algorithm de novo molecule generator over RDKit molecules.

This is a *surrogate* for deep generative models (diffusion/VAE). It evolves a population of
valid molecules toward a :class:`~neuroforge.models.TargetProfile` using mutation + fragment
crossover, scored by :func:`~neuroforge.design.objectives.design_score`. Deterministic given a seed.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from rdkit import Chem

from ..config import SETTINGS
from ..models import TargetProfile
from .library import SEED_SMILES
from .objectives import design_score

_ELEMENTS = [6, 7, 8, 9, 16]  # C, N, O, F, S
_MIN_HEAVY, _MAX_HEAVY = 6, 55


@dataclass
class GAResult:
    """A generated molecule with its design score and provenance."""

    smiles: str
    score: float
    provenance: dict = field(default_factory=dict)


def _valid(mol: Chem.Mol | None) -> Chem.Mol | None:
    if mol is None:
        return None
    try:
        Chem.SanitizeMol(mol)
    except Exception:
        return None
    n = mol.GetNumHeavyAtoms()
    if n < _MIN_HEAVY or n > _MAX_HEAVY:
        return None
    return mol


class MoleculeGenerator:
    def __init__(self, seed: int | None = None):
        self.seed = SETTINGS.default_seed if seed is None else seed

    # ------------------------------------------------------------------ #
    def design(
        self,
        target: TargetProfile,
        population: int | None = None,
        generations: int | None = None,
        top_k: int | None = None,
    ) -> list[GAResult]:
        rng = np.random.default_rng(self.seed)
        pop_size = population or SETTINGS.ga_population
        n_gen = generations or SETTINGS.ga_generations
        k = top_k or SETTINGS.ga_top_k

        pop = self._init_population(pop_size, rng)
        cache: dict[str, float] = {}
        best_history: list[float] = []

        def fitness(mol: Chem.Mol) -> float:
            smi = Chem.MolToSmiles(mol)
            if smi not in cache:
                cache[smi] = design_score(mol, target)
            return cache[smi]

        for gen in range(n_gen):
            scored = sorted(pop, key=fitness, reverse=True)
            best_history.append(fitness(scored[0]))
            n_elite = max(2, pop_size // 5)
            elites = scored[:n_elite]
            children: list[Chem.Mol] = list(elites)
            while len(children) < pop_size:
                p1 = self._tournament(scored, fitness, rng)
                p2 = self._tournament(scored, fitness, rng)
                child = self._crossover(p1, p2, rng) or p1
                if rng.random() < SETTINGS.ga_mutation_rate:
                    child = self._mutate(child, rng) or child
                children.append(child)
            pop = children

        # Final ranking, deduplicated by canonical SMILES.
        final = sorted(pop, key=fitness, reverse=True)
        seen: set[str] = set()
        results: list[GAResult] = []
        for mol in final:
            smi = Chem.MolToSmiles(mol)
            if smi in seen:
                continue
            seen.add(smi)
            results.append(
                GAResult(
                    smiles=smi,
                    score=fitness(mol),
                    provenance={"generations": n_gen, "population": pop_size},
                )
            )
            if len(results) >= k:
                break
        return results

    # ------------------------------------------------------------------ #
    def _init_population(self, size: int, rng: np.random.Generator) -> list[Chem.Mol]:
        seeds = [m for m in (Chem.MolFromSmiles(s) for s in SEED_SMILES) if _valid(m)]
        pop: list[Chem.Mol] = []
        i = 0
        while len(pop) < size:
            base = seeds[i % len(seeds)]
            mol = base if i < len(seeds) else (self._mutate(base, rng) or base)
            pop.append(mol)
            i += 1
        return pop

    @staticmethod
    def _tournament(scored, fitness, rng: np.random.Generator, t: int = 3) -> Chem.Mol:
        idx = rng.integers(0, len(scored), size=t)
        contenders = [scored[i] for i in idx]
        return max(contenders, key=fitness)

    # ------------------------------------------------------------------ #
    def _mutate(self, mol: Chem.Mol, rng: np.random.Generator) -> Chem.Mol | None:
        op = rng.integers(0, 4)
        rw = Chem.RWMol(mol)
        try:
            if op == 0:  # change an atom's element
                a = int(rng.integers(0, rw.GetNumAtoms()))
                atom = rw.GetAtomWithIdx(a)
                if not atom.GetIsAromatic():
                    atom.SetAtomicNum(int(rng.choice(_ELEMENTS)))
            elif op == 1:  # append a carbon to a random atom
                a = int(rng.integers(0, rw.GetNumAtoms()))
                new_idx = rw.AddAtom(Chem.Atom(6))
                rw.AddBond(a, new_idx, Chem.BondType.SINGLE)
            elif op == 2:  # remove a terminal atom
                terminals = [
                    at.GetIdx() for at in rw.GetAtoms() if at.GetDegree() == 1 and not at.GetIsAromatic()
                ]
                if terminals:
                    rw.RemoveAtom(int(rng.choice(terminals)))
            else:  # toggle a non-aromatic bond order
                bonds = [b for b in rw.GetBonds() if not b.GetIsAromatic()]
                if bonds:
                    b = bonds[int(rng.integers(0, len(bonds)))]
                    b.SetBondType(
                        Chem.BondType.DOUBLE
                        if b.GetBondType() == Chem.BondType.SINGLE
                        else Chem.BondType.SINGLE
                    )
            return _valid(rw.GetMol())
        except Exception:
            return None

    def _crossover(self, m1: Chem.Mol, m2: Chem.Mol, rng: np.random.Generator) -> Chem.Mol | None:
        f1 = self._break_random_bond(m1, rng)
        f2 = self._break_random_bond(m2, rng)
        if not f1 or not f2:
            return None
        a = f1[int(rng.integers(0, len(f1)))]
        b = f2[int(rng.integers(0, len(f2)))]
        return self._join_fragments(a, b)

    @staticmethod
    def _break_random_bond(mol: Chem.Mol, rng: np.random.Generator) -> list[Chem.Mol] | None:
        bonds = [
            b.GetIdx()
            for b in mol.GetBonds()
            if b.GetBondType() == Chem.BondType.SINGLE and not b.IsInRing()
        ]
        if not bonds:
            return None
        bidx = int(rng.choice(bonds))
        try:
            frag = Chem.FragmentOnBonds(mol, [bidx], addDummies=True)
            pieces = Chem.GetMolFrags(frag, asMols=True, sanitizeFrags=False)
            return [p for p in pieces if p.GetNumHeavyAtoms() >= 2]
        except Exception:
            return None

    @staticmethod
    def _join_fragments(a: Chem.Mol, b: Chem.Mol) -> Chem.Mol | None:
        try:
            combined = Chem.RWMol(Chem.CombineMols(a, b))
            dummies = [at.GetIdx() for at in combined.GetAtoms() if at.GetAtomicNum() == 0]
            if len(dummies) < 2:
                return None
            dA, dB = dummies[0], dummies[1]
            nA = combined.GetAtomWithIdx(dA).GetNeighbors()[0].GetIdx()
            nB = combined.GetAtomWithIdx(dB).GetNeighbors()[0].GetIdx()
            combined.AddBond(nA, nB, Chem.BondType.SINGLE)
            for idx in sorted([dA, dB], reverse=True):
                combined.RemoveAtom(idx)
            return _valid(combined.GetMol())
        except Exception:
            return None
