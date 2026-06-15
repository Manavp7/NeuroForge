"""Render a molecule to a 2D SVG string (for the frontend)."""

from __future__ import annotations

from rdkit import Chem
from rdkit.Chem.Draw import rdMolDraw2D


def molecule_to_svg(smiles: str, width: int = 340, height: int = 260) -> str:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">'
            f'<text x="10" y="20" fill="red">Invalid SMILES</text></svg>'
        )
    drawer = rdMolDraw2D.MolDraw2DSVG(width, height)
    rdMolDraw2D.PrepareAndDrawMolecule(drawer, mol)
    drawer.FinishDrawing()
    return drawer.GetDrawingText()
