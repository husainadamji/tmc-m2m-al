"""
Mol2 utilities for metal-containing structures (requires molSimplify).

Used to obtain ligand-only mol2 strings from mol2 with a dummy metal, e.g. for
RDKit or SCScore. Optional dependency: molSimplify.
"""


def _get_mol3d():
    try:
        from molSimplify.Classes.mol3D import mol3D
        return mol3D
    except ImportError as e:
        raise ImportError(
            "mol2_utils requires molSimplify. Install it or call this code in an "
            "environment where molSimplify is available."
        ) from e


def mol2_ligand_from_mol2_metal(mol2_w_metal: str) -> str:
    """
    Remove dummy metal from a mol2 string and return the ligand mol2 string.

    Assumes the metal is represented as symbol "X" (molSimplify dummy). The
    returned string is the mol2 block of the ligand only, suitable for RDKit
    or SCScore.

    Parameters
    ----------
    mol2_w_metal : str
        Full mol2 file content (metal + ligand).

    Returns
    -------
    str
        Mol2 content of the ligand only.
    """
    mol3d = _get_mol3d()()
    mol3d.readfrommol2(mol2_w_metal, readstring=True)
    metal_idx = mol3d.findAtomsbySymbol("X")[0]
    mol3d.deleteatom(metal_idx)
    return mol3d.writemol2("tmp", writestring=True)
