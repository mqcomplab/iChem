from rdkit import Chem # type: ignore
from rdkit.Chem import Draw # type: ignore
from rdkit.Chem import rdFMCS # type: ignore
from ..utils.utils import smiles_standarization 
import numpy as np # type: ignore
from collections import defaultdict, deque

def smiles_to_grid_image(smiles,
                         mols_per_row=5,
                         sub_img_size=(250, 250),
                         legends=None,
                         standarize=True,
                         MCS=False,
                         max_items=50):
    """
    Convert a list of SMILES strings to a grid image of molecules.

    Parameters:
    - smiles_list: List of SMILES strings.
    - mols_per_row: Number of molecules per row in the grid.
    - sub_img_size: Size of each sub-image (width, height).
    - legends: Optional list of legends for each molecule.
    - standarize: Boolean indicating whether to standardize molecules.
    - MCS: Boolean indicating whether to highlight the Maximum Common Substructure (MCS).
    Returns:
    - A PIL Image object containing the grid of molecule images.
    """
    if legends is None:
        if len(smiles) > max_items:
            idx = np.random.choice(len(smiles), max_items, replace=False).tolist()
        else:
            idx = list(range(len(smiles)))
    else:
        idx = _balanced_sample_indices(legends, max_items=max_items)

    smiles = [smiles[i] for i in idx]
    if legends is not None:
        legends = [legends[i] for i in idx]

    mols = [Chem.MolFromSmiles(s) for s in smiles]
    
    if standarize:
        mols = [smiles_standarization(mol) for mol in mols]
    if MCS:
        return _mols_to_grid_MCS(mols,
                                    mols_per_row,
                                    sub_img_size,
                                    legends)
    if legends is not None:
        img = Draw.MolsToGridImage(mols,
                                   molsPerRow=mols_per_row,
                                   subImgSize=sub_img_size,
                                   legends=legends)
    else:
        img = Draw.MolsToGridImage(mols,
                                   molsPerRow=mols_per_row,
                                   subImgSize=sub_img_size)
    
    return img

def _mols_to_grid_MCS(mols,
                      mols_per_row=5,
                      sub_img_size=(250, 250),
                      legends=None):
    MCS = rdFMCS.FindMCS(mols, threshold=0.75)
    MCS_mol = Chem.MolFromSmarts(MCS.smartsString)
    for mol in mols:
        if mol.HasSubstructMatch(MCS_mol):
            match = mol.GetSubstructMatch(MCS_mol)
            atom_indices = list(match)
            highlight_atoms = atom_indices
            # Highlight the matching substructure
            mol.SetProp('_highlightAtoms', ','.join(map(str, highlight_atoms)))
    highlight_lists = []
    for mol in mols:
        if mol.HasProp('_highlightAtoms'):
            vals = mol.GetProp('_highlightAtoms').split(',')
            highlight_lists.append(list(map(int, vals)))
        else:
            highlight_lists.append([])
    if legends is not None:
        img = Draw.MolsToGridImage(mols,
                                   highlightAtomLists=highlight_lists,
                                   molsPerRow=mols_per_row,
                                   subImgSize=sub_img_size,
                                   legends=legends)
    else:
        img = Draw.MolsToGridImage(mols,
                                   highlightAtomLists=highlight_lists,
                                   molsPerRow=mols_per_row,
                                   subImgSize=sub_img_size)
    return img


def MCS_image(smiles,
              n_samples=50,
              MCS_threshold=0.75,
              standarize=True):
    if len(smiles) > n_samples:
        smiles = np.random.choice(smiles, n_samples, replace=False)
    mols = [Chem.MolFromSmiles(smile) for smile in smiles]
    if standarize:
        mols = [smiles_standarization(mol) for mol in mols]
    
    MCS = rdFMCS.FindMCS(mols, threshold=MCS_threshold)
    MCS_mol = Chem.MolFromSmarts(MCS.smartsString)
    return Draw.MolToImage(MCS_mol,
                           size=(350, 350),
                           useSVG=True)


def _balanced_sample_indices(legends, max_items=50):
    """
    Select indices in a balanced round-robin fashion across legend classes.

    Parameters
    ----------
    legends : list
        Legend/class label for each molecule.
    max_items : int
        Maximum number of indices to return.

    Returns
    -------
    list
        Selected indices.

    Notes
    -----
    Molecules are sampled by alternating between legend groups. This ensures
    that less frequent classes are represented early in the selection.
    """

    if legends is None:
        return []

    # Group indices by legend
    groups = defaultdict(deque)
    for idx, legend in enumerate(legends):
        groups[legend].append(idx)

    if len(groups) <= 1 or all(len(indices) == 1 for indices in groups.values()):
        if len(legends) > max_items:
            return np.random.choice(len(legends), max_items, replace=False).tolist()
        return list(range(len(legends)))

    ordered_legends = list(groups.keys())

    selected = []

    while len(selected) < max_items:
        added = False

        for legend in ordered_legends:
            if groups[legend]:
                selected.append(groups[legend].popleft())
                added = True

                if len(selected) == max_items:
                    break

        if not added:
            break

    return selected