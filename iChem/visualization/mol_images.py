from rdkit import Chem # type: ignore
from rdkit.Chem import Draw # type: ignore
import numpy as np # type: ignore

def smiles_to_grid_image(smiles,
                         mols_per_row=5,
                         sub_img_size=(250, 250)):
    """
    Convert a list of SMILES strings to a grid image of molecules.

    Parameters:
    - smiles_list: List of SMILES strings.
    - mols_per_row: Number of molecules per row in the grid.
    - sub_img_size: Size of each sub-image (width, height).

    Returns:
    - A PIL Image object containing the grid of molecule images.
    """
    if len(smiles) > 50:
        smiles = np.random.choice(smiles, 50, replace=False)
    mols = [Chem.MolFromSmiles(smile) for smile in smiles]
    img = Draw.MolsToGridImage(mols, molsPerRow=mols_per_row, subImgSize=sub_img_size)
    return img