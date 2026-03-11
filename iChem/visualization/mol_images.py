from rdkit import Chem # type: ignore
from rdkit.Chem import Draw # type: ignore
from rdkit.Chem import rdFMCS, SaltRemover # type: ignore
from ..utils import smiles_standarization 
import numpy as np # type: ignore

def smiles_to_grid_image(smiles,
                         mols_per_row=5,
                         sub_img_size=(250, 250),
                         legends=None,
                         standarize=True):
    """
    Convert a list of SMILES strings to a grid image of molecules.

    Parameters:
    - smiles_list: List of SMILES strings.
    - mols_per_row: Number of molecules per row in the grid.
    - sub_img_size: Size of each sub-image (width, height).
    - legends: Optional list of legends for each molecule.
    Returns:
    - A PIL Image object containing the grid of molecule images.
    """
    if len(smiles) > 50:
        smiles = np.random.choice(smiles, 50, replace=False)
    mols = [Chem.MolFromSmiles(smile) for smile in smiles]
    if standarize:
        mols = [smiles_standarization(mol) for mol in mols]
    if legends is not None:
        img = Draw.MolsToGridImage(mols, molsPerRow=mols_per_row, subImgSize=sub_img_size, legends=legends)
    else:
        img = Draw.MolsToGridImage(mols, molsPerRow=mols_per_row, subImgSize=sub_img_size)
    
    return img


def sampled_MSC(cluster,
                smiles,
                n_samples=100,
                MCS_threshold=0.75,
                return_samples=False):
    """Calculate the MSC for a sampled subset of molecules in a cluster."""
    n_samples = min(n_samples, len(cluster))

    # Get the SMILES for the sampled molecules
    remover = SaltRemover.SaltRemover()
    np.random.shuffle(cluster)
    sampled_smiles = [smiles[i] for i in cluster[:n_samples]]
    mols = [Chem.MolFromSmiles(smi) for smi in sampled_smiles]
    mols = [remover.StripMol(mol) for mol in mols]

    # Find the MSC of those molecules
    MCS = rdFMCS.FindMCS(mols, threshold=MCS_threshold)
    MCS_mol = Chem.MolFromSmarts(MCS.smartsString)

    if return_samples:
        return mols, MCS_mol
    

def sampled_MSC_image(cluster,
                        smiles,
                        n_samples=100,
                        MCS_threshold=0.75,
                        return_samples=False):
    
    if len(cluster) == 1:
        mol = Chem.MolFromSmiles(smiles[cluster[0]])
        return Draw.MolToImage(mol)

    if not return_samples:
        MCS_mol = sampled_MSC(cluster, smiles,
                            n_samples=n_samples,
                            MCS_threshold=MCS_threshold,
                            return_samples=False)
        return Draw.MolToImage(MCS_mol)
    
    mols, MCS_mol = sampled_MSC(cluster, smiles,
                                n_samples=n_samples,
                                MCS_threshold=MCS_threshold,
                                return_samples=True)

    # Generate the image of the MCS
    img1 = Draw.MolToImage(MCS_mol)

    # Find substructure matches in the sampled molecules with the MCS
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

    img2 = Draw.MolsToGridImage(mols,
                                highlightAtomLists=highlight_lists,
                                molsPerRow=5,
                                subImgSize=(250, 250))
    
    return img1, img2


def clusters_MSC(clusters: list,
                 smiles: list,
                 top_clusters: int = 10,
                 n_samples: int = 100,
                 MCS_threshold: float = 0.75,
                 save_path: str = None):
    """Generate and save/display MSC images for all clusters.

    Args:
        clusters (list): List of clusters (each cluster is a list of molecule indices).
        smiles (list): List of SMILES strings corresponding to the molecules.
        n_samples (int, optional): Number of molecules to sample from each cluster. Defaults to 100.
        MCS_threshold (float, optional): Threshold for MCS calculation. Defaults to 0.75.
        save_path (str, optional): Path to save the combined image. Defaults to None.
    """

    MCS_mols = []
    for cluster in clusters[:top_clusters]:
        MCS = sampled_MSC(cluster,
                           smiles,
                           n_samples=n_samples,
                           MCS_threshold=MCS_threshold,
                           return_samples=False)
        MCS_mols.append(MCS)

    img = Draw.MolsToGridImage(MCS_mols,
                                molsPerRow=5,
                                subImgSize=(250, 250),
                                legends=[f'Cluster {i+1}' for i in range(len(MCS_mols))])
    if save_path:
        img.save(save_path)
    else:
        return img
    

def cluster_mix_MCS_image(cluster,
                              smiles,
                              flags,
                              n_samples=50,
                              MCS_threshold=0.75,
                              save_path: str = None):
    """Generate MCS images for mixed clusters.

    Args:
        cluster (list): List of molecule indices in the cluster.
        smiles (list): List of SMILES strings corresponding to the molecules.
        n_samples (int, optional): Number of molecules to sample from the cluster. Defaults to 100."""
    
    n_samples = min(n_samples, len(cluster))
    if n_samples == 1:
        mol = Chem.MolFromSmiles(smiles[cluster[0]])
        return Draw.MolToImage(mol)

    # Get the SMILES for the sampled molecules
    remover = SaltRemover.SaltRemover()
    np.random.shuffle(cluster)
    sampled_smiles = [smiles[i] for i in cluster[:n_samples]]
    flags_sampled = [flags[i] for i in cluster[:n_samples]]
    mols = [Chem.MolFromSmiles(smi) for smi in sampled_smiles]
    mols = [remover.StripMol(mol) for mol in mols]

    # Find the MSC of those molecules
    MCS = rdFMCS.FindMCS(mols, threshold=MCS_threshold)
    MCS_mol = Chem.MolFromSmarts(MCS.smartsString)

    # Find substructure matches in the sampled molecules with the MCS
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

    img = Draw.MolsToGridImage(mols,
                                highlightAtomLists=highlight_lists,
                                molsPerRow=5,
                                subImgSize=(250, 250),
                                legends=flags_sampled)
    if save_path:
        img.save(save_path)
    else:
        return img