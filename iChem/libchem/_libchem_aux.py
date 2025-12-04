import numpy as np # type: ignore
from ..bblean.similarity import jt_isim_packed

def interiSIM(fps1: np.ndarray,
              fps2: np.ndarray) -> float:
    """Calculate the inter-library iSIM between two sets of packed fingerprints.

    Args:
        fps1 (np.ndarray): Packed fingerprints of the first library.
        fps2 (np.ndarray): Packed fingerprints of the second library.

    Returns:
        float: The inter-library iSIM value.
    """
    n1 = len(fps1)
    n2 = len(fps2)
    n3 = n1 + n2

    iSIM_1 = jt_isim_packed(np.array(fps1))

    iSIM_2 = jt_isim_packed(np.array(fps2))

    iSIM_3 = jt_isim_packed(
        np.vstack((fps1, fps2)),
    )

    interiSIM = (iSIM_3 * n3 * (n3 - 1) - iSIM_1 * n1 * (n1 - 1) - iSIM_2 * n2 * (n2 - 1)) / (2 * n1 * n2)

    return interiSIM

def intraiSIM(fps1: np.ndarray,
              fps2: np.ndarray) -> float:
    """Calculate the intra-library iSIM for a set of packed fingerprints.

    Args:
        fps1 (np.ndarray): Packed fingerprints of the first library.
        fps2 (np.ndarray): Packed fingerprints of the second library.
    Returns:
        float: The intra-library iSIM value.
    """

    isim_value = jt_isim_packed(
                np.vstack((fps1, fps2))
            )
    
    return isim_value

def MaxSum(similarityMatrix: np.ndarray) -> float:
    """Determine the MaxSum cluster medoids from a similarity matrix.

    Args:
        SimilarityMatrix (np.ndarray): A square matrix of pairwise similarities.
    
    Returns:
        MaxSum float: The MaxSum value. Average max sim by column and row.
    """
    total_similarity = 0.0
    for i in range(similarityMatrix.shape[0]):
        max_sim = np.max(similarityMatrix[i, :])
        total_similarity += max_sim
    for k in range(similarityMatrix.shape[1]):
        max_sim = np.max(similarityMatrix[:, k])
        total_similarity += max_sim
    medoids_maxsum = total_similarity / (similarityMatrix.shape[0] + similarityMatrix.shape[1])

    return medoids_maxsum

def MinSum(similarityMatrix: np.ndarray) -> float:
    """Determine the MinSum cluster medoids from a similarity matrix.

    Args:
        SimilarityMatrix (np.ndarray): A square matrix of pairwise similarities.

    Returns:
        MinSum float: The MinSum value. Average min sim by column and row.
    """
    total_similarity = 0.0
    for i in range(similarityMatrix.shape[0]):
        min_sim = np.min(similarityMatrix[i, :])
        total_similarity += min_sim
    for k in range(similarityMatrix.shape[1]):
        min_sim = np.min(similarityMatrix[:, k])
        total_similarity += min_sim
    medoids_minsum = total_similarity / (similarityMatrix.shape[0] + similarityMatrix.shape[1])

    return medoids_minsum