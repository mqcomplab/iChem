# BitBIRCH-Lean Python Package: An open-source clustering module based on iSIM.
#
# If you find this software useful please cite the following articles:
# - BitBIRCH: efficient clustering of large molecular libraries:
#   https://doi.org/10.1039/D5DD00030K
# - BitBIRCH Clustering Refinement Strategies:
#   https://doi.org/10.1021/acs.jcim.5c00627
# - BitBIRCH-Lean:
#   (preprint) https://www.biorxiv.org/content/10.1101/2025.10.22.684015v1
#
# Copyright (C) 2025  The Miranda-Quintana Lab and other BitBirch developers, comprised
# exclusively by:
# - Ramon Alain Miranda Quintana <ramirandaq@gmail.com>, <quintana@chem.ufl.edu>
# - Krisztina Zsigmond <kzsigmond@ufl.edu>
# - Ignacio Pickering <ipickering@chem.ufl.edu>
# - Kenneth Lopez Perez <klopezperez@chem.ufl.edu>
# - Miroslav Lzicar <miroslav.lzicar@deepmedchem.com>
#
# This program is free software: you can redistribute it and/or modify it under the
# terms of the GNU General Public License as published by the Free Software Foundation,
# version 3 (SPDX-License-Identifier: GPL-3.0-only).
#
# Portions of this file are licensed under the BSD 3-Clause License
# Copyright (c) 2007-2024 The scikit-learn developers. All rights reserved.
# (SPDX-License-Identifier: BSD-3-Clause). Copies or reproductions of code in this
# file must in addition adhere to the BSD-3-Clause license terms. A
# copy of the BSD-3-Clause license can be located at the root of this repository, under
# ./LICENSES/BSD-3-Clause.txt.
#
# Portions of this file were previously licensed under the LGPL 3.0
# license (SPDX-License-Identifier: LGPL-3.0-only), they are relicensed in this program
# as GPL-3.0, with permission of all original copyright holders:
# - Ramon Alain Miranda Quintana <ramirandaq@gmail.com>, <quintana@chem.ufl.edu>
# - Vicky (Vic) Jung <jungvicky@ufl.edu>
# - Kenneth Lopez Perez <klopezperez@chem.ufl.edu>
# - Kate Huddleston <kdavis2@chem.ufl.edu>
# - Lexin Chen <le.chen@ufl.edu>
# - Jherome Brylle Woody Santos <ja.santos@ufl.edu>
#
# This program is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
# PARTICULAR PURPOSE.  See the GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License along with this
# program. This copy can be located at the root of this repository, under
# ./LICENSES/GPL-3.0-only.txt.  If not, see <http://www.gnu.org/licenses/gpl-3.0.html>.

import numpy as np # type: ignore
from scipy import sparse # type: ignore
from ._bbreal import calc_centroid, max_separation, jt_isim_real

def set_merge(merge_criterion):
    """
    Sets merge_accept function for merge_subcluster, based on user specified merge_criteria. 

    Diameter: Average Tanimoto distance between cluster members. 

    Parameters:
    -----------
    merge_criterion : str
                        merge criterion to use. Currently only 'diameter' is supported
    features : int
                        number of features in the data. 

    Returns:
    --------
    merge_accept : function 
                        function that determines if cluster is accepted to merge based on the specified criteria
    """
    if merge_criterion == 'diameter':
        def merge_accept_function(threshold, new_ls, new_ss, new_n):
            ij_array = 0.5 * (new_ls ** 2 - new_ss)
            ij = np.sum(ij_array)
            inners = (new_n - 1) * np.sum(new_ss)
            return ij/(inners - ij) >= threshold
    else:
        raise ValueError(f"Unsupported merge criterion: '{merge_criterion}'. Currently only 'diameter' is supported.")
    
    globals()['merge_accept_function'] = merge_accept_function


def _iterate_sparse_X(X):
    """This little hack returns a densified row when iterating over a sparse
    matrix, instead of constructing a sparse matrix for every row that is
    expensive.
    """
    n_samples, n_features = X.shape
    X_indices = X.indices
    X_data = X.data
    X_indptr = X.indptr

    for i in range(n_samples):
        row = np.zeros(n_features, dtype=np.float32)
        startptr, endptr = X_indptr[i], X_indptr[i + 1]
        nonzero_indices = X_indices[startptr:endptr]
        row[nonzero_indices] = X_data[startptr:endptr]
        yield row

def _split_node(node, threshold, branching_factor):
    """The node has to be split if there is no place for a new subcluster
    in the node.
    1. Two empty nodes and two empty subclusters are initialized.
    2. The pair of distant subclusters are found.
    3. The properties of the empty subclusters and nodes are updated
       according to the nearest distance between the subclusters to the
       pair of distant subclusters.
    4. The two nodes are set as children to the two subclusters.
    """
    new_subcluster1 = _BFSubcluster()
    new_subcluster2 = _BFSubcluster()
    new_node1 = _BFNode(
        threshold=threshold,
        branching_factor=branching_factor,
        is_leaf=node.is_leaf,
        n_features=node.n_features,
        dtype=node.init_centroids_.dtype,
    )
    new_node2 = _BFNode(
        threshold=threshold,
        branching_factor=branching_factor,
        is_leaf=node.is_leaf,
        n_features=node.n_features,
        dtype=node.init_centroids_.dtype,
    )
    new_subcluster1.child_ = new_node1
    new_subcluster2.child_ = new_node2

    if node.is_leaf:
        if node.prev_leaf_ is not None:
            node.prev_leaf_.next_leaf_ = new_node1
        new_node1.prev_leaf_ = node.prev_leaf_
        new_node1.next_leaf_ = new_node2
        new_node2.prev_leaf_ = new_node1
        new_node2.next_leaf_ = node.next_leaf_
        if node.next_leaf_ is not None:
            node.next_leaf_.prev_leaf_ = new_node2  
    
    # O(N) implementation of max separation
    farthest_idx, node1_sim, node2_sim = max_separation(node.centroids_)    
    # Notice that max_separation is returning similarities
    node1_closer = node1_sim > node2_sim
    # Make sure node1 is closest to itself even if all distances are equal.
    # This can only happen when all node.centroids_ are duplicates leading to all
    # distances between centroids being zero.
    node1_closer[farthest_idx[0]] = True

    for idx, subcluster in enumerate(node.subclusters_):
        if node1_closer[idx]:
            new_node1.append_subcluster(subcluster)
            new_subcluster1.update(subcluster)
        else:
            new_node2.append_subcluster(subcluster)
            new_subcluster2.update(subcluster)

    # Break circular references and free memory
    node.subclusters_ = []
    del node      
    del new_node1, new_node2
    return new_subcluster1, new_subcluster2


class _BFNode:
    """Each node in a BFTree is called a BFNode.

    The BFNode can have a maximum of branching_factor
    number of BFSubclusters.

    Parameters
    ----------
    threshold : float
        Threshold needed for a new subcluster to enter a BFSubcluster.

    branching_factor : int
        Maximum number of BF subclusters in each node.

    is_leaf : bool
        We need to know if the BFNode is a leaf or not, in order to
        retrieve the final subclusters.

    n_features : int
        The number of features.

    Attributes
    ----------
    subclusters_ : list
        List of subclusters for a particular BFNode.

    prev_leaf_ : _BFNode
        Useful only if is_leaf is True.

    next_leaf_ : _BFNode
        next_leaf. Useful only if is_leaf is True.
        the final subclusters.

    init_centroids_ : ndarray of shape (branching_factor + 1, n_features)
        Manipulate ``init_centroids_`` throughout rather than centroids_ since
        the centroids are just a view of the ``init_centroids_`` .

    centroids_ : ndarray of shape (branching_factor + 1, n_features)
        View of ``init_centroids_``.

    """

    def __init__(self, *, threshold, branching_factor, is_leaf, n_features, dtype):
        self.threshold = threshold
        self.branching_factor = branching_factor
        self.is_leaf = is_leaf
        self.n_features = n_features

        # The list of subclusters, centroids and squared norms
        # to manipulate throughout.
        self.subclusters_ = []
        self.init_centroids_ = np.zeros((branching_factor + 1, n_features), dtype=dtype)
        self.prev_leaf_ = None
        self.next_leaf_ = None

    def append_subcluster(self, subcluster):
        n_samples = len(self.subclusters_)
        self.subclusters_.append(subcluster)
        self.init_centroids_[n_samples] = subcluster.centroid_
        
        # Keep centroids as views. In this way
        # if we change init_centroids, it is sufficient
        self.centroids_ = self.init_centroids_[: n_samples + 1, :]
        
    def update_split_subclusters(self, subcluster, new_subcluster1, new_subcluster2):
        """Remove a subcluster from a node and update it with the
        split subclusters.
        """
        ind = self.subclusters_.index(subcluster)
        self.subclusters_[ind] = new_subcluster1
        self.init_centroids_[ind] = new_subcluster1.centroid_
        self.centroids_[ind] = new_subcluster1.centroid_
        self.append_subcluster(new_subcluster2)

    def insert_bf_subcluster(self, subcluster):
        """Insert a new subcluster into the node."""
        if not self.subclusters_:
            self.append_subcluster(subcluster)
            return False

        threshold = self.threshold
        branching_factor = self.branching_factor
        # We need to find the closest subcluster among all the
        # subclusters so that we can insert our new subcluster.
        a = np.dot(self.centroids_, subcluster.centroid_)
        sim_matrix = a / (np.sum(self.centroids_**2, axis = 1) + np.sum(subcluster.centroid_**2) - a)
        closest_index = np.argmax(sim_matrix)
        closest_subcluster = self.subclusters_[closest_index]

        # If the subcluster has a child, we need a recursive strategy.
        if closest_subcluster.child_ is not None:
            split_child = closest_subcluster.child_.insert_bf_subcluster(subcluster)

            if not split_child:
                # If it is determined that the child need not be split, we
                # can just update the closest_subcluster
                closest_subcluster.update(subcluster)
                self.init_centroids_[closest_index] = self.subclusters_[closest_index].centroid_
                self.centroids_[closest_index] = self.subclusters_[closest_index].centroid_
                return False

            # things not too good. we need to redistribute the subclusters in
            # our child node, and add a new subcluster in the parent
            # subcluster to accommodate the new child.
            else:
                new_subcluster1, new_subcluster2 = _split_node(
                    closest_subcluster.child_,
                    threshold,
                    branching_factor
                )
                self.update_split_subclusters(
                    closest_subcluster, new_subcluster1, new_subcluster2
                )

                if len(self.subclusters_) > self.branching_factor:
                    return True
                return False

        # good to go!
        else:
            merged = closest_subcluster.merge_subcluster(subcluster, self.threshold)
            if merged:
                self.centroids_[closest_index] = closest_subcluster.centroid_
                self.init_centroids_[closest_index] = closest_subcluster.centroid_
                return False

            # not close to any other subclusters, and we still
            # have space, so add.
            elif len(self.subclusters_) < self.branching_factor:
                self.append_subcluster(subcluster)
                return False

            # We do not have enough space nor is it closer to an
            # other subcluster. We need to split.
            else:
                self.append_subcluster(subcluster)
                return True


class _BFSubcluster:
    """Each subcluster in a BFNode is called a BFSubcluster.

    A BFSubcluster can have a BFNode has its child.

    Parameters
    ----------
    linear_sum : ndarray of shape (n_features,), default=None
        Sample. This is kept optional to allow initialization of empty
        subclusters.

    Attributes
    ----------
    n_samples_ : int
        Number of samples that belong to each subcluster.

    linear_sum_ : ndarray
        Linear sum of all the samples in a subcluster. Prevents holding
        all sample data in memory.

    centroid_ : ndarray of shape (branching_factor + 1, n_features)
        Centroid of the subcluster. Prevent recomputing of centroids when
        ``BFNode.centroids_`` is called.
    
    mol_indices : list, default=[]
        List of indices of molecules included in the given cluster.

    child_ : _BFNode
        Child Node of the subcluster. Once a given _BFNode is set as the child
        of the _BFNode, it is set to ``self.child_``.
    """

    def __init__(self, *, linear_sum = None, mol_indices = []):
        if linear_sum is None:
            self.n_samples_ = 0
            self.centroid_ = self.linear_sum_ = 0
            self.mol_indices = []
            self.sq_sum = 0
        else:
            self.n_samples_ = 1
            self.centroid_ = self.linear_sum_ = linear_sum
            self.mol_indices = mol_indices
            self.sq_sum = self.centroid_**2
        
        self.child_ = None

    def update(self, subcluster):
        self.n_samples_ += subcluster.n_samples_
        self.linear_sum_ += subcluster.linear_sum_
        self.mol_indices += subcluster.mol_indices
        self.centroid_ = calc_centroid(self.linear_sum_, self.n_samples_)
        self.sq_sum += subcluster.sq_sum

    def merge_subcluster(self, nominee_cluster, threshold):
        """Check if a cluster is worthy enough to be merged. If
        yes then merge.
        """
        new_ls = self.linear_sum_ + nominee_cluster.linear_sum_
        new_n = self.n_samples_ + nominee_cluster.n_samples_
        new_ss = self.sq_sum + nominee_cluster.sq_sum
        
        if merge_accept_function(threshold, new_ls, new_ss, new_n):
            new_centroid = calc_centroid(new_ls, new_n)
            (
                self.n_samples_,
                self.linear_sum_,
                self.centroid_,
                self.mol_indices,
                self.sq_sum
            ) = (new_n, new_ls, new_centroid, self.mol_indices + nominee_cluster.mol_indices, new_ss)
            return True
        return False
    

class BBReal():
    """Implements the BBReal clustering algorithm for online clustering of molecular trajectories.
    
    BBReal combines a BIRCH CF-tree with RMSD-calibrated merge decisions to provide
    simple and physically interpretable clustering. The algorithm performs online clustering
    in a single pass, maintaining a clear guarantee that the average distance to the
    centroid of each cluster remains within the chosen tolerance throughout the process.
    
    For each new frame insertion, the algorithm evaluates the candidate cluster after
    hypothetical inclusion of the new frame. If the merge satisfies the bound implied
    by the threshold, the frame is accepted and the clustering features are updated.
    This approach is memory-bounded, fast, and enables efficient assignment of frames
    in an incremental operation with clusters that are easy to explain.
    
    Parameters
    ----------
    threshold : float, default=1.0
        The RMSD tolerance threshold for cluster membership. This serves as an intuitive
        control for granularity: increasing threshold reduces the number of clusters,
        concentrates coverage in the largest states, and broadens within-cluster RMSD
        distributions. The threshold represents the actual physical tolerance and provides
        interpretable choices for clustering resolution.

    branching_factor : int, default=1024
        Maximum number of subclusters in each node. When a new sample would cause
        a node to exceed this limit, the node is split into two nodes with subclusters
        redistributed between them. The parent subcluster is removed and two new
        subclusters are added as parents of the split nodes.

    Attributes
    ----------
    root_ : _BFNode
        Root of the BF Tree.

    dummy_leaf_ : _BFNode
        Start pointer to all the leaves.

    subcluster_centers_ : ndarray
        Centroids of all subclusters read directly from the leaves.

    index_tracker : int
        Counter tracking the current frame/molecule index.

    first_call : bool
        Flag indicating if this is the first call to fit.

    Notes
    -----
    The tree data structure consists of nodes with each node containing a number of
    subclusters. The maximum number of subclusters in a node is determined by the
    branching factor. Each subcluster maintains a linear sum, squared sum, molecule
    indices, and the number of samples in that subcluster. Additionally, each
    subcluster can have a child node if it is not a member of a leaf node.

    For a new point entering the root, it is merged with the closest subcluster
    and the linear sum, molecule indices, and sample count are updated. This process
    continues recursively until the properties of the leaf node are updated.

    """


    def __init__(
        self,
        *,
        threshold: float = 0.5,
        branching_factor: int = 1024,
        merge_criterion: str = 'diameter',
    ):
        self.threshold = threshold
        self.branching_factor = branching_factor
        self.index_tracker = 0
        self.first_call = True
        if merge_criterion is not None:
            set_merge(merge_criterion)

    def fit(self, X):
        """
        Build a BF Tree for the input data.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Input data.

        Returns
        -------
        self
            Fitted estimator.
        """

        # TODO: Add input verification

        return self._fit(X)

    def _fit(self, X):
        threshold = self.threshold
        branching_factor = self.branching_factor

        n_features = X.shape[1]
        d_type = np.float32  # Force float32 for memory efficiency

        # If partial_fit is called for the first time or fit is called, we
        # start a new tree.
        if self.first_call:
            # The first root is the leaf. Manipulate this object throughout.
            self.root_ = _BFNode(
                threshold=threshold,
                branching_factor=branching_factor,
                is_leaf=True,
                n_features=n_features,
                dtype=d_type,
            )
    
            # To enable getting back subclusters.
            self.dummy_leaf_ = _BFNode(
                threshold=threshold,
                branching_factor=branching_factor,
                is_leaf=True,
                n_features=n_features,
                dtype=d_type,
            )
            self.dummy_leaf_.next_leaf_ = self.root_
            self.root_.prev_leaf_ = self.dummy_leaf_

        # Cannot vectorize. Enough to convince to use cython.
        if not sparse.issparse(X):
            iter_func = iter
        else:
            iter_func = _iterate_sparse_X

        for sample in iter_func(X):
            #set_bits = np.sum(sample)
            subcluster = _BFSubcluster(linear_sum=sample.astype(np.float32).copy(), mol_indices = [self.index_tracker])
            split = self.root_.insert_bf_subcluster(subcluster)

            if split:
                new_subcluster1, new_subcluster2 = _split_node(
                    self.root_, threshold, branching_factor
                )
                del self.root_
                self.root_ = _BFNode(
                    threshold=threshold,
                    branching_factor=branching_factor,
                    is_leaf=False,
                    n_features=n_features,
                    dtype=d_type,
                )
                self.root_.append_subcluster(new_subcluster1)
                self.root_.append_subcluster(new_subcluster2)
            self.index_tracker += 1
        
        self.first_call = False
        return self
    
    def fit_BFs(self, X):
        """Method to fit a BitBirch Real model from a list of BitFeatures.
        
        Parameters:
        -----------
        
        X : list of shape (n_BFs, 4)
            List of BitFeatures to fit the model from. Computed by a previous bbreal model.

        Returns:
        -------
        self
        """

        # Check if the inpute is a list of BitFeatures
        if not isinstance(X, list):
            raise TypeError("X must be a list of BitFeatures.")
        if len(X[0]) != 4:
            raise ValueError("Each BitFeature must be a tuple of (n_samples_, linear_sum_, sq_sum, mol_indices).")
        
        # Set parameters
        threshold = self.threshold
        branching_factor = self.branching_factor

        n_features = len(X[0][1])
        d_type = np.float32  # Force float32 for memory efficiency


        # Initialize the tree
        if self.first_call:
            self.root_ = _BFNode(
                threshold=threshold,
                branching_factor=branching_factor,
                is_leaf=True,
                n_features=n_features,
                dtype=d_type,
            )
    
            self.dummy_leaf_ = _BFNode(
                threshold=threshold,
                branching_factor=branching_factor,
                is_leaf=True,
                n_features=n_features,
                dtype=d_type,
            )
            self.dummy_leaf_.next_leaf_ = self.root_
            self.root_.prev_leaf_ = self.dummy_leaf_

        for sample in iter(X):

            cluster = _BFSubcluster()
            cluster.n_samples_ = sample[0]
            cluster.linear_sum_ = sample[1].astype(np.float32)
            cluster.sq_sum = sample[2].astype(np.float32)
            cluster.mol_indices = sample[3]

            cluster.centroid_ = calc_centroid(cluster.linear_sum_, cluster.n_samples_)

            split = self.root_.insert_bf_subcluster(cluster)

            if split:
                new_subcluster1, new_subcluster2 = _split_node(
                    self.root_, threshold, branching_factor
                )
                del self.root_
                self.root_ = _BFNode(
                    threshold=threshold,
                    branching_factor=branching_factor,
                    is_leaf=False,
                    n_features=n_features,
                    dtype=d_type,
                )
                self.root_.append_subcluster(new_subcluster1)
                self.root_.append_subcluster(new_subcluster2)

        self.first_call = False
        return self
    
    def recluster_inplace(
        self,
        iterations: int = 1,
        extra_threshold: float = 0.0,
        shuffle: bool = False,
        seed: int | None = None,
        verbose: bool = False,
        stop_early: bool = False,
    ):
        """Refine singleton clusters by re-inserting them into the tree.
        
        This method extracts all current subclusters from the tree, resets it,
        optionally increases the threshold, and refits all subclusters. This can
        help reduce the number of singleton clusters by giving them another chance
        to merge with existing clusters under potentially more relaxed conditions.
        
        Parameters
        ----------
        iterations : int, default=1
            The maximum number of refinement iterations to perform.
            
        extra_threshold : float, default=0.0
            The amount to increase the current threshold in each iteration.
            Increasing the threshold makes the merge criterion more permissive,
            allowing more subclusters to merge.
            
        shuffle : bool, default=False
            Whether to shuffle the order of subclusters before reinsertion.
            Shuffling can help avoid systematic biases in the clustering order.
            
        seed : int or None, default=None
            Random seed for shuffling. Only used if shuffle=True.
            
        verbose : bool, default=False
            Whether to print progress information during reclustering.
            
        stop_early : bool, default=False
            Whether to stop iterations early if no singletons remain or if
            the number of singletons hasn't changed since the last iteration.
            
        Returns
        -------
        self : BBReal
            The fitted estimator with refined clusters.
            
        Raises
        ------
        ValueError
            If the model has not been fitted yet.
            
        Notes
        -----
        - This method extracts BitFeatures directly as tuples (n_samples_, linear_sum_, sq_sum, mol_indices)
          from all leaf subclusters before resetting and refitting the tree.
        - Singleton clusters (clusters with only 1 sample) are particularly likely
          to find better matches during reclustering, especially when threshold is increased.
        - The order of reinsertion can affect the final clustering, which is why
          the shuffle parameter is provided.
        - The method preserves all clustering features (linear_sum_, sq_sum, mol_indices)
          from the previous clustering without creating unnecessary intermediate objects.
        """
        import random
        
        if self.first_call:
            raise ValueError("The model has not been fitted yet.")
        
        singletons_before = 0
        for iteration_num in range(iterations):
            # Get all BitFeatures as tuples directly (memory efficient)
            bf_tuples = self.get_BFs(as_objects=False)
            
            # Count the number of clusters and singletons from tuples
            # Tuple format: (n_samples_, linear_sum_, sq_sum, mol_indices)
            n_clusters = len(bf_tuples)
            singleton_bfs = sum(1 for bf_tuple in bf_tuples if bf_tuple[0] == 1)
            
            # Check stopping criteria
            if stop_early:
                if singleton_bfs == 0 or singleton_bfs == singletons_before:
                    # No more singletons to refine or no progress made
                    if verbose:
                        print(f"Stopping early at iteration {iteration_num + 1}")
                    break
            singletons_before = singleton_bfs
            
            # Print progress
            if verbose:
                print(f"Iteration {iteration_num + 1}/{iterations}")
                print(f"  Current number of clusters: {n_clusters}")
                print(f"  Current number of singletons: {singleton_bfs}")
            
            # Optionally shuffle the subclusters
            if shuffle:
                random.seed(seed)
                random.shuffle(bf_tuples)
            
            # Reset the tree
            self.reset()
            
            # Change the threshold
            self.threshold += extra_threshold
            
            # Refit using fit_BFs method
            self.fit_BFs(bf_tuples)
        
        # Print final stats
        if verbose:
            bf_tuples = self.get_BFs(as_objects=False)
            n_clusters = len(bf_tuples)
            singleton_bfs = sum(1 for bf_tuple in bf_tuples if bf_tuple[0] == 1)
            print(f"\nFinal Results:")
            print(f"  Final number of clusters: {n_clusters}")
            print(f"  Final number of singletons: {singleton_bfs}")
        
        return self

    def _get_leaves(self):
        """
        Retrieve the leaves of the BF Node.

        Returns
        -------
        leaves : list of shape (n_leaves,)
            List of the leaf nodes.
        """
        leaf_ptr = self.dummy_leaf_.next_leaf_
        leaves = []
        while leaf_ptr is not None:
            leaves.append(leaf_ptr)
            leaf_ptr = leaf_ptr.next_leaf_
        return leaves
    
    def _get_BFs(self):
        """Method to return BitFeatures of the leaves subclusters"""
        if self.first_call:
            raise ValueError('The model has not been fitted yet.')
        
        BFs = []
        for leaf in self._get_leaves():
            for subcluster in leaf.subclusters_:
                BFs.append(subcluster)

        # Sort the BFs based on the size of the clusters (number of molecules in each cluster)
        BFs = sorted(BFs, key = lambda x: x.n_samples_, reverse=True)

        return BFs
    
    def get_BFs(self, as_objects = False):
        """Method to return BitFeatures of the leaves subclusters as a list of tuples or as a list of objects
        
        tuple format: (n_samples_, linear_sum_, sq_sum, mol_indices)
        object format: list of _BFSubcluster objects"""
        if self.first_call:
            raise ValueError('The model has not been fitted yet.')
        
        BFs = self._get_BFs()

        if as_objects:
            return BFs
        else:
            return [(subcluster.n_samples_,
                     subcluster.linear_sum_,
                     subcluster.sq_sum,
                     subcluster.mol_indices
                     ) for subcluster in BFs]
    
    def get_centroids(self):
        """Method to return a list of Numpy arrays containing the centroids' fingerprints"""
        if self.first_call:
            raise ValueError('The model has not been fitted yet.')
        
        centroids = [calc_centroid(subcluster.linear_sum_, subcluster.n_samples_) for subcluster in self._get_BFs()]
        
        return centroids
    
    def get_cluster_mol_ids(self):
        """Method to return the indices of molecules in each cluster"""
        if self.first_call:
            raise ValueError('The model has not been fitted yet.')
        
        clusters_mol_id = [subcluster.mol_indices for subcluster in self._get_BFs()]

        return clusters_mol_id
    
    def get_cluster_populations(self) -> list[int]:
        """Method to return the number of molecules in each cluster"""
        if self.first_call:
            raise ValueError('The model has not been fitted yet.')
        
        clusters_populations = [subcluster.n_samples_ for subcluster in self._get_BFs()]

        return clusters_populations
    
    def get_iSIM_clusters(self) -> list[float]:
        """Method to return the iSIM values of each cluster"""
        if self.first_call:
            raise ValueError('The model has not been fitted yet.')
        
        clusters_iSIM = [jt_isim_real(subcluster.linear_sum_, subcluster.sq_sum, subcluster.n_samples_) for subcluster in self._get_BFs()]

        return clusters_iSIM
    
    def reset(self):
        """Method to reset the model to its initial state."""
        self.root_ = None
        self.dummy_leaf_ = None
        self.index_tracker = 0
        self.first_call = True

