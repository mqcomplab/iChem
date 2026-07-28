import numpy as np  # type: ignore
from typing import TYPE_CHECKING
from collections import defaultdict
from ..bitbirch._config import FINGERPRINT_TYPE, THRESHOLD, BRANCHING_FACTOR, MERGE_CRITERION, N_BITS
from ..bitbirch import cluster

from bblean.similarity import jt_sim_matrix_packed  # type: ignore

from ._libchem_aux import (
    MaxSum,
    MinSum,
    interiSIM,
    intraiSIM,
    composition_per_cluster,
    weighted_composition_per_cluster,
)

from .lib_chem import LibChem

REPRESENTATIVES_ONLY = True

class LibComparison:
    """Compare named representative LibChem objects."""

    def __init__(self, *libraries: "LibChem") -> None:
        self.libraries: dict[str, "LibChem"] = {}
        for lib in libraries:
            self.add_library(lib)
        self.cluster_results: dict | None = None

    def add_library(
            self,
            library: "LibChem",
    ) -> None:
        """Register a representative library for comparison."""
        if library.name in self.libraries:
            raise ValueError(f"Library '{library.name}' already exists.")

        self.libraries[library.name] = library

    def _resolve_pair_names(
            self,
            lib1_name: str | None,
            lib2_name: str | None,
    ) -> tuple[str, str]:
        if lib1_name is None or lib2_name is None:
            raise ValueError(
                "Please specify both library names for comparison. Available libraries: "
                f"{', '.join(self.libraries.keys())}"
            )
        if lib1_name not in self.libraries:
            raise ValueError(f"Library '{lib1_name}' not found.")
        if lib2_name not in self.libraries:
            raise ValueError(f"Library '{lib2_name}' not found.")

        return lib1_name, lib2_name

    def _get_pair_fingerprints(
            self,
            lib1_name: str | None,
            lib2_name: str | None,
    ) -> tuple[np.ndarray, np.ndarray]:
        lib1_name, lib2_name = self._resolve_pair_names(lib1_name, lib2_name)
        if self.libraries[lib1_name].fingerprints is None:
            self.libraries[lib1_name].generate_fingerprints(fp_type=FINGERPRINT_TYPE, n_bits=N_BITS)
            print(f"Generating fingerprints for library '{lib1_name}'...")
            print(f"Using default parameters: ECFP4, 2048 bits.")
            print("Pre-generate fingerprints if other fp_type or n_bits are desired.")
        if self.libraries[lib2_name].fingerprints is None:
            self.libraries[lib2_name].generate_fingerprints(fp_type=FINGERPRINT_TYPE, n_bits=N_BITS)
            print(f"Generating fingerprints for library '{lib2_name}'...")
            print(f"Using default parameters: ECFP4, 2048 bits.")
            print("Pre-generate fingerprints if other fp_type or n_bits are desired.")
        fps1 = self.libraries[lib1_name].fingerprints
        fps2 = self.libraries[lib2_name].fingerprints
        return np.asarray(fps1), np.asarray(fps2)

    @staticmethod
    def _compare_fingerprints(
            fps1: np.ndarray,
            fps2: np.ndarray,
            methodology: str,
    ) -> float:
        if methodology == "intraiSIM":
            return intraiSIM(fps1, fps2)

        if methodology == "interiSIM":
            return interiSIM(fps1, fps2)

        if methodology in {"MaxSum", "_MaxSum"}:
            sim_matrix = jt_sim_matrix_packed(fps1, fps2)
            return MaxSum(sim_matrix)

        if methodology in {"MinSum", "_MinSum"}:
            sim_matrix = jt_sim_matrix_packed(fps1, fps2)
            return MinSum(sim_matrix)

        raise ValueError(f"Unknown methodology: {methodology}")

    def compare_two_libraries(
            self,
            methodology: str = "intraiSIM",
            lib1_name: str | None = None,
            lib2_name: str | None = None,
            # Parameters for shared cluster analysis
            threshold: float = THRESHOLD,
            branching_factor: int = BRANCHING_FACTOR,
            merge_criterion: str = MERGE_CRITERION,
            representatives_only: bool = REPRESENTATIVES_ONLY,
    ) -> float:
        """Compare two representative libraries."""

        if methodology in {'intraiSIM', 'interiSIM', 'MaxSum', '_MaxSum', 'MinSum', '_MinSum'}:
            fps1, fps2 = self._get_pair_fingerprints(lib1_name, lib2_name)
            return self._compare_fingerprints(fps1, fps2, methodology)
        
        if methodology == "shared-clusters":
            if lib1_name == lib2_name:
                return 100.0  # A library is always 100% shared with itself
            results = self._cluster_library_pair(
                self.libraries[lib1_name],
                self.libraries[lib2_name],
                threshold=threshold,
                branching_factor=branching_factor,
                merge_criterion=merge_criterion
            )
            return LibComparison.shared_space_fraction(results,
                                                       representatives_only=representatives_only)
        
    def compare_all_libraries(
            self,
            methodology: str = "intraiSIM",
            # Parameters for shared cluster analysis
            threshold: float = THRESHOLD,
            branching_factor: int = BRANCHING_FACTOR,
            merge_criterion: str = MERGE_CRITERION,
            representatives_only: bool = REPRESENTATIVES_ONLY,
    ) -> np.ndarray:
        """Compare all registered libraries pairwise."""
        library_names = list(self.libraries.keys())
        n = len(library_names)
        results = np.zeros((n, n))
        for i in range(n):
            for j in range(i, n):
                lib1_name = library_names[i]
                lib2_name = library_names[j]
                value = self.compare_two_libraries(
                    methodology=methodology,
                    lib1_name=lib1_name,
                    lib2_name=lib2_name,
                    threshold=threshold,
                    branching_factor=branching_factor,
                    merge_criterion=merge_criterion,
                    representatives_only=representatives_only,
                )
                results[i, j] = value
                results[j, i] = value  # Symmetric matrix
        return results

    def compare_all_heatmap(
            self,
            methodology: str = "MaxSum",
            save_path: str | None = None,
            # Parameters for shared cluster analysis
            representatives_only: bool = REPRESENTATIVES_ONLY,
            threshold: float = THRESHOLD,
            branching_factor: int = BRANCHING_FACTOR,
            merge_criterion: str = MERGE_CRITERION,
    ) -> None:
        """Generate a heatmap for all representative-library comparisons."""
        from ..visualization.plots import symmetric_heatmap

        symmetric_heatmap(
            self.compare_all_libraries(methodology=methodology,
                                       representatives_only=representatives_only,
                                       threshold=threshold,
                                       branching_factor=branching_factor,
                                       merge_criterion=merge_criterion),
            labels=list(self.libraries.keys()),
            save_path=save_path,
        )

    ##### BELOW IS CLUSTERING METHODS #####
    def cluster_libraries(self,
                          threshold: float = None,
                          branching_factor: int = BRANCHING_FACTOR,
                          merge_criterion: str = MERGE_CRITERION) -> dict:
        """Cluster all or specified libraries together, tracking origin and composition.

        Parameters
        ----------
        library_names : list[str] | None
            Libraries to cluster. If None, clusters all registered libraries.
        threshold : float
            Similarity threshold for clustering.
        branching_factor : int
            BitBirch branching factor.
        merge_criterion : str
            BitBirch merge criterion.

        Returns
        -------
        dict
            Clustering results with keys:
            - 'cluster_ids': list of cluster indices per molecule
            - 'cluster_flags': list of library origins per cluster
            - 'cluster_sizes_count': count of clusters
            - 'composition_counts': counts by library combination
        """
        library_names = list(self.libraries.keys())

        print(
        "Clustering libraries together: "
        + ", ".join(library_names)
        )
                
        # Obtaining fingerprint and flags from LibChem objects
        n_total = sum([self.libraries[name].n_molecules for name in library_names])

        # Retrieve flags
        flags_list = np.empty(n_total, dtype=object)

        # Retrieve cluster
        sizes_list = np.empty(n_total, dtype=int)

        # Fingerprint generation, size tracking, and flag collection
        if self.libraries[library_names[0]].fingerprints is not None:
            dimension = self.libraries[library_names[0]].fingerprints.shape[1]
            fps_combined = np.zeros((n_total, dimension), dtype=np.uint8)
            tracker = 0
        else:
            self._generate_default_fps(self.libraries[library_names[0]], library_names[0])
            dimension = self.libraries[library_names[0]].fingerprints.shape[1]
            fps_combined = np.zeros((n_total, dimension), dtype=np.uint8)
            tracker = self.libraries[library_names[0]].fingerprints.shape[0]
            fps_combined[:tracker, :] = self.libraries[library_names[0]].fingerprints
            flags_list[:tracker] = self.libraries[library_names[0]].flags
            sizes_list[:tracker] = self.libraries[library_names[0]].cluster_sizes

        for k, library in enumerate(library_names):
            if tracker > 0 and k == 0:
                continue  # Skip first library since already processed
            if self.libraries[library].fingerprints is None:
                self._generate_default_fps(self.libraries[library], library)
            fps_combined[tracker:tracker + self.libraries[library].n_molecules, :] = self.libraries[library].fingerprints
            flags_list[tracker:tracker + self.libraries[library].n_molecules] = self.libraries[library].flags
            sizes_list[tracker:tracker + self.libraries[library].n_molecules] = self.libraries[library].cluster_sizes
            assert (
            self.libraries[library].n_molecules
            == self.libraries[library].fingerprints.shape[0]
            == len(self.libraries[library].flags)
            ), (
                f"Mismatch in molecule count, fingerprint count, and "
                f"flag count for library '{library}'."
            )
            tracker += self.libraries[library].n_molecules

        assert tracker == n_total, "Mismatch in total molecule count and fingerprint tracking."

        # Save temp_fingerprint file
        np.save("temp_fingerprints.npy", fps_combined)

        # Cluster
        cluster_ids = cluster(file_path="temp_fingerprints.npy",
                threshold=threshold,
                branching_factor=branching_factor,
                merge_criterion=merge_criterion,)

        # Delete temp file
        import os
        os.remove("temp_fingerprints.npy")

        cluster_flags = self._group_flags_by_cluster(cluster_ids, flags_list)
        cluster_sizes = self._group_flags_by_cluster(cluster_ids, sizes_list)
        cluster_smiles = self._group_flags_by_cluster(cluster_ids, np.concatenate([self.libraries[name].smiles for name in library_names]))
        cluster_compositions = composition_per_cluster(cluster_flags)
        weighted_compositions = weighted_composition_per_cluster(cluster_flags, cluster_sizes)

        from ._libchem_aux import combo_counts, weighted_combo_counts
        composition_counts, composition_mapping = combo_counts(cluster_flags, library_names)
        weighted_composition_counts, _ = weighted_combo_counts(cluster_flags, cluster_sizes, library_names)

        self.cluster_results = {
            # Raw clustering outputs
            'cluster_ids': cluster_ids,
            'cluster_flags': cluster_flags,
            'cluster_sizes': cluster_sizes,
            'cluster_smiles': cluster_smiles,

            # Per-cluster composition outputs
            'cluster_compositions': cluster_compositions,
            'cluster_compositions_weighted': weighted_compositions,

            # Overlap counts outputs
            'overlap_counts': composition_counts,
            'overlap_counts_weighted': weighted_composition_counts,

            # Overlap mapping outputs
            'overlap_mapping': composition_mapping
        }

    def get_cluster_results(self, **kwargs):
        """Return clustering results.
        Accepts:
        'cluster_ids': list of of lists with molecule indices per cluster (order as loaded)
        'cluster_flags': list of lists with flag values per cluster
        'cluster_sizes': list of cluster sizes
        'cluster_smiles': list of lists with SMILES strings per cluster
        'cluster_compositions': list of compositions per cluster
        'cluster_compositions_weighted': list of weighted compositions per cluster
        'overlap_counts': list of overlap counts per cluster
        'overlap_counts_weighted': list of weighted overlap counts per cluster
        'overlap_mapping': dictionary mapping overlap counts to cluster IDs"""
        if not self.cluster_results:
            raise ValueError("No clustering results available. Please run cluster_libraries() first.")
        if kwargs:
            # Validate kwargs
            valid_keys = {'cluster_ids', 'cluster_flags', 'cluster_sizes', 'cluster_smiles',
                          'cluster_compositions', 'cluster_compositions_weighted',
                          'overlap_counts', 'overlap_counts_weighted', 'overlap_mapping'}
            for key in kwargs:
                if key not in valid_keys:
                    raise ValueError(f"Invalid key '{key}' in kwargs. Valid keys are: {valid_keys}")
            return {key: self.cluster_results[key] for key in kwargs}
        return self.cluster_results
    
    @staticmethod
    def shared_space_fraction(clustering_results: dict,
                              representatives_only: bool = REPRESENTATIVES_ONLY) -> float:
        """Compute the fraction of clusters that are shared between libraries."""
        if clustering_results is None:
            raise ValueError("No clustering results provided.")
        
        if representatives_only:
            data = clustering_results['overlap_counts']
        else:
            data = clustering_results['overlap_counts_weighted']

        total = sum(data.values())
        shared = 0
        for key, count in data.items():
            if "+" in key:  # Indicates a shared cluster
                shared += count

        return (shared * 100) / total if total > 0 else 0.0

    @staticmethod
    def _cluster_library_pair(lib1 : LibChem,
                             lib2 : LibChem,
                             threshold: float = None,
                             branching_factor: int = BRANCHING_FACTOR,
                             merge_criterion: str = MERGE_CRITERION) -> dict:
        """Cluster two libraries together and compute shared cluster space fraction."""
        # Create a temporary LibComparison object with just the two libraries
        temp_comparison = LibComparison(lib1, lib2)

        temp_comparison.cluster_libraries(
            threshold=threshold,
            branching_factor=branching_factor,
            merge_criterion=merge_criterion
        )

        return temp_comparison.cluster_results

    @staticmethod
    def _generate_default_fps(lib, name):
        lib.generate_fingerprints(
            fp_type=FINGERPRINT_TYPE,
            n_bits=N_BITS,
            packed=True
        )
        print(f"Generating fingerprints for library '{name}'...")
        print(f"Using default parameters: {FINGERPRINT_TYPE}, {N_BITS} bits.")
        print("Pre-generate fingerprints if other fp_type or n_bits are desired.")

    @staticmethod
    def _group_flags_by_cluster(
            cluster_ids: list,
            flags: list,
    ) -> list:
        """Group flags by cluster ID.

        Returns
        -------
        list
            Per-cluster list of flags, indexed by cluster ID.
        """
        cluster_flags = []

        for cluster in cluster_ids:
            cluster_flags.append([flags[mol_idx] for mol_idx in cluster])

        return cluster_flags
    
    @staticmethod
    def _exclusive_shared_from_composition(
        composition,
    ):
        """
        Compute the fraction of each library that appears in
        exclusive clusters versus clusters shared with other libraries.

        When weighted=True, contributions are weighted by the
        represented population size.

        When weighted=False, contributions are based on
        representative counts only.

        Parameters
        ----------
        composition : list[Counter]
            Output of weighted_composition_per_cluster() or composition_per_cluster()

        Returns
        -------
        dict
            Per-library totals, exclusive and shared masses.
        """

        total = defaultdict(float)
        exclusive = defaultdict(float)
        shared = defaultdict(float)

        for comp in composition:

            # comp = Counter({lib: weight_in_cluster})
            unique_libs = list(comp.keys())

            # accumulate totals first
            for lib, mass in comp.items():
                total[lib] += mass

            # exclusive cluster
            if len(unique_libs) == 1:
                lib = unique_libs[0]
                exclusive[lib] += comp[lib]

            # shared cluster
            else:
                for lib, mass in comp.items():
                    shared[lib] += mass

        return {
            lib: {
                "total": float(total[lib]),
                "exclusive": float(exclusive[lib]),
                "shared": float(shared[lib]),
            }
            for lib in total
        }

    @staticmethod
    def exclusive_shared_proportions(
        cluster_flags,
        cluster_sizes=None,
        representatives_only = REPRESENTATIVES_ONLY,

    ):
        """
        Compute exclusive vs shared contributions using composition output.
        User can choose to use either representative counts (composition) or weighted composition.

        Parameters
        ----------
        cluster_flags : list
            Per-cluster list of flags, indexed by cluster ID.
        cluster_sizes : list
            Per-cluster list of sizes, indexed by cluster ID.
        representatives_only : bool
            Whether to use only representative counts (True) or all counts (False).

        Returns
        -------
        dict
            Per-library totals, exclusive and shared fractions.
        """
        if representatives_only:
            composition = composition_per_cluster(
                cluster_flags
            )
        else:
            assert cluster_sizes is not None, "cluster_sizes must be provided when weighted composition is desired."
            composition = weighted_composition_per_cluster(
                cluster_flags,
                cluster_sizes,
            )

        stats = LibComparison._exclusive_shared_from_composition(
            composition
        )

        # convert to fractions
        for lib in stats:
            total = stats[lib]["total"]
            stats[lib]["exclusive_fraction"] = stats[lib]["exclusive"] / total if total else 0.0
            stats[lib]["shared_fraction"] = stats[lib]["shared"] / total if total else 0.0

        return stats
    
    def venn_diagram_overlap(self,
                            percentages : bool = False,
                            save_path: str | None = None,
                            representatives_only : bool = REPRESENTATIVES_ONLY) -> None:
        """
        Generate a Venn diagram of library overlaps.
        Accepts a maximum of 3 libraries for visualization purposes.
        For more use the upset plot.

        Parameters
        ----------
        library_names : list[str]
            Libraries to include in the Venn diagram.
        representatives_only : bool
            Whether to use only representative counts (True) or all counts (False).

        Returns
        -------
        img | None
            Venn diagram image object or None if saved to a file.
        """
        from ..visualization.plots import venn_overlap

        library_names = list(self.libraries.keys())

        if len(library_names) > 3:
            raise ValueError("Venn diagram visualization is limited to a maximum of 3 libraries.")
        
        # Get the overlap counts
        if representatives_only:
            overlap_counts = self.cluster_results['overlap_counts']
        else:
            overlap_counts = self.cluster_results['overlap_counts_weighted']

        # Generate the Venn diagram
        venn_overlap(
            overlap_counts=overlap_counts,
            library_names=library_names,
            percentages=percentages,
            save_path=save_path
        )

    
    def upset_plot_overlap(self,
                           percentages: bool = False,
                           save_path: str | None = None,
                           representatives_only: bool = REPRESENTATIVES_ONLY) -> None:
        """
        Generate an UpSet plot of library overlaps.
        Useful for visualizing overlaps between more than 3 libraries.

        Parameters
        ----------
        library_names : list[str]
            Libraries to include in the UpSet plot.
        percentages : bool
            Whether to display percentages instead of raw counts.
        representatives_only : bool
            Whether to use only representative counts (True) or all counts (False).
        
        Returns
        ----------
        img | None
            UpSet plot image object or None if saved to a file.
        """

        from ..visualization.plots import upset_overlap

        library_names = list(self.libraries.keys())


        # Get the overlap counts        
        if representatives_only:
            overlap_counts = self.cluster_results['overlap_counts']
        else:
            overlap_counts = self.cluster_results['overlap_counts_weighted']

        # Generate the UpSet plot
        upset_overlap(
            overlap_counts=overlap_counts,
            library_names=library_names,
            percentages=percentages,
            save_path=save_path
        )

    def cluster_population_plot(self,
                                save_path: str | None = None,
                                top=20,
                                initial=0,
                                representatives_only: bool = REPRESENTATIVES_ONLY,
                                ):
        """Generate a plot of cluster populations.
        Each bar represents a cluster, colored by library composition."""

        composition_key = 'cluster_compositions' if representatives_only else 'cluster_compositions_weighted'
        from ..visualization.plots import bar_chart_library_comparison
        bar_chart_library_comparison(
            values=self.cluster_results[composition_key][initial:top],
            lib_names=list(self.libraries.keys()),
            save_path=save_path,
        )

    def display_cluster_molecules(
            self,
            cluster_id: int,
            mols_per_row: int = 5,
            sub_img_size=(250, 250),
            display_MCS: bool = True,
            save_path: str | None = None,
            max_items: int | None = None,
    ):
        """
        Display molecules from a cluster with the maximum common
        substructure highlighted if needed.

        Parameters
        ----------
        cluster_id
            Cluster to visualize.
        """

        if self.cluster_results is None:
            raise ValueError("No clustering results available.")

        if cluster_id >= len(self.cluster_results["cluster_ids"]):
            raise ValueError(
                f"Cluster {cluster_id} not found."
            )

        cluster_smiles = self.cluster_results["cluster_smiles"][cluster_id]
        cluster_flags = self.cluster_results["cluster_flags"][cluster_id]

        from ..visualization.mol_images import smiles_to_grid_image
        img = smiles_to_grid_image(
            cluster_smiles,
            mols_per_row=mols_per_row,
            sub_img_size=sub_img_size,
            legends=cluster_flags,
            MCS=display_MCS,
            max_items=max_items if max_items is not None else 50,
        )

        if save_path:
            try:
                img.save(save_path)
                print(f"Saved to {save_path}")
            except Exception as e:
                print(f"Error saving image: {e}")
                return None

            try:
                from PIL import Image
                img = Image.open(save_path)
            except Exception as e:
                print(f"Error opening saved image: {e}")
                img = None

        return img
