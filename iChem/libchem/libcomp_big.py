import numpy as np # type: ignore
from collections import Counter
from typing import TYPE_CHECKING

from ._libchem_aux import (
	interiSIM,
	intraiSIM,
	MaxSum,
	MinSum,
	combo_counts,
	composition_per_cluster,
)
from ..bblean.similarity import jt_isim_medoid
from ..bblean.similarity import jt_sim_matrix_between_packed

if TYPE_CHECKING:
	from .libchem_big import LibChemBig
	from .libchem import LibChem


class LibCompBig:
	"""Class for comparing multiple large chemical libraries.

	This class mirrors the core comparison methods available in ``LibComparison``
	but consumes ``LibChemBig`` objects, which expose large-library data through
	attributes such as ``full_fps`` and ``cluster_medoids_fps``.
	"""

	def __init__(self):
		self.libraries: dict[str, "LibChemBig"] = {}
		self.library_names: list[str] = []
		self.combined_library: "LibChem | None" = None

	def add_library(self, library: "LibChemBig", lib_name: str) -> None:
		"""Add a ``LibChemBig`` instance to the comparison."""
		self.libraries[lib_name] = library
		self.library_names.append(lib_name)

	def _resolve_pair(self, lib1_name: str = None, lib2_name: str = None) -> tuple[str, str]:
		if lib1_name is None and lib2_name is None and len(self.libraries) == 2:
			lib1_name, lib2_name = list(self.libraries.keys())
		if lib1_name not in self.library_names or lib2_name not in self.library_names:
			raise ValueError("Both libraries must be specified and exist in the comparison libraries.")
		return lib1_name, lib2_name

	def _get_library_medoids(self, library: "LibChemBig") -> np.ndarray:
		medoids = getattr(library, "cluster_medoids_fps", None)
		if medoids is None:
			# Backward-compatible fallback for alternate naming.
			medoids = getattr(library, "medoid_fps", None)

		if medoids is None and hasattr(library, "save_cluster_medoids"):
			try:
				library.save_cluster_medoids()
			except Exception:
				# Some LibChemBig versions expose inconsistent attribute names.
				pass
			medoids = getattr(library, "cluster_medoids_fps", None)
			if medoids is None:
				medoids = getattr(library, "medoid_fps", None)

		if medoids is None:
			raise ValueError(
				"Cluster medoids are not available. Run cluster() and save_cluster_medoids() on each LibChemBig object first."
			)

		return np.array(medoids)

	def _get_library_medoids_with_smiles(self, library: "LibChemBig") -> tuple[np.ndarray, list[str]]:
		medoids_fps = self._get_library_medoids(library)
		medoids_smiles = getattr(library, "cluster_medoids_smiles", None)
		if medoids_smiles is None:
			medoids_smiles = getattr(library, "medoid_smiles", None)

		return np.array(medoids_fps), list(medoids_smiles)

	def compare_medoids(
		self,
		methodology: str = "MaxSum",
		lib1_name: str = None,
		lib2_name: str = None,
	) -> float:
		"""Compare cluster medoids between two ``LibChemBig`` libraries."""
		lib1_name, lib2_name = self._resolve_pair(lib1_name, lib2_name)
		lib1 = self.libraries[lib1_name]
		lib2 = self.libraries[lib2_name]

		fps1 = self._get_library_medoids(lib1)
		fps2 = self._get_library_medoids(lib2)

		if methodology == "MaxSum":
			sim_matrix = jt_sim_matrix_between_packed(fps1, fps2)
			return MaxSum(sim_matrix)
		if methodology == "MinSum":
			sim_matrix = jt_sim_matrix_between_packed(fps1, fps2)
			return MinSum(sim_matrix)
		if methodology == "intraiSIM":
			return intraiSIM(fps1, fps2)
		if methodology == "interiSIM":
			return interiSIM(fps1, fps2)

		raise ValueError(
			"Methodology must be one of: 'MaxSum', 'MinSum', 'intraiSIM', 'interiSIM'."
		)

	def compare_medoids_all(self, methodology: str = "MaxSum") -> list[list[float]]:
		"""Compare medoids across all registered libraries."""
		results = []

		for i, library_1 in enumerate(self.library_names):
			result_line = []
			for j, library_2 in enumerate(self.library_names):
				if j < i:
					result_line.append(results[j][i])
				else:
					score = self.compare_medoids(
						methodology=methodology,
						lib1_name=library_1,
						lib2_name=library_2,
					)
					result_line.append(score)
			results.append(result_line)

		return results

	def compare_medoids_heatmap(self, methodology: str = "MaxSum", save_path: str = None) -> None:
		"""Generate a symmetric heatmap for all-medoid comparisons."""
		from ..visualization.plots import symmetric_heatmap

		results = self.compare_medoids_all(methodology=methodology)
		symmetric_heatmap(results, labels=self.library_names, save_path=save_path)

	def cluster_libraries(
		self,
		methodology: str = "medoids",
		lib_names: list[str] = None,
		threshold: float = None,
		n_samples: int = None,
		verbose: bool = False,
	) -> None:
		"""Cluster molecules from multiple large libraries into a combined medoid library.

		Only ``methodology='medoids'`` is supported for ``LibChemBig``.
		The ``n_samples`` argument is accepted for compatibility and ignored.
		"""
		if methodology != "medoids":
			raise ValueError("Methodology must be 'medoids' for LibComparisonBig.")
		_ = n_samples
		self.combined_library = self._cluster_medoid_mix(
			threshold=threshold,
			lib_names=lib_names,
			verbose=verbose,
		)

	def _cluster_medoid_mix(
		self,
		threshold: float = None,
		lib_names: list[str] = None,
		verbose: bool = False,
	) -> "LibChem":
		"""Cluster combined medoids from selected ``LibChemBig`` libraries."""
		if lib_names is None:
			lib_names = list(self.library_names)
		if len(lib_names) < 2:
			raise ValueError("At least two libraries must be specified for comparison.")

		medoids_fps = []
		medoids_smiles = []
		medoids_flags = []

		for lib_name in lib_names:
			if lib_name not in self.library_names:
				raise ValueError(f"Library '{lib_name}' not found in comparison libraries.")

			lib = self.libraries[lib_name]
			lib_medoids_fps, lib_medoids_smiles = self._get_library_medoids_with_smiles(lib)
			medoids_fps.extend(lib_medoids_fps)
			medoids_smiles.extend(lib_medoids_smiles)
			medoids_flags.extend([lib_name] * len(lib_medoids_smiles))
			if verbose:
				print(f"Number of medoids in Library {lib_name}: {len(lib_medoids_smiles)}")

		medoids_fps = np.array(medoids_fps)
		n_medoids = len(medoids_fps)
		if verbose:
			print(f"Total number of medoids: {n_medoids}")

		from .libchem import LibChem
		combined_lib = LibChem()

		indices = np.arange(n_medoids)
		np.random.seed(42)
		np.random.shuffle(indices)
		medoids_fps = medoids_fps[indices]
		medoids_flags = [medoids_flags[i] for i in indices]
		medoids_smiles = [medoids_smiles[i] for i in indices]

		combined_lib.set_fingerprints(fingerprints=medoids_fps, packed=True)
		combined_lib.set_smiles(medoids_smiles)
		combined_lib.set_flags(medoids_flags)

		if threshold is None:
			combined_lib.set_threshold()
			threshold_combined = combined_lib.threshold
			library_thresholds = [
				self.libraries[name].threshold
				for name in lib_names
				if self.libraries[name].threshold is not None
			]
			library_thresholds.append(threshold_combined)
			threshold_final = max(library_thresholds)
			if verbose:
				print(f"Using clustering threshold: {threshold_final:.4f}")
		else:
			threshold_final = threshold
			if verbose:
				print(f"Using clustering threshold: {threshold_final:.4f}")

		combined_lib.set_threshold(threshold=threshold_final)
		combined_lib.cluster()

		return combined_lib

	def cluster_classification_counts(
		self,
		lib_names: list[str] = None,
	) -> tuple:
		"""Get counts and mapping of cluster compositions by library."""
		if self.combined_library is None:
			raise ValueError("No combined library found. Please run cluster_libraries() first.")

		lib_names = lib_names if lib_names is not None else list(self.library_names)
		counts, combo_map = combo_counts(self.combined_library.get_cluster_flags(), library_names=lib_names)
		return counts, combo_map

	def pie_chart_composition(
		self,
		lib_names: list[str] = None,
		save_path: str = None,
	) -> None:
		"""Generate a pie chart showing cluster composition by library."""
		from ..visualization.plots import pie_chart_mixed_clusters

		if self.combined_library is None:
			raise ValueError("No combined library found. Please run cluster_libraries() first.")
		labels = lib_names if lib_names is not None else list(self.library_names)
		counts, _ = combo_counts(self.combined_library.get_cluster_flags(), library_names=labels)
		pie_chart_mixed_clusters(counts, save_path=save_path)

	def venn_diagram_composition(
		self,
		lib_names: list[str] = None,
		save_path: str = None,
		upset: bool = False,
	) -> None:
		"""Generate a Venn or UpSet diagram showing overlap in cluster compositions."""
		from ..visualization.plots import venn_lib_comp

		if self.combined_library is None:
			raise ValueError("No combined library found. Please run cluster_libraries() first.")
		labels = lib_names if lib_names is not None else self.library_names
		counts, _ = combo_counts(self.combined_library.get_cluster_flags(), library_names=labels)
		venn_lib_comp(counts, lib_names=labels, save_path=save_path, upset=upset)

	def cluster_composition_counts(
		self,
		top: int = 20,
	) -> list[Counter]:
		"""Get detailed composition of the largest clusters."""
		if self.combined_library is None:
			raise ValueError("No combined library found. Please run cluster_libraries() first.")
		composition = composition_per_cluster(
			self.combined_library.get_cluster_flags(),
			top=top,
		)
		return composition

	def plot_cluster_composition(
		self,
		lib_names: list[str],
		top: int = 20,
		save_path: str = None,
	) -> None:
		"""Generate a bar chart showing composition of the largest clusters."""
		from ..visualization.plots import bar_chart_library_comparison

		if self.combined_library is None:
			raise ValueError("No combined library found. Please run cluster_libraries() first.")
		composition = composition_per_cluster(
			self.combined_library.get_cluster_flags(),
			top=top,
		)
		bar_chart_library_comparison(
			values=composition,
			lib_names=lib_names,
			save_path=save_path,
		)

	def cluster_visualization(
		self,
		cluster_number: int,
		save_path: str = None,
	):
		"""Visualize molecules in a combined cluster with MCS highlighting."""
		if self.combined_library is None:
			raise ValueError("No combined library found. Please run cluster_libraries() first.")

		from ..visualization.mol_images import cluster_mix_MCS_image

		if cluster_number > len(self.combined_library.clusters) - 1:
			raise ValueError(
				f"Cluster number {cluster_number} is out of range. There are only {len(self.combined_library.clusters)} clusters."
			)

		img = cluster_mix_MCS_image(
			cluster=self.combined_library.clusters[cluster_number],
			smiles=self.combined_library.smiles,
			flags=self.combined_library.flags,
			n_samples=25,
			MCS_threshold=0.75,
			save_path=save_path,
		)
		if save_path is None:
			return img



