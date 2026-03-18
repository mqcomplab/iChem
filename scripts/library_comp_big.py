from iChem.libchem.libcomp_big import LibCompBig
from iChem.libchem.libchem_big import LibChemBig
import glob


LibComp = LibCompBig()
for medoid_file in glob.glob("*medoids_fps.npy"):
    name = medoid_file.split("_medoids_fps.npy")[0].split("/")[-1]
    print(f"Processing {name}...")
    Lib1 = LibChemBig(library_name=name)
    Lib1.load_medoids_fps_and_smiles(medoid_file, medoid_file.replace("medoids_fps.npy", "medoids_smiles.smi"))

    LibComp.add_library(Lib1, name)

LibComp.cluster_libraries()
counts, mapping = LibComp.cluster_classification_counts()

LibComp.venn_diagram_composition(save_path='library_composition_venn.png', upset=True)
LibComp.pie_chart_composition(save_path='library_composition_pie.png')
LibComp.plot_cluster_composition(lib_names=list(LibComp.library_names), top=25, save_path='library_cluster_composition.png')

for key in mapping.keys():
    LibComp.cluster_visualization(cluster_number=mapping[key][0],
                                    save_path=f'cluster_{key}_structures.png')

