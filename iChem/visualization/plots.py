import matplotlib.pyplot as plt # type: ignore
import numpy as np # type: ignore
import pandas as pd # type: ignore
import seaborn as sns # type: ignore
from seaborn import heatmap # type: ignore
#import plotly.graph_objects as go
from collections import Counter, defaultdict
import iChem.bblean.similarity as iSIM

def clusters_pop_plot(bitbirch_obj,
                      save_path: str = None,
                      ):

    """Plot the population distribution of clusters as a stacked bar chart.

    Args:
        bitbirch_obj: BitBirch clustering object with fitted clusters.
        save_path (str, optional): Path to save the plot. Defaults to None.
    """

    # Calculate the counts of the populations
    populations = bitbirch_obj.get_cluster_populations()
    n_1000 = sum(1 for pop in populations if pop > 1000)
    n_100 = sum(1 for pop in populations if pop > 100)
    n_10 = sum(1 for pop in populations if pop > 10)
    n_1 = sum(1 for pop in populations if pop > 1)
    n_0 = sum(1 for pop in populations if pop > 0)

    plt.figure(figsize=(3, 4))
    plt.bar('Num_cluster', n_0, label='>0', color='blue')
    plt.bar('Num_cluster', n_1, label='>1', color='orange')
    plt.bar('Num_cluster', n_10, label='>10', color='gray')
    plt.bar('Num_cluster', n_100, label='>100', color='green')
    plt.bar('Num_cluster', n_1000, label='>1000', color='red')
    plt.legend()
    plt.ylabel('Number of Clusters')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=400)
    else:
        plt.show()


def clusters_pop_isim_plot(bitbirch_obj,
                           save_path: str = None,
                           figsize: tuple = (12, 6),
                           top=20,
                           initial=0):
    """Plot cluster population as bars with iSIM values on secondary axis.

    Args:
        bitbirch_obj: BitBirch clustering object with fitted clusters.
        save_path (str, optional): Path to save the plot. Defaults to None.
        figsize (tuple, optional): Figure size (width, height). Defaults to (12, 6).
        top (int, optional): Number of top clusters to display. Defaults to 20.
        initial (int, optional): Starting index for clusters to display. Defaults to 0."""

    # Get cluster populations and iSIM values
    all_populations = bitbirch_obj.get_cluster_populations()
    isim_values = bitbirch_obj.get_iSIM_clusters()
    
    # Calculate statistics before limiting
    total_clusters = len(all_populations)
    n_singletons = sum(1 for pop in all_populations if pop == 1)

    # Limit to top clusters for display
    populations = all_populations[initial:top]
    isim_values = isim_values[initial:top]

    # Create figure and primary axis
    fig, ax1 = plt.subplots(figsize=figsize)

    # Plot cluster populations as bars
    x = np.arange(len(populations))
    bars = ax1.bar(x, populations, alpha=0.7, color='blue', label='Population')
    ax1.set_xlabel('Cluster')
    ax1.set_ylabel('Population', color='blue')
    ax1.tick_params(axis='y', labelcolor='blue')
    ax1.set_xticks(x)
    ax1.set_xticklabels([str(i) for i in x], rotation=45, ha='right')

    # Create secondary axis for iSIM values
    ax2 = ax1.twinx()
    line = ax2.plot(x, isim_values, color='darkorange', marker='o', 
                    linewidth=2, markersize=6, label='iSIM')
    ax2.set_ylabel('iSIM', color='darkorange')
    ax2.tick_params(axis='y', labelcolor='darkorange')
    ax2.set_ylim(0, 1)

    # Add legends
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

    # Add annotation with cluster statistics
    annotation_text = f'Total Clusters: {total_clusters}\nSingletons: {n_singletons}'
    ax1.text(0.98, 0.98, annotation_text, transform=ax1.transAxes,
             fontsize=10, verticalalignment='top', horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.title('Cluster Population and iSIM')
    fig.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=400)
    else:
        plt.show()

def pie_chart_mixed_clusters(counts: dict,
                             save_path: str = None):
    """Generate a pie chart of mixed cluster compositions.

    Args:
        counts (dict): Dictionary with cluster composition counts.
        save_path (str, optional): Path to save the pie chart. Defaults to None.
    """
    plt.figure(figsize=(10, 5))
    counts = dict(sorted(counts.items(), key=lambda item: item[1], reverse=False))
    plt.pie(counts.values(), labels=[None]*len(counts), autopct='%1.1f%%')
    plt.legend(labels=counts.keys(), loc="center left", bbox_to_anchor=(1, 0.5))
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=400)
    else:
        plt.show()

def symmetric_heatmap(results: list,
                      labels: list,
                      save_path: str = None,
                      only_upper: bool = True):
    """Generate a symmetric heatmap from a square results matrix.

    Args:
        results (list): 2D list or array of results.
    """
    # Do a heatmap wit only the upper triangle and the diagonal filled
    if only_upper:
        heatmap(np.array(results),
                xticklabels=labels, yticklabels=labels,
                annot=True, fmt='.2f', cmap='viridis', mask=np.tril(np.ones_like(results, dtype=bool), k=-1))
    else:
        heatmap(np.array(results),
                xticklabels=labels, yticklabels=labels,
                annot=True, fmt='.2f', cmap='viridis')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=400)
    else:
        plt.show()

def bar_chart_library_comparison(values: list[Counter],
                                 lib_names: list,
                                 save_path: str = None):
    """Generate a stacked bar chart showing population composition per cluster.

    Each element in `values` is a Counter mapping library label -> count.
    Bars are stacked so each library occupies a segment; x labels are 0..N-1.
    """
    if not values:
        return

    n_clusters = len(values)
    x = np.arange(n_clusters)

    # Choose a color palette with enough distinct colors
    colors = sns.color_palette("tab10", n_colors=max(10, len(lib_names)))
    colors = colors[:len(lib_names)]

    bottoms = np.zeros(n_clusters, dtype=float)
    plt.figure(figsize=(10, 5))

    # For each library, plot its segment on each cluster bar
    for lib, col in zip(lib_names, colors):
        heights = np.array([ctr.get(lib, 0) for ctr in values], dtype=float)
        plt.bar(x, heights, bottom=bottoms, color=col, label=lib)
        bottoms += heights

    plt.xticks(x, [str(i) for i in x], fontsize=8, rotation=45)
    plt.xlabel("Cluster")
    plt.ylabel("Count")
    plt.yticks(np.arange(0, max(bottoms) + 1, step=max(1, max(bottoms) // 5)))
    plt.legend(title="Library", bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=400)
    else:
        plt.show()

def venn_lib_comp(counts: dict,
                  lib_names: list = None,
                save_path: str = None,
                upset = False):
    """Generate a Venn diagram showing library overlaps for 2-3 libraries, or UpSet plot for 4+ libraries.

    Args:
        counts (dict): Dictionary with library overlap counts.
        lib_names (list): List of library names.
        save_path (str, optional): Path to save the visualization. Defaults to None.

    Returns:
        fig: Matplotlib figure object.
    """

    n_libs = len(lib_names)
    
    # Get total number of clusters for calculation of percentages
    total_clusters = sum(counts.values())

    # Sort the counts dictionary labels
    counts = dict(sorted(counts.items()))

    # Pass the counts to percentage with only one decimal place
    counts_pct = {key: round((value / total_clusters) * 100, 1) for key, value in counts.items()}

    if n_libs <= 3 and not upset:
        # Use traditional Venn diagrams for 2-3 libraries
        from matplotlib_venn import venn2, venn3 # type: ignore
        
        # Change the count labels to be used in venn diagrams
        new_counts = {}
        for key in counts_pct.keys():
            if len(key.split('+')) == 3:
                new_counts["111"] = counts_pct[key]
            elif len(key.split('+')) == 1:
                if key == lib_names[0]:
                    new_counts["100"] = counts_pct[key]
                elif key == lib_names[1]:
                    new_counts["010"] = counts_pct[key]
                elif n_libs > 2 and key == lib_names[2]:
                    new_counts["001"] = counts_pct[key]
            elif len(key.split('+')) == 2:
                libs = key.split('+')
                if lib_names[0] in libs and lib_names[1] in libs:
                    new_counts["110"] = counts_pct[key]
                elif n_libs > 2 and lib_names[0] in libs and lib_names[2] in libs:
                    new_counts["101"] = counts_pct[key]
                elif n_libs > 2 and lib_names[1] in libs and lib_names[2] in libs:
                    new_counts["011"] = counts_pct[key]

        # Plot the venn diagram
        plt.figure(figsize=(6, 6))
        if n_libs == 2:
            venn2(subsets=(new_counts.get("100", 0),
                           new_counts.get("010", 0),
                           new_counts.get("110", 0)),
                  set_labels=(f"{lib_names[0]}", f"{lib_names[1]}"),
                  set_colors=('blue', 'orange'),
                  alpha=0.75)
            plt.tight_layout()
        else:  # n_libs == 3
            venn3(subsets=(new_counts.get("100", 0),
                           new_counts.get("010", 0),
                           new_counts.get("110", 0),
                           new_counts.get("001", 0),
                           new_counts.get("101", 0),
                           new_counts.get("011", 0),
                           new_counts.get("111", 0)),
                  set_labels=(f"{lib_names[0]}", f"{lib_names[1]}", f"{lib_names[2]}"),
                  set_colors=('blue', 'orange', 'green'),
                  alpha=0.75)
            plt.tight_layout()
    else:
        # Use UpSet plot for 4+ libraries (better than Venn for many sets)
        try:
            from upsetplot import UpSet # type: ignore
            from upsetplot import from_memberships # type: ignore
            import warnings
            warnings.filterwarnings("ignore", category=FutureWarning)
            warnings.filterwarnings("ignore", category=UserWarning)

            # Get the memberships lists
            members_lists = []
            for key in counts.keys():
                members_lists.append(list(key.split('+')))

            # Create the data structure for UpSet plot
            upset_data = from_memberships(members_lists, data=list(counts.values()))
            
            # Create UpSet plot
            upset = UpSet(upset_data,
                          show_percentages=True,
                          totals_plot_elements=0,
                          with_lines=True,
                          element_size=50,
                          facecolor='C0')
            fig = plt.figure(figsize=(12, 8))
            upset.plot(fig=fig)
            plt.grid(False)
            
        except ImportError:
            # Fallback: create a custom matrix-style visualization
            print("upsetplot library not installed. Using fallback visualization.")
            print("Install with: pip install upsetplot")
    
    if save_path:
        plt.savefig(save_path, dpi=400)
    else:
        plt.show()

    plt.close('all')

class Node:
    def __init__(self, members, height=0):
        self.members = frozenset(members)
        self.height = height
        self.children = []
        self.x = None


def build_tree(hierarchical_clusters):
    steps = sorted(hierarchical_clusters.keys())
    
    # Create leaf nodes (step 0)
    nodes = {frozenset(c): Node(c, height=0)
             for c in hierarchical_clusters[steps[0]]}
    
    for step in steps[1:]:
        prev_clusters = hierarchical_clusters[step - 1]
        new_clusters = hierarchical_clusters[step]
        
        prev_sets = [frozenset(c) for c in prev_clusters]
        new_sets = [frozenset(c) for c in new_clusters]
        
        for new_set in new_sets:
            if new_set not in nodes:
                # Find children
                children = [nodes[c] for c in prev_sets if c.issubset(new_set)]
                
                if len(children) >= 2:
                    parent = Node(new_set, height=step)
                    parent.children = children
                    
                    # Remove merged nodes
                    for c in children:
                        del nodes[c.members]
                    
                    nodes[new_set] = parent
    
    # Only root remains
    return list(nodes.values())[0]


def assign_x_positions(node, x_counter):
    if not node.children:
        node.x = next(x_counter)
    else:
        for child in node.children:
            assign_x_positions(child, x_counter)
        node.x = sum(child.x for child in node.children) / len(node.children)


def draw_tree(node, ax):
    for child in node.children:
        # vertical line
        ax.plot([child.x, child.x],
                [child.height, node.height],
                'k-', linewidth=1)

    if node.children:
        xs = [child.x for child in node.children]
        ax.plot([min(xs), max(xs)],
                [node.height, node.height],
                'k-', linewidth=1)

    for child in node.children:
        draw_tree(child, ax)


def dendrogram_bitbirch(hierarchical_clusters, initial_threshold=None):
    """Plot dendrogram from hierarchical clustering results.
    
    Parameters
    ----------
    hierarchical_clusters : dict
        Dictionary where keys are step indices and values are lists of clusters.
    initial_threshold : float, optional
        The initial similarity threshold used. If provided, the y-axis will show
        distance (1 - threshold) ranging from (1 - initial_threshold) to 1.
    """
    root = build_tree(hierarchical_clusters)
    
    fig, ax = plt.subplots(figsize=(8, 5))
    
    # Assign leaf positions
    from itertools import count
    assign_x_positions(root, count())
    
    # Convert heights to distance if threshold is provided
    if initial_threshold is not None:
        # Determine number of steps from the cluster data
        n_steps = max(hierarchical_clusters.keys()) + 1
        
        # Create mapping from step to distance (1 - threshold)
        thresholds = np.linspace(initial_threshold, 0, num=n_steps)
        distances = 1 - thresholds
        
        # Convert node heights to distances
        def convert_heights(node):
            step = int(node.height)
            if step >= len(distances):
                node.height = distances[-1]
            else:
                node.height = distances[step]
            for child in node.children:
                convert_heights(child)
        
        convert_heights(root)
    
    # Draw structure
    draw_tree(root, ax)
    
    # Configure y-axis and calculate positioning based on whether we have distance information
    if initial_threshold is not None:
        # Set y-axis limits from (1 - initial_threshold) to 1.0
        min_distance = 1 - initial_threshold
        max_distance = 1.0  # Maximum distance is always 1.0
        
        # Add small margin only at bottom for labels
        y_margin = 0.08 * (max_distance - min_distance)
        
        # Extension at top for tick visibility
        extension = 0.03 * (max_distance - min_distance)
        
        ax.set_ylim(min_distance - y_margin, max_distance + extension)
        
        ax.spines['left'].set_visible(True)
        ax.spines['left'].set_bounds(min_distance, max_distance)  # Spine from min_distance to 1.0
        ax.tick_params(axis='y', which='both', left=True, labelleft=True)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        
        # Calculate label position
        y_pos = min_distance - 0.04 * (max_distance - min_distance)
        
        # Vertical line extends from root to max_distance (1.0)
        ax.plot([root.x, root.x], [root.height, max_distance], 
               'k-', linewidth=1)
    else:
        # Original behavior for non-distance mode
        extension = 0.5
        y_margin = 0.5
        max_distance = root.height
        ax.set_ylim(-y_margin, root.height + extension)
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        y_pos = -0.2
        
        # Add vertical line at root as visual extension
        ax.plot([root.x, root.x], [root.height, root.height + extension], 
               'k-', linewidth=1)
    
    # Add bottom labels (only step 0 clusters)
    def add_labels(node):
        if not node.children:
            label = "\n".join(str(i) for i in sorted(node.members))
            ax.text(node.x, y_pos, label,
                    ha='center', va='top', fontsize=8)
        else:
            for child in node.children:
                add_labels(child)
    
    add_labels(root)
    
    # Set up axes
    ax.set_xticks([])
    ax.set_xlim(-1, None)
    
    plt.tight_layout()
    plt.show()
