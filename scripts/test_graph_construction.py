import os
import sys
import pandas as pd
import torch
import matplotlib.pyplot as plt
import networkx as nx
from pathlib import Path

# Add the project root directory to the Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.data import IOCounterGraph

def test_graph_construction():
    """
    Test the graph construction module by:
    1. Loading mutual information data
    2. Constructing a graph with the specified threshold
    3. Visualizing the graph structure
    """
    print("Testing graph construction module...")
    
    # Set paths
    data_dir = os.path.join(project_root, "data")
    mi_file = os.path.join(data_dir, "mutual_information2.csv")
    output_dir = os.path.join(project_root, "logs", "testing")
    os.makedirs(output_dir, exist_ok=True)
    
    # Set MI threshold
    mi_threshold = 0.3259  # 90th percentile threshold
    
    # Initialize graph constructor
    graph_constructor = IOCounterGraph(mi_threshold=mi_threshold)
    
    # Load mutual information data
    mi_df = graph_constructor.load_mutual_information(mi_file)
    print(f"Loaded mutual information matrix with shape: {mi_df.shape}")
    
    # Construct graph
    edge_index, edge_attr = graph_constructor.construct_graph(mi_df)
    print(f"Constructed graph with {len(graph_constructor.counter_names)} nodes and {edge_attr.shape[0]} edges")
    
    # Visualize graph
    visualize_graph(graph_constructor, output_dir)
    
    print(f"Graph construction test completed. Visualization saved to {output_dir}")
    return graph_constructor

def visualize_graph(graph_constructor, output_dir):
    """
    Visualize the constructed graph using NetworkX.
    
    Args:
        graph_constructor (IOCounterGraph): Graph constructor object
        output_dir (str): Directory to save the visualization
    """
    # Create NetworkX graph
    G = nx.Graph()
    
    # Add nodes
    for i, name in enumerate(graph_constructor.counter_names):
        G.add_node(i, name=name)
    
    # Add edges
    edge_index = graph_constructor.edge_index.numpy()
    edge_attr = graph_constructor.edge_attr.numpy()
    
    # Only add one direction of edges (undirected graph)
    added_edges = set()
    for i in range(edge_index.shape[1]):
        src, dst = edge_index[0, i], edge_index[1, i]
        if (src, dst) not in added_edges and (dst, src) not in added_edges:
            G.add_edge(src, dst, weight=float(edge_attr[i]))
            added_edges.add((src, dst))
    
    # Plot graph
    plt.figure(figsize=(16, 12))
    
    # Use spring layout for node positions
    pos = nx.spring_layout(G, seed=42)
    
    # Draw nodes
    nx.draw_networkx_nodes(G, pos, node_size=300, node_color='lightblue')
    
    # Draw edges with width based on mutual information
    edge_weights = [G[u][v]['weight'] for u, v in G.edges()]
    nx.draw_networkx_edges(G, pos, width=edge_weights, alpha=0.7)
    
    # Draw node labels
    labels = {i: f"{i}: {data['name']}" for i, data in G.nodes(data=True)}
    nx.draw_networkx_labels(G, pos, labels, font_size=8)
    
    # Draw edge labels (mutual information values)
    edge_labels = {(u, v): f"{G[u][v]['weight']:.2f}" for u, v in G.edges()}
    nx.draw_networkx_edge_labels(G, pos, edge_labels, font_size=6)
    
    plt.title(f"I/O Counter Graph (MI Threshold: {graph_constructor.mi_threshold})")
    plt.axis('off')
    plt.tight_layout()
    
    # Save figure
    plt.savefig(os.path.join(output_dir, "graph_visualization.png"), dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    test_graph_construction()
