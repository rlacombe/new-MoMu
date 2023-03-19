import torch
import numpy as np
import torch_geometric.data 
import torch_geometric.utils 
import networkx 


def methylation(data):
    """
    Methylation reaction: Replaces a hydrogen atom with a methyl group (-CH3).

    'TODO':
    Hydroxylation: Adds a hydroxyl group (-OH) to a molecule.    
    Reduction: Converts a functional group (e.g., -C=O) to a different functional group (e.g., -CH2).
    Oxidation: Converts a functional group to a different functional group with more oxygen atoms (e.g., -CH2 to -COOH).
    Amination: Amination: Replaces a hydrogen atom with an amino group (-NH2).
    """

    # Turn PyG graph into NetworkX graph
    G = to_networkx(data, node_attrs=["x"], edge_attrs=["edge_attr"])

    # Find the index of the node with the desired feature value
    target_index = None
    for i in range(G.num_nodes):
        if G.x[i][4] > 0: # 'NOTE' this means the atom has a least one implicit H
            target_index = i
            break

    if target_index is not None:
        # Add the new node to the graph
        new_node_features = {
            'x': torch.tensor([5, 0, 4, 5, 3, 0, 2, 0, 0]),  # C atom with 3 H (-CH3 radical)
        }
        new_node_index = G.num_nodes  # Index of the new node
        G.add_nodes(1, **new_node_features)

        # Add an edge between the new node and the target node
        edge_index = torch.tensor([[new_node_index, target_index]], dtype=torch.long)
        new_node_features = {
            'edge_attr': torch.tensor([1, 0, 0]),  # single bond in -CH3
        }
        G.add_edges(edge_index.t(), **new_edge_features)

    # Return PyG graph
    data = from_networkx(G)
    return data









