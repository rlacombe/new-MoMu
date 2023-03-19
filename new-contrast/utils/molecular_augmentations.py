import torch
import random
import numpy as np
import networkx as nx
from torch_geometric.utils import to_networkx, from_networkx

"""
'NOTE': this module implements graph augmentations inspired by chemical reactions:
- Methylation: Replaces a hydrogen atom with a methyl group (-CH3).
- Hydroxylation: Adds a hydroxyl group (-OH) to a molecule. 
- Amination: Replaces a hydrogen atom with an amino group (-NH2).
- Oxidation/Reduction: Converts a methyl group (-CH3) to a -C=O or -COOH.
"""


def methylation(data):
    """
    Methylation reaction: Replaces a hydrogen atom with a methyl group (-CH3).
    """

    # Turn PyG graph into NetworkX graph
    G = graph_data_obj_to_nx_simple(data)

    # Find the index of the node with the desired feature value
    target_atoms = []
    for i in range(G.number_of_nodes()):
        if G.nodes[i]['x'][4] > 0: # 'NOTE' this means the atom has a least one implicit H
            target_atoms.append(i)

    if target_atoms is not []:

        # Pick a random atom to which to bind -CH3
        target_index = random.choice(target_atoms)

        # Add a new C atom node to the graph
        new_node_features = {
            'x': torch.tensor([5, 0, 4, 5, 3, 0, 2, 0, 0]),  # C atom with 3 H (-CH3 radical)
        }
        new_node_index = G.number_of_nodes()  # Index of the new node
        G.add_node(new_node_index, **new_node_features)

        # Replace H atom
        G.nodes[target_index]['x'][4] -= 1

        # Add an edge between the new node and the target node
        #edge_index = torch.tensor([[new_node_index, target_index]], dtype=torch.long)
        new_edge_features = {
            'edge_attr': torch.tensor([0, 0, 0]),  # single bond in -CH3
        }
        G.add_edge(new_node_index, target_index, **new_edge_features)

    # Return PyG graph
    data = nx_to_graph_data_obj_simple(G)
    return data


def demethylation(data):
    """
    Demethylation reaction: Replaces a methyl group (-CH3) with a hydrogen atom.
    """
    
    # Turn PyG graph into NetworkX graph
    G = graph_data_obj_to_nx_simple(data)

    # Filter out small molecules
    if G.number_of_nodes() < 5: return data

    # Find the index of the node with the desired feature value
    target_atoms = []
    for i in range(G.number_of_nodes()):
        if (G.nodes[i]['x'][0] == 5) and (G.nodes[i]['x'][4] == 3): # 'NOTE' -CH3 radical
            target_atoms.append(i)

    if target_atoms is not []:

        # Pick a random atom to replace by H
        target_index = random.choice(target_atoms)

        # Find the other atom this binds to and increment its H count 
        neighbor_index = G.neighbors(target_index)[0]
        G.nodes[neighbor_index]['x'][4] += 1

        # Remove target node
        G.remove_node(target_index)

    # Return PyG graph
    data = nx_to_graph_data_obj_simple(G)
    return data


def graph_data_obj_to_nx_simple(data):
    """
    Converts graph Data object required by the pytorch geometric package to
    network x data object. NB: Uses simplified atom and bond features,
    and represent as indices. NB: possible issues with recapitulating relative
    stereochemistry since the edges in the nx object are unordered.
    :param data: pytorch geometric Data object
    :return: network x object
    """
    G = nx.Graph()

    # atoms
    atom_features = data.x.cpu().numpy()
    num_atoms = atom_features.shape[0]
    for i in range(num_atoms):
        atomic_num_idx, chirality_tag_idx = atom_features[i]
        G.add_node(i, atom_num_idx=atomic_num_idx, chirality_tag_idx=chirality_tag_idx)
        pass

    # bonds
    edge_index = data.edge_index.cpu().numpy()
    edge_attr = data.edge_attr.cpu().numpy()
    num_bonds = edge_index.shape[1]
    for j in range(0, num_bonds, 2):
        begin_idx = int(edge_index[0, j])
        end_idx = int(edge_index[1, j])
        bond_type_idx, bond_dir_idx = edge_attr[j]
        if not G.has_edge(begin_idx, end_idx):
            G.add_edge(begin_idx, end_idx, bond_type_idx=bond_type_idx,
                       bond_dir_idx=bond_dir_idx)

    return G


def nx_to_graph_data_obj_simple(G):
    """
    Converts nx graph to pytorch geometric Data object. Assume node indices
    are numbered from 0 to num_nodes - 1. NB: Uses simplified atom and bond
    features, and represent as indices. NB: possible issues with
    recapitulating relative stereochemistry since the edges in the nx
    object are unordered.
    :param G: nx graph obj
    :return: pytorch geometric Data object
    """
    # atoms
    num_atom_features = 2  # atom type,  chirality tag
    atom_features_list = []
    for _, node in G.nodes(data=True):
        atom_feature = [node['atom_num_idx'], node['chirality_tag_idx']]
        atom_features_list.append(atom_feature)
    x = torch.tensor(np.array(atom_features_list), dtype=torch.long)

    # bonds
    num_bond_features = 2  # bond type, bond direction
    if len(G.edges()) > 0:  # mol has bonds
        edges_list = []
        edge_features_list = []
        for i, j, edge in G.edges(data=True):
            edge_feature = [edge['bond_type_idx'], edge['bond_dir_idx']]
            edges_list.append((i, j))
            edge_features_list.append(edge_feature)
            edges_list.append((j, i))
            edge_features_list.append(edge_feature)

        # data.edge_index: Graph connectivity in COO format with shape [2, num_edges]
        edge_index = torch.tensor(np.array(edges_list).T, dtype=torch.long)

        # data.edge_attr: Edge feature matrix with shape [num_edges, num_edge_features]
        edge_attr = torch.tensor(np.array(edge_features_list),
                                 dtype=torch.long)
    else:   # mol has no bonds
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, num_bond_features), dtype=torch.long)

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

    return data



