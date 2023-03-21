import torch
import numpy as np
import math


"""Some more utils"""

"""
'TODO':
Hydroxylation: Adds a hydroxyl group (-OH) to a molecule.    
Reduction: Converts a functional group (e.g., -C=O) to a different functional group (e.g., -CH2).
Oxidation: Converts a functional group to a different functional group with more oxygen atoms (e.g., -CH2 to -COOH).
"""

def torch_diff(t1, t2):
  """
  Get difference between tensors t1 and t2.
  Ex: torch_diff(torch.arange(10), torch.tensor([3, 5, 7]))
      returns: torch.tensor([0, 1, 4, 6, 8, 9]), torch.tensor([3, 5, 7])
  
  Returns: (difference between t1 and t2, intersection between t1 and t2)
  """
  combined = torch.cat((t1, t2))
  uniques, counts = combined.unique(return_counts=True)
  return uniques[counts == 1], uniques[counts > 1]


def get_sampled_indices(cond, rate):
  """
  Get indices where 'cond' holds.
  """
  idx = torch.where(cond)[0]
  num_idx_to_sample = math.ceil(idx.shape[0] * rate)
  idx = idx[torch.randperm(idx.shape[0])]
  return idx[:num_idx_to_sample].tolist()

def add_new_nodes(data, num_nodes, node_features):
  """
  data: graph
  num_nodes (int): number of nodes to add.
  node_features (list): node features for new nodes being added.
  """
  old_num_nodes = data.x.shape[0]
  new_node_features = torch.tile( torch.tensor(node_features), (num_nodes, 1))
  new_x = torch.cat((data.x, new_node_features)) # Add new nodes into x
  data.x = new_x # Assign new x to graph.
  return data

def get_edge_remove_idx(data, idx):
  """
  TODO: add comments
  """
  edge_idx_to_remove = []
  for i in idx:
    edge_idx_to_remove.extend(
        torch.where(data.edge_index[0, :] == i)[0].tolist()
    )
    edge_idx_to_remove.extend(
        torch.where(data.edge_index[1, :] == i)[0].tolist()
    )
  remove_idx = list(set(edge_idx_to_remove))
  keep_idx, _ = torch_diff(torch.arange(data.edge_index.shape[-1]), torch.tensor(remove_idx))

  return keep_idx, remove_idx

def add_new_edges(data, old_num_nodes, idx, edge_features):
  """
  TODO: add comments
  """
  new_node_idx = [_ for _ in range(old_num_nodes, old_num_nodes + len(idx))]
  edges_to_add = [[i, j] for i,j in zip(idx, new_node_idx)]
  edges_to_add += [[j, i] for i,j in zip(idx, new_node_idx)]
  edges_to_add = torch.tensor(edges_to_add).T
  new_edges = torch.cat((data.edge_index, edges_to_add), dim=-1)
  data.edge_index = new_edges

  # Add in attributes for the new edges.
  edge_attrs_to_add = torch.tile(torch.tensor(edge_features), (2*len(idx), 1))
  new_edge_attrs = torch.cat((data.edge_attr, edge_attrs_to_add))
  data.edge_attr = new_edge_attrs
  
  return data

def remove_nodes(data, idx):
  """
  Removes nodes from data.x according to the indices in idx.
  """
  idx_to_keep, _ = torch_diff(torch.arange(data.x.shape[0]), torch.tensor(idx))
  data.x = data.x[idx_to_keep]
  return data

def remove_edges(data, keep_idx):
  """
  Remove all edges involving indices in idx.
  """
  data.edge_index = data.edge_index[:, keep_idx]
  data.edge_attr = data.edge_attr[keep_idx, :]
  return data

def replace_hydrogen_with_group(data, rate, node_features, edge_features):
    # Sample indices of nodes to modify.
    idx = get_sampled_indices(data.x[:, 4] > 0, rate)
    if not idx: return data, False # No augmentations to be made.

    # Add new nodes.
    old_num_nodes = data.x.shape[0]
    data.x[idx, 4] -= 1 # Remove a Hydrogen.
    data = add_new_nodes(data, len(idx), node_features)

    # Now, add in the indices to hook the molecules up.
    # We want to add edges from idx to the new nodes.
    data = add_new_edges(data, old_num_nodes, idx, edge_features)

    return data, True

def replace_group_with_hydrogen(data, rate, cond):
  # Sample indices of nodes to modify.
  idx = get_sampled_indices(cond, rate)
  if not idx: return data, False # No augmentations to be made.
  
  # Now, get the nodes connected to those nodes.
  # Also eliminate the edges.
  edge_keep_idx, edge_remove_idx = get_edge_remove_idx(data, idx)

  # OK, now we'll loop over the edges to make the required adjustments.
  # Can't think of a faster way to do this...
  set_idx = set(idx)
  undirected_edges = torch.sort(data.edge_index[:, edge_remove_idx])[0].unique().reshape(1,-1).tolist() # make sure we don't count the same edge twice
  for edge in undirected_edges:
    node_idx = edge[0] if edge[0] not in set_idx else edge[1]  # get the node the removed node connects to
    data.x[node_idx, 4] += 1  # Add a hydrogen bond to that node.
  
  # Now, eliminate those nodes.
  data = remove_nodes(data, idx)

  # remove the edges containing the node we eliminated.
  data = remove_edges(data, edge_keep_idx)

  return data, True



# this works on some test examples. I tested it.
def methylation(data, rate=0.1):
    """
    Methylation reaction: Replaces a hydrogen atom with a methyl group (-CH3).
    Rate is how many to sample.
    """
    node_features =  [
        5,  # Atomic number is 6.
        0,  # Chirality (not sure)
        1,  # Degree. It's just one since we're only hooking it up to one atom.
        5,  # Formal charge (not sure)
        3,  # Number of hydrogen is 3.
        0,  # Num rad e (not sure)
        2,  # hybridization (not sure)
        0,  # is_armoatic (not sure)
        0   # is_in_ring (not sure)
    ]

    edge_features = [
        0,  # Single bond.
        0,
        0
    ]
    return replace_hydrogen_with_group(data, rate, node_features, edge_features)

def amination(data, rate=0.1):
    """
    Amination reaction: Replaces a hydrogen atom with an amino group (-NH2).
    Rate is how many to sample.
    """
    node_features = [
        6, # Nitrogen has atomic number 7
        0, # Chirality 
        1, # Degree. It's 1 because it only connects to the atom for which it's replacing a hydrogen
        5, # Formal charge. Not sure (sak romain)
        2, # Number of hydrogen. It's NH2, so 2 hydrogen
        0, # num_rad_e. Looks like it's 0 for all nitrogen I see.
        2,  # Hybridization. Not sure (ask romain)
        0,  # Not sure (ask romain)
        0  # Not sure (ask romain)
    ]

    edge_features = [
        1,  # It's a double bond
        0,  # Not sure (ask romain)
        0   # Not sure (ask romain)
    ]

    return replace_hydrogen_with_group(data, rate, node_features, edge_features)


def demethylation(data, rate=0.1):
  """
  Replace a methly gruop with a hydrogen atom.
  """
  cond = torch.logical_and(
    data.x[:, 4] == 3, # It has 3 hydrogen bonds
    data.x[:, 0] == 5  # It is a carbon atom.
  )
  return replace_group_with_hydrogen(data, rate, cond)

def deamination(data, rate=0.1):
  """
  Reaplce NH2 with a hydrogen.
  """
  # Sample indices of nodes to modify.
  cond = torch.logical_and(
    data.x[:, 4] == 2, # It has 2 hydrogen bonds
    data.x[:, 0] == 6  # It is a carbon atom.
  )
  return replace_group_with_hydrogen(data, rate, cond)


"""
Graph Augmentation Functions.
"""
def identity(data, rate=0.1):
    return data

def chemical_augmentation(data, rate=0.1, num_augs_to_try=2):
    chemical_aug_fns = [
        methylation,
        amination,
        demethylation,
        deamination
    ]
    # Randomly permute the indices so we can do several augmentations.
    aug_indices = np.random.permutation(np.arange(len(chemical_aug_fns))).tolist()
    completed_augs = 0
    for aug_index in aug_indices:
        data, aug = chemical_aug_fns[aug_index](data)
        if aug: completed_augs += 1
        if completed_augs >= num_augs_to_try: break  # We'll keep going if some augmentations don't take affect.
    
    return data


def drop_nodes(data, rate=0.1):
    """
    Randomly dropping certain ratio of nodes.
    For those nodes to be dropped, remove all their edges by the following statements:
    adj[drop_node_idx, :] = 0, adj[:, drop_node_idx] = 0.

    :param data: input (class: torch_geometric.data.Data)
    :param rate: drop node rate
    :return: output (class: torch_geometric.data.Data)
    """
    node_num, _ = data.x.size()
    _, edge_num = data.edge_index.size()
    drop_num = int(node_num * rate)

    idx_drop = np.random.choice(node_num, drop_num, replace=False)
#     print(idx_drop)
    edge_index = data.edge_index.numpy()
    ori_edge_index = edge_index.T.tolist()

    adj = torch.zeros((node_num, node_num))
    adj[edge_index[0], edge_index[1]] = 1
    adj[idx_drop, :] = 0
    adj[:, idx_drop] = 0
    edge_index = adj.nonzero().t()

    data.edge_index = edge_index
    aft_edge_index = edge_index.numpy().T.tolist()
    keep_idx = []
    for idx, each in enumerate(ori_edge_index):
        if each in aft_edge_index:
            keep_idx.append(idx)
    data.edge_attr = data.edge_attr[keep_idx, :]
#     print(list(set(range(node_num))-set(idx_drop.tolist())))
#     data.x = data.x[list(set(range(node_num))-set(idx_drop.tolist())),:]

    return data


def permute_edges(data, rate=0.1, only_drop=True):
    """
    Randomly adding and dropping certain ratio of edges.

    :param data: input (class: torch_geometric.data.Data)
    :param rate: add or drop edge rate
    :param only_drop: if True, only drop edges; if False, not only add but also drop edges
    :return: output (class: torch_geometric.data.Data)
    """

    node_num, _ = data.x.size()
    _, edge_num = data.edge_index.size()
    permute_num = int(edge_num * rate)

    edge_index = data.edge_index.transpose(0, 1).numpy()

    idx_add = np.random.choice(node_num, (permute_num, 2))
    idx_add = [[idx_add[n, 0], idx_add[n, 1]] for n in range(permute_num) if
               [idx_add[n, 0], idx_add[n, 1]] not in edge_index.tolist()]
    # print(idx_add)
    if not only_drop and idx_add:
        edge_index = np.concatenate(
            (edge_index[np.random.choice(edge_num, edge_num - permute_num, replace=False)], idx_add), axis=0)
    else:
        edge_index = edge_index[np.random.choice(edge_num, edge_num - permute_num, replace=False)]

    data.edge_index = torch.tensor(edge_index).transpose_(0, 1)

    return data


def subgraph(data, rate=0.8):
    """
    Samples a subgraph using random walk.

    :param data: input (class: torch_geometric.data.Data)
    :param rate: rate
    :return: output (class: torch_geometric.data.Data)
    """

    node_num, _ = data.x.size()
    _, edge_num = data.edge_index.size()
    sub_num = int(node_num * rate)

    edge_index = data.edge_index.numpy()
    ori_edge_index = edge_index.T.tolist()

    idx_sub = [np.random.randint(node_num, size=1)[0]]
#     print(idx_sub)
    idx_neigh = set([n for n in edge_index[1][edge_index[0] == idx_sub[0]]])
#     print(idx_neigh)

    count = 0
    while len(idx_sub) <= sub_num:
        count = count + 1
        if count > 1.5 * node_num:
            break
        if len(idx_neigh) == 0:
            break
        sample_node = np.random.choice(list(idx_neigh))
        if sample_node in idx_sub:
            continue
        idx_sub.append(sample_node)
        idx_neigh = idx_neigh.union(set([n for n in edge_index[1][edge_index[0] == idx_sub[-1]]]))
#         print(idx_neigh)
#     print(idx_sub)
#     print(idx_neigh)

    idx_drop = [n for n in range(node_num) if n not in idx_sub]
    edge_index = data.edge_index.numpy()

    adj = torch.zeros((node_num, node_num))
    adj[edge_index[0], edge_index[1]] = 1
    adj[idx_drop, :] = 0
    adj[:, idx_drop] = 0
    edge_index = adj.nonzero().t()

    data.edge_index = edge_index
    aft_edge_index = edge_index.numpy().T.tolist()
    keep_idx = []
    for idx, each in enumerate(ori_edge_index):
        if each in aft_edge_index:
            keep_idx.append(idx)
    data.edge_attr = data.edge_attr[keep_idx,:]

    return data


def mask_nodes(data, rate=0.1):
    """
    Randomly masking certain ratio of nodes.
    For those nodes to be masked, replace their features with vectors sampled in a normal distribution.

    :param data: input (class: torch_geometric.data.Data)
    :param rate: mask node rate
    :return: output (class: torch_geometric.data.Data)
    """

    node_num, feat_dim = data.x.size()
    mask_num = int(node_num * rate)

    idx_mask = np.random.choice(node_num, mask_num, replace=False)
    data.x[idx_mask] = torch.tensor(np.random.normal(loc=0.5, scale=0.5, size=(mask_num, feat_dim)),
                                    dtype=torch.float32)

    return data
