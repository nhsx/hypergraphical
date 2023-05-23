import networkx as nx
import hypernetx as hnx
import matplotlib.pyplot as plt
import random
from string import ascii_uppercase as auc
from PIL import Image
from itertools import chain, combinations
from ast import literal_eval

# import streamlit as st
import math
import numpy as np
import pandas as pd


###############################################################################
# IMPORT .TXT FILES AND IMAGES
###############################################################################
def display_markdown_from_file(file_path, tab):
    """Import markdown from .txt file

    Args:
        file_path (str): file path where .txt file is
        tab (var): variable name of the tab this is to be viewed in
    """
    with open(file_path, "r") as f:
        text = f.read()
        tab.markdown(text)


def add_image(image_path, width, height):
    """Read and return a resized image"""
    image = Image.open(image_path)
    modified_image = image.resize((width, height))
    return modified_image


###############################################################################
# CALCULATE MAXIMUM HYPEREDGES AND HYPERARCS
###############################################################################

# def N_deg_hyperarcs(n, d, b_hyperarcs=True):
#     """
#     Given an n-node directed hypergraph, how many d-degree hyperarcs are
#     there?

#     INPUTS:
#     -----------------
#         n (int) : Number of total nodes in directed hypergraph.

#         d (int) : Degree of hyperarcs to count

#         b_hyperarcs (bool) : Flag to tell function whether to only count
#         B-hyperarcs or all hyperarc variants (B-, BF- and F-hyperarcs).
#     """
#     # Estimate n choose k
#     no_hyperedges = int(N_choose_k(n, d))
#     if b_hyperarcs:
#         no_hyperarcs = d
#     else:
#         no_hyperarcs = 0
#         for i in range(1, d):
#             # no_i_hyp = int(d! / (i! * (d - i)!))
#             no_i_hyp = int(N_choose_k(d, i))
#             no_hyperarcs += no_i_hyp

#     return no_hyperedges * no_hyperarcs


def N_max_hyperarcs(n_diseases, b_hyp=True):
    """
    Compute the maximum possible number of hyperarcs

    INPUTS:
    ----------------
        n_diseases (int) : Number of diseases (nodes) in the directed
        hypergraph

        b_hyp (bool) : Flag to only count the number of B-hyperarcs. If False,
        will count B-, F- and BF-hyperarcs. Note, if set to True, this is
        incidentally how many F-hyperarcs are possible due to the symmetry of
        constructing the tail and head node sets.
    """
    # Initialise sum as number of nodes as to account for self-loops
    hyperarc_sum = n_diseases

    # Loop over hyperedge degrees
    for k in range(2, n_diseases + 1):
        # Estimate n choose k
        # comb = N_choose_k(n_diseases, k)
        comb = math.comb(n_diseases, k)

        # Count possible hyperarcs of hyperedge degree, depending on
        # if we only count B-hyperarcs or not
        if not b_hyp:
            hyperarc_sum += (2**k - 2) * comb
        else:
            hyperarc_sum += k * comb

    return int(hyperarc_sum)


def N_max_hyperedges(n_diseases):
    """
    Given an n-node hypergraph, how many edges of degree 2 or more are there?

    INPUTS:
    -----------------
        n_diseases (int) : Number of total nodes in hypergraph.
    """

    # Total number of hyperedges of n disease node undirected hypergraph
    # Ignores self-edges (hyperedges with only one node included)
    # Also ignores a hypergraph with no edges
    no_hyperedges = (2**n_diseases) - n_diseases - 1

    return int(no_hyperedges)


###############################################################################
# GENERATE FAKE PATIENTS, THEIR AGGREGATE PROGRESSIONS & WORKLISTS
###############################################################################


def agg_prog(final_prog):
    """Find the aggregate progression set from a final progression set.
    This is used for defining the hyperarcs.

    Args:
        final_prog (list): A list of strings where the order of the
                elements match the order of occurence e.g. ['a', 'c', 'b']

    Returns:
        list: Aggregated progression sets to show disease progression.
                NOTE: This does not consider duplicates (no temporal element)
    """
    aggregate_prog_list = []
    for k in range(
        2, len(final_prog) + 1
    ):  # 2 as self-loops aren't used patients with >1 disease
        aggregate_prog = final_prog[:k]
        aggregate_prog_list.append(tuple(aggregate_prog))

    if len(final_prog) == 1:  # to get self-loops for patients with one disease
        self_loop = [x for pair in zip(final_prog, final_prog) for x in pair]
        aggregate_prog = tuple(self_loop)
        aggregate_prog_list = [aggregate_prog]
    return aggregate_prog_list


def print_hyperarc_str(dis_tup):
    """Convert a tuple of diseases and convert them to a string where the
    last comma is a right arrow.

    Args:
        dis_tup (tuple): Tuple containing diseases.

    Returns:
        str: Hyperarc string.
    """
    if len(dis_tup) > 1:
        string = ", ".join(map(str, dis_tup))
        comma_idx = string.rfind(",")
        if comma_idx != -1:
            string = f"{string[:comma_idx]} ->{string[comma_idx + 1 :]}"

    else:
        string = ", ".join(map(str, dis_tup))
        string = string + " -> " + string
    return string


def patient_maker(num_dis, num_patients, max_deg):
    """Create random fake patients directed disease trajectories for NetworkX
        visual directed hypergraph.

    Args:
        num_dis (int): Number of different disease types.
        num_patients (int): Number of patients to generate progressions for.
        max_deg (int): Maximum number of diseases a patient can have.
        #verbose (bool): If True print the final progression for each patient.
    Returns:
        list, list: Patient disease progression list (edge list) and node list
            (dis_list).
    """
    # Set seed
    random.seed(1)

    dis_list = [*auc][:num_dis]  # Get the list of diseases
    patient_list = list()
    for patient in range(num_patients):
        deg = random.randint(1, max_deg)
        indiv_path = random.sample(dis_list, k=deg)
        patient_list.append(indiv_path)

    # if verbose:
    #     # Print the final progressions of the randomly generated patients
    #     print(
    #         f"There are {num_patients} patient(s) and their final /
    # progressions are:\n"
    #     )
    final_prog_list = list()
    for patient in patient_list:
        # print(patient)
        string = ", ".join(map(str, patient))
        last_comma_idx = string.rfind(",")
        if last_comma_idx == -1:
            print(string)
            final_prog_list.append(string)
        else:
            # print(string[:last_comma_idx]+"->"+string[last_comma_idx + 1 :])
            final_prog_list.append(
                string[:last_comma_idx] + " -> " + string[last_comma_idx + 1 :]
            )
        final_prog_df = pd.DataFrame({"Final Pathway": final_prog_list})

    # Get aggregated list of progressions for a list of patients
    all_progs = list()
    for patient in patient_list:
        all_progs.append(agg_prog(patient))
    # format required for NetworkX
    edge_list = [ii for i in all_progs for ii in i]

    return edge_list, dis_list, final_prog_df, all_progs


def create_worklists(num_dis, edge_list):
    """Create binmat, conds_worklist and idx_worklist using an edge list from
        each patient.

    Args:
        num_dis (int): Number of different possible diseases.
        edge_list (list of tuples): Each tuple is an individuals trajectory of
                    diseases where the first element in the tuple is the first
                    disease, the second the second disease etc.
    Returns:
        numpy array: binmat, gives binary array for whether disease occur.
        numpy array: conds_worklist, gives the order of diseases.
        numpy array: idx_worklist, shows whether duplicates occurred (+int), or
                whether only one disease occurred [1, -1, -1].
    """
    diseases = [*auc][:num_dis]

    # binary matrix to show inclusion of a disease in a patients trajectory
    binmat = np.zeros((len(edge_list), num_dis), dtype=np.int8)
    for row_index, row in enumerate(edge_list):
        for disease in row:
            binmat[row_index, diseases.index(disease)] = 1

    # conds_worklist
    conds_worklist = np.full((len(edge_list), num_dis), -1, dtype=np.int8)
    for row_index, row in enumerate(edge_list):
        if len(set(row)) == 1:
            for disease in set(row):
                conds_worklist[row_index, 0] = diseases.index(disease)
        else:
            for col_index, disease in enumerate(row):
                conds_worklist[row_index, col_index] = diseases.index(disease)

    # idx_worklist
    # Assume no duplicates to keep examples simple for Streamlit?
    # If no duplicates just an array of -1's
    # when only one disease occurs the first column should be -2
    idx_worklist = np.full((len(edge_list), num_dis), -1, dtype=np.int8)

    for row_index, row in enumerate(edge_list):
        if len(set(row)) == 1:
            idx_worklist[row_index, 0] = -2
    return binmat, conds_worklist, idx_worklist


###############################################################################
# DRAW HYPERGRAPHS
###############################################################################
plt.rcParams["figure.figsize"] = (5, 5)  # global figsize


def get_coords(n, radius):
    coords = list()
    if n != 0:
        angle_incr = 2 * math.pi / n
        for i in range(n):
            angle = i * angle_incr
            x = radius * math.cos(angle)
            y = radius * math.sin(angle)
            coords.append((x, y))

    return coords


def remove_hyperarc_dups(edges):
    """Removes duplicates hyperedges and hyperarcs.
        Removes an element from a list if there is another
        element with the same set equivalent and the same
        last element.

    Args:
        edges (list): Hyperedges and hyperarcs/ patient trajectories.

    Returns:
        list: Hyperarc list without duplicates to enable drawing.
    """
    # Remove duplicate hyperarcs
    edges = list(set(edges))

    # Remove hyperarcs with the same heads and tails
    def check_condition(a, b):
        return set(a[:-1]) == set(b[:-1]) and a[-1] == b[-1]

    # Create a new list with unique elements
    unique_edges = []
    for item in edges:
        if not any(check_condition(item, x) for x in unique_edges):
            unique_edges.append(item)

    return unique_edges


# B-Hypergraphs drawing function
def draw_b_hypergraph(nodes, edges, tab):
    """Draw B-Hypergraphs using NetworkX.
       B-Hypergraphs meaning there can be only one head node, but an unlimited
       number of tail nodes.
       NOTE: This doesn't currently consider duplicates.

    Args:
        nodes (list): Nodes to include in the graph.
        edges (list): Edges to include in the graph, where the last node
            is the head node.
        tab (variable): Variable name of tab the graph should be produced in.
    Returns:
        pyplot: Directed hypergraph showing the fictious patients graph.
    """
    g = nx.DiGraph()
    g.add_nodes_from(nodes)

    # Remove duplicate hyperarcs that have the same tails and head
    edges = remove_hyperarc_dups(edges)

    # Calculate the edge degree for each edge
    node_degree_dict = {value: len(value) for value in edges}

    tail_edge_list = list()
    head_edge_list = list()

    for i, edge_degree in enumerate(node_degree_dict.values()):
        if edge_degree == 2:
            print(edges[i])
            g.add_edges_from([edges[i]])

        else:
            # B-hypergraphs only - so from infinite tails to one head only
            head_node = edges[i][-1]
            tail_nodes = edges[i][:-1]
            g.add_node(i)  # create a new fake node named with number i
            for tail_node in tail_nodes:
                tail_edge_list.append((tail_node, i))

            head_edge_list.append((i, head_node))

    extra_nodes = set(g.nodes) - set(nodes)

    # Get coordinates to place real nodes in a circle with a large radius
    # Extra nodes are centred around a smaller radius
    node_radius = 6
    real_node_coords = get_coords(len(nodes), node_radius)
    extra_node_coords = get_coords(len(extra_nodes), node_radius / 3)

    real_node_coords_dict = dict(zip(nodes, real_node_coords))
    extra_node_coords_dict = dict(zip(extra_nodes, extra_node_coords))

    pos_dict = {**real_node_coords_dict, **extra_node_coords_dict}

    pos = pos_dict

    # Plot true nodes
    nx.draw_networkx_nodes(g, pos, node_size=150, nodelist=nodes)

    # Draw pairwise hyperarcs
    nx.draw_networkx_edges(
        g,
        pos,
        edge_color="red",
        connectionstyle="arc3,rad=0.1",
        arrowstyle="-|>",
        width=1,
    )

    # Generate random colours based on number of hyperarcs
    random.seed(1)
    colors = [
        "#" + "".join([random.choice("0123456789ABCDEF") for j in range(6)])
        for i in range(len(edges))
    ]

    for edge_num in range(0, len(edges)):
        for head_edge in head_edge_list:
            if edge_num in head_edge:
                # Draw hyperarc heads
                nx.draw_networkx_edges(
                    g,
                    pos,
                    edge_color=colors[edge_num],
                    edgelist=[head_edge],
                    arrowstyle="-|>",
                    width=1.5,
                )
        for tail_edge in tail_edge_list:
            if edge_num in tail_edge:
                # Draw hyperarc tails
                nx.draw_networkx_edges(
                    g,
                    pos,
                    edge_color=colors[edge_num],
                    edgelist=[tail_edge],
                    connectionstyle="arc3,rad=0.4",
                    arrowstyle="-",
                    style="dashdot",
                    alpha=0.4,
                    width=2,
                )
        for extra_node in extra_nodes:
            if edge_num == extra_node:
                # Draw extra nodes with same colour
                nx.draw_networkx_nodes(
                    g,
                    pos,
                    node_size=20,
                    nodelist=[extra_node],
                    node_color=colors[edge_num],
                    node_shape="h",
                )

    # Draw labels only for true nodes
    labels = {node: str(node) for node in nodes}
    nx.draw_networkx_labels(g, pos, labels, font_size=10)
    plt.axis("off")
    tab.pyplot()


# B-Hypergraphs drawing with weights function
def draw_weight_b_hypergraph(nodes, top_n_hyparcs, tab):
    """Draw B-Hypergraphs using NetworkX.
       B-Hypergraphs meaning there can be only one head node, but an unlimited
       number of tail nodes.
       NOTE: This doesn't currently consider duplicates.

    Args:
        nodes (list): Nodes to include in the graph.
        top_n_hyparcs (DataFrame): Two column pandas DataFrame with a hyperarc
            column and weights column.
        tab (variable): Variable name of tab the graph should be produced in.
    Returns:
        pyplot: Directed hypergraph showing the fictious patients graph.
    """
    g = nx.DiGraph()
    g.add_nodes_from(nodes)
    arcs_df = top_n_hyparcs.copy()
    arcs_df["Hyperarc"] = arcs_df["Hyperarc"].str.replace(" ", "")
    arcs_df["Hyperarc"] = arcs_df["Hyperarc"].str.replace("->", ",")

    # Turn the hyperarc column into a list of edges
    edges = [arcs_df.iloc[i, 0].split(",") for i, _ in arcs_df.iterrows()]

    # Calculate the edge degree for each edge
    node_degree_dict = {idx: len(sublist) for idx, sublist in enumerate(edges)}

    tail_edge_list = list()
    head_edge_list = list()
    # Create new columns in the df which store the tail (possibly fake nodes) #
    # and head nodes
    arcs_df["Tail"] = 0
    arcs_df["Head"] = 0
    for i, edge_degree in enumerate(node_degree_dict.values()):
        if edge_degree == 2:
            print(edges[i])
            g.add_edges_from([edges[i]])
            arcs_df.loc[i, "Tail"] = edges[i][0]
            arcs_df.loc[i, "Head"] = edges[i][1]

        else:
            # B-hypergraphs only - so from infinite tails to one head only
            head_node = edges[i][-1]
            tail_nodes = edges[i][:-1]
            g.add_node(i)  # create a new fake node named with number i
            for tail_node in tail_nodes:
                tail_edge_list.append((tail_node, i))
                arcs_df.loc[i, "Tail"] = i

            head_edge_list.append((i, head_node))
            arcs_df.loc[i, "Head"] = edges[i][-1]

    extra_nodes = set(g.nodes) - set(nodes)

    # Get coordinates to place real nodes in a circle with a large radius
    # Extra nodes are centred around a smaller radius
    node_radius = 6
    real_node_coords = get_coords(len(nodes), node_radius)
    extra_node_coords = get_coords(len(extra_nodes), node_radius / 3)

    real_node_coords_dict = dict(zip(nodes, real_node_coords))
    extra_node_coords_dict = dict(zip(extra_nodes, extra_node_coords))

    pos_dict = {**real_node_coords_dict, **extra_node_coords_dict}

    pos = pos_dict

    # Plot true nodes
    nx.draw_networkx_nodes(g, pos, node_size=150, nodelist=nodes)

    # Draw pairwise hyperarcs
    nx.draw_networkx_edges(
        g,
        pos,
        edge_color="red",
        connectionstyle="arc3,rad=0.1",
        arrowstyle="-|>",
        width=1,
    )

    # Adding weight labels to the graph
    for j in range(0, len(arcs_df)):
        t = arcs_df.loc[j, "Tail"]
        h = arcs_df.loc[j, "Head"]
        w = arcs_df.loc[j, "w(h_i)"]
        g.add_edge(t, h, weight=round(w, 2))
    # tab.dataframe(arcs_df)

    # Pairwise edges
    edge_labels = {}
    for u, v, d in g.edges(data=True):
        # If bi-directional 2 degree edges exist (otherwise get blank overlap)
        if u != v and tuple(reversed((u, v))) in g.edges:
            label = f'{d["weight"]}\n\n{g.edges[(v,u)]["weight"]}'
            edge_labels[(u, v)] = label

        elif u in nodes and pos[u][0] > pos[v][0]:
            label = f'{d["weight"]}\n\n\n'
            edge_labels[(u, v)] = label

        elif u in nodes and pos[u][0] < pos[v][0]:
            label = f'\n\n{d["weight"]}'
            edge_labels[(u, v)] = label

        # Arcs from pseudo to real nodes
        elif u != v:
            label = f'{d["weight"]}'
            edge_labels[(u, v)] = label

    # Self edges (need to be seperate to get a blank background)
    self_edge_labels = {}
    for u, v, d in g.edges(data=True):
        if u == v:
            label = f'{d["weight"]}\n\n\n'
            self_edge_labels[(u, v)] = label

    # Generate random colours based on number of hyperarcs
    random.seed(1)
    colors = [
        "#" + "".join([random.choice("0123456789ABCDEF") for j in range(6)])
        for i in range(len(edges))
    ]

    for edge_num in range(0, len(edges)):
        for head_edge in head_edge_list:
            if edge_num in head_edge:
                # Draw hyperarc heads
                nx.draw_networkx_edges(
                    g,
                    pos,
                    edge_color=colors[edge_num],
                    edgelist=[head_edge],
                    arrowstyle="-|>",
                    width=1.5,
                )
        for tail_edge in tail_edge_list:
            if edge_num in tail_edge:
                # Draw hyperarc tails
                nx.draw_networkx_edges(
                    g,
                    pos,
                    edge_color=colors[edge_num],
                    edgelist=[tail_edge],
                    connectionstyle="arc3,rad=0.4",
                    arrowstyle="-",
                    style="dashdot",
                    alpha=0.4,
                    width=2,
                )
        for extra_node in extra_nodes:
            if edge_num == extra_node:
                # Draw extra nodes with same colour
                nx.draw_networkx_nodes(
                    g,
                    pos,
                    node_size=20,
                    nodelist=[extra_node],
                    node_color=colors[edge_num],
                    node_shape="h",
                )

    # Draw the edge labels as the weights
    nx.draw_networkx_edge_labels(g, pos, edge_labels=edge_labels)
    nx.draw_networkx_edge_labels(
        g,
        pos,
        edge_labels=self_edge_labels,
        font_size=8,
        font_color="blue",
        bbox=dict(alpha=0),
    )

    # Draw labels only for true nodes
    labels = {node: str(node) for node in nodes}
    nx.draw_networkx_labels(g, pos, labels, font_size=10)
    plt.axis("off")
    tab.pyplot()


# Remove duplicate Hyperedges
def remove_dup_tuples(lst):
    """
    Args:
        lst (list): List of edges (tuples) to include.

    Returns:
        list: Edge list with duplicate hyperedges removed.
    """
    unique_sets = set(frozenset(t) for t in lst)
    unique_lst = [tuple(sorted(s)) for s in unique_sets]
    return unique_lst


def draw_trans_mat_graph(nodes, all_dis_pairs, tab, transition_df):
    """Draw transition matrix graph using NetworkX.

    Args:
        nodes (list): Nodes to include in the graph.
        all_dis_pairs (list): All possible edge to include in the graph,
            where the last node is the head node.
        tab (variable): Variable name of tab the graph should be produced in.
        transition_df (dataframe): Transition transition matrix df.
    Returns:
        pyplot: Directed hypergraph showing the transition probability.
    """
    g = nx.DiGraph()
    g.add_nodes_from(nodes)

    pos = nx.circular_layout(g)

    # tab.write(all_dis_pairs)
    for i in range(0, len(all_dis_pairs)):
        tail_node = all_dis_pairs[i][0]
        head_node = all_dis_pairs[i][1]
        weight = transition_df.loc[tail_node, head_node]
        g.add_edge(tail_node, head_node, weight=round(weight, 2))

    nx.draw(
        g,
        pos,
        with_labels=True,
        connectionstyle="arc3, rad = 0.15",
        edge_color="grey",
        node_color="yellow",
        alpha=0.9,
    )

    # Pairwise edges
    edge_labels = {}
    for u, v, d in g.edges(data=True):
        if pos[u][0] > pos[v][0]:
            label = f'{d["weight"]}\n\n\n\n{g.edges[(v,u)]["weight"]}'
            edge_labels[(u, v)] = label

    nx.draw_networkx_edge_labels(
        g,
        pos,
        edge_labels=edge_labels,
        font_size=9,
        font_color="blue",
        # bbox=dict(alpha=0),
    )

    # self edges
    self_edge_labels = {}
    for u, v, d in g.edges(data=True):
        if pos[u][0] == pos[v][0]:
            label = f'\n\n\n\n\n{d["weight"]}'
            # label = f'{g.edges[v,u]["weight"]}\n\n\n\n{d["weight"]}'
            self_edge_labels[(u, v)] = label
    nx.draw_networkx_edge_labels(
        g,
        pos,
        edge_labels=self_edge_labels,
        font_size=9,
        font_color="red",
        bbox=dict(alpha=0),
    )

    plt.axis("off")
    tab.pyplot()


###############################################################################
# UNDIRECTED MATRICES AND EIGENVECTOR CALCULATIONS
###############################################################################


def pandas_inc_mat(edge_list, dis_list):
    """Create an incidence matrix for an undirected hypergraph.

    Args:
        edge_list (list of tuples): List of edges from the population.
        dis_list (list of strings): List of nodes includes.

    Returns:
        dataframe: Index is the node, the column names are the edges.
    """
    dups_removed = remove_dup_tuples(edge_list)
    selfs_removed = [t for t in dups_removed if len(t) != 1]

    num_edges = len(selfs_removed)

    inc_mat_shape = np.full((len(dis_list), num_edges), 0)
    df = pd.DataFrame(inc_mat_shape, columns=list(selfs_removed))
    df = df.set_index(pd.Index(dis_list))
    df

    for row_num, row_name in enumerate(df.index):
        for col_num, col_name in enumerate(df):
            if row_name in col_name:
                df.iloc[row_num, col_num] = 1
    return df


def node_deg_mat(edge_list, dis_list):
    """Create node degree matrix

    Args:
        edge_list (list of tuples): List of edges from the population.
        dis_list (list of strings): List of nodes includes.

    Returns:
        np.array: Diagonal matrix of node degrees (number of edges that are
        connected to the node)
    """
    # start empty dict with all nodes
    node_counts_dict = dict.fromkeys(dis_list, 0)
    dups_removed = remove_dup_tuples(edge_list)
    selfs_removed = [t for t in dups_removed if len(t) != 1]
    for ele in selfs_removed:
        for dis in ele:
            node_counts_dict[dis] = node_counts_dict.get(dis, 0) + 1

    # turn values to list
    node_counts_list = list(node_counts_dict.values())

    # turn list into diagonal matrix
    node_deg_mat = np.diag(np.array(node_counts_list))
    return node_deg_mat


def powerset(iterable):
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s) + 1))


def superset(dis_list, dis_set):
    """Get the super set of a disease set/ hyperedge.

    Args:
        dis_list (list): List of disease/node strings.
        dis_set (tuple): Disease set, tuple containing disease/node names.

    Returns:
        super_set: Super set of the disease tuple, list of tuples.
    """
    all_sets_higher_deg = []
    for i in range(len(dis_set) + 1, len(dis_list) + 1):
        combos = list(combinations(dis_list, i))
        all_sets_higher_deg.append(combos)

    super_set = list()
    not_super_set = list()
    for deg in all_sets_higher_deg:
        for tup in deg:
            super_set.append(tup)
    for item in super_set:
        count = 0
        for dis in dis_set:
            if dis in item:
                count += 1
        if count != len(dis_set):
            not_super_set.append(item)

    super_set = list(set(super_set) - set(not_super_set))
    return super_set


def create_powsupset_tab(edge_list, dis_list, undirected=True):
    # Create hyperedge, power set and super set table.
    # All the hyperedges:
    dups_removed = remove_dup_tuples(edge_list)
    if undirected:
        all_hyperedges = [t for t in dups_removed if len(t) != 1]
    else:
        all_hyperedges = [t for t in dups_removed]
        # for t in dups_removed:
        #     if len(t) == 0:
        #         t = t * 2
        #         all_hyperedges.append(tuple(t))
        #     else:
        #         all_hyperedges.append(tuple(t))

    # column one of the df should be all_hyperedges
    pow_sup_df = pd.Series(all_hyperedges).to_frame("Hyperedge")

    # column two powerset of the column to the left
    # column three = superset of the furthest left column
    pow_sup_df["Power Set"] = ""
    pow_sup_df["Super Set"] = ""

    for row_num, row_name in enumerate(pow_sup_df.index):
        for col_num, col_name in enumerate(pow_sup_df):
            if col_num == 1:
                pow_set = powerset(pow_sup_df.iloc[row_num, 0])
                # power set as list
                pow_set = list(pow_set)
                # remove first empty tuple and last tuple as this is the same
                pow_set = pow_set[1:-1]
                pow_sup_df.iloc[row_num, 1] = str(pow_set)
            if col_num == 2:
                sup_set = superset(
                    dis_set=pow_sup_df.iloc[row_num, 0], dis_list=dis_list
                )
                pow_sup_df.iloc[row_num, 2] = str(sup_set)
    return pow_sup_df


def soren_dice_create_df(edge_list, dis_list, undirected=True):
    # Weight calculation UNDIRECTED hypergraph
    # list of all patients progressive trajectories (INCLUDING SELFEDGES)
    edge_list_with_self_edge = list()
    for tup in edge_list:
        tup = [i for i in set(tup)]
        edge_list_with_self_edge.append(tuple(sorted(tup)))

    dups_removed = remove_dup_tuples(edge_list)

    if undirected:
        # ignoring selfloop hyperedges
        all_hyperedges = [tuple(t) for t in dups_removed if len(t) != 1]
    else:
        all_hyperedges = [tuple(t) for t in dups_removed]

    # create dataframe for: hyperedge | C(e_i) | Sum C(e_j) | Sum C(e_k)
    edge_weight_df = pd.Series(all_hyperedges).to_frame("Hyperedge")

    edge_weight_df["C(e_i)"] = ""
    edge_weight_df["Sum C(e_j)"] = ""
    edge_weight_df["Sum C(e_k)"] = ""
    edge_weight_df["W_e"] = ""

    # get the power set and super set table
    pow_sup_df = create_powsupset_tab(edge_list, dis_list, undirected)

    for row_num, row in enumerate(edge_weight_df.iterrows()):
        edge = edge_weight_df.iloc[row_num, 0]
        # how often does the edge appear in the list of all patient hyperedges
        C_e = edge_list_with_self_edge.count(edge)
        edge_weight_df.iloc[row_num, 1] = C_e

        # ! Sum C(e_j)
        power_set = pow_sup_df.iloc[row_num, 1]
        power_set = literal_eval(power_set)
        power_count = 0
        for power in power_set:
            power_count += edge_list_with_self_edge.count(power)
        edge_weight_df.iloc[row_num, 2] = power_count

        # ! Sum C(e_k)
        super_set = pow_sup_df.iloc[row_num, 2]
        super_set = literal_eval(super_set)
        super_count = 0
        for super in super_set:
            super_count += edge_list_with_self_edge.count(super)
        edge_weight_df.iloc[row_num, 3] = super_count

        edge_numer = C_e
        edge_denom = C_e + power_count + super_count
        soren_dice = edge_numer / edge_denom
        edge_weight_df.iloc[row_num, 4] = soren_dice

    edge_weight_df = edge_weight_df.sort_values(by=["W_e"], ascending=False)
    edge_weight_df = edge_weight_df.reset_index(drop=True)

    return edge_weight_df


def create_hyperedge_weight_df(edge_list, dis_list):
    soren_dice_df = soren_dice_create_df(edge_list, dis_list)
    # W_e matrix
    dups_removed = remove_dup_tuples(edge_list)
    # ignoring selfloop hyperedges
    all_hyperedges = [str(tuple(t)) for t in dups_removed if len(t) != 1]

    num_edges = len(all_hyperedges)
    W_e_shape = np.full((num_edges, num_edges), 0)
    df = pd.DataFrame(W_e_shape, columns=list(all_hyperedges))
    df = df.set_index(pd.Index(all_hyperedges))

    for i in range(len(df)):
        df.iloc[i, i] = soren_dice_df.iloc[i, 4]
    return df


def hnx_visual(edge_list, dis_list, tab, weight_labels=False):
    """Create a HypernetX visual representation of the undirected hypergraph
    which includes edge labels.

    Args:
        edge_list (list): List of edges in population hypergraph.
        dis_list (list): List of nodes/diseases in population hypergraph.
        tab (streamlit tab): Name of streamlit tab to display the output in.

    Returns:
        HypernetX undirected graph visualisation with weights.
    """

    if weight_labels:
        # create hypernetx graph with edges labelled with the weights
        soren_dice_df = soren_dice_create_df(edge_list, dis_list)

        dict = {}
        # get the node list (edge) and weights for each edge
        for row_num, row in soren_dice_df.iterrows():
            tup = soren_dice_df.iloc[row_num, 0]
            node_list = list(tup)
            weight = soren_dice_df.iloc[row_num, 4]
            weight = round(weight, 3)
            edge_label = str(tup) + " : " + str(weight)
            dict[edge_label] = node_list

        H = hnx.Hypergraph(dict)

    else:
        hnx_edge_list = remove_dup_tuples(edge_list)

        # Remove tuples with length == 1 to prevent self-connections
        hnx_edge_list = [t for t in hnx_edge_list if len(t) != 1]

        hnx_edge_dict = {v: k for v, k in enumerate(hnx_edge_list)}
        H = hnx.Hypergraph(hnx_edge_dict)

    hnx.draw(H)
    tab.pyplot()
