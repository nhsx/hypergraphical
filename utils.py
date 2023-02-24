import networkx as nx
import hypernetx as hnx
import matplotlib.pyplot as plt
import random
from string import ascii_uppercase as auc
import streamlit as st
import math
import numpy as np
import numba
import pandas as pd


###############################################################################
# CALCULATE MAXIMUM HYPEREDGES AND HYPERARCS
###############################################################################


@numba.njit(fastmath=True, nogil=True)
def N_choose_k(n, k):
    """
    Numba-compiled combinatorial calculator - number of ways to choose k items
    from n items without repetition and without order.

    INPUTS:
    -------------------
        n (int) : Total number of items.

        k (int) : Number of items to choose.
    """
    # n factorial as numerator divided by k factorial multiplied by n-k
    # factorial as denominator
    # Note that n! ~ \gamma(n+1) where \gamma is the gamma function.
    numerator = math.gamma(n + 1)
    denom = math.gamma(k + 1) * math.gamma(n - k + 1)

    return numerator / denom


@numba.njit(fastmath=True, nogil=True)
def N_deg_hyperarcs(n, d, b_hyperarcs=True):
    """
    Given an n-node directed hypergraph, how many d-degree hyperarcs are there?

    INPUTS:
    -----------------
        n (int) : Number of total nodes in directed hypergraph.

        d (int) : Degree of hyperarcs to count

        b_hyperarcs (bool) : Flag to tell function whether to only count
        B-hyperarcs or all hyperarc variants (B-, BF- and F-hyperarcs).
    """
    # Estimate n choose k using math.gamma supported by numba
    no_hyperedges = int(N_choose_k(n, d))
    if b_hyperarcs:
        no_hyperarcs = d
    else:
        no_hyperarcs = 0
        for i in range(1, d):
            # numerator = int(math.gamma(d+1))
            # denom = int(math.gamma(i+1)) * int(math.gamma(d-i+1))
            # no_i_hyp = int(numerator / denom)
            no_i_hyp = int(N_choose_k(d, i))
            no_hyperarcs += no_i_hyp

    return no_hyperedges * no_hyperarcs


@numba.njit(fastmath=True, nogil=True)
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

        # Estimate n choose k using math.gamma supported by numba
        comb = N_choose_k(n_diseases, k)
        # comb = math.comb(n_diseases, k) NOT IMPLEMENTED IN NUMBA

        # Count possible hyperarcs of hyperedge degree, depending on
        # if we only count B-hyperarcs or not
        if not b_hyp:
            hyperarc_sum += (2**k - 2) * comb
        else:
            hyperarc_sum += k * comb

    return int(hyperarc_sum)


@numba.njit(fastmath=True, nogil=True)
def N_max_hyperedges(n_diseases):
    """
    Given an n-node hypergraph, how many edges of degree 2 or more are there?

    INPUTS:
    -----------------
        n_diseases (int) : Number of total nodes in hypergraph.
    """
    # Total number of hyperedges of n disease node undirected hypergraph
    no_hyperedges = 2 ** (n_diseases)

    return int(no_hyperedges)


###############################################################################
# GENERATE FAKE PATIENTS AND THEIR AGGREGATE PROGRESSIONS
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
        aggregate_prog = [x for pair in zip(final_prog, final_prog) for x in pair]
        aggregate_prog = tuple(aggregate_prog)
        aggregate_prog_list = [aggregate_prog]
    return aggregate_prog_list


def patient_maker(num_dis, num_patients, max_deg):
    """Create random fake patients directed disease trajectories for NetworkX
        visual directed hypergraph.

    Args:
        num_dis (int): Number of different disease types.
        num_patients (int): Number of patients to generate progressions for.
        max_deg (int): Maximum number of diseases a patient can have.
        #verbose (bool): If True print the final progression for each patient.
    Returns:
        list, list: Patient disease progression list (edge list) and node list (dis_list).
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
    #         f"There are {num_patients} patient(s) and their final progressions are:\n"
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
            print(string[:last_comma_idx] + " -> " + string[last_comma_idx + 1 :])
            final_prog_list.append(
                string[:last_comma_idx] + " -> " + string[last_comma_idx + 1 :]
            )
        final_prog_df = pd.DataFrame({"Final Pathway": final_prog_list})

    # Get aggregated list of progressions for a list of patients
    all_progs = list()
    for patient in patient_list:
        all_progs.append(agg_prog(patient))
    edge_list = [ii for i in all_progs for ii in i]  # format required for NetworkX

    return edge_list, dis_list, final_prog_df


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
        pyplot: Undirected hypergraph showing the fictious patients graph.
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

    pos_dict = real_node_coords_dict | extra_node_coords_dict

    pos = pos_dict

    # Plot true nodes in orange
    nx.draw_networkx_nodes(g, pos, node_size=150, nodelist=nodes, node_color="#f77f00")

    # Draw pariwise hyperarcs
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
    unique_lst = [tuple(s) for s in unique_sets]
    return unique_lst


# Undirected Hypergraphs Drawing
def draw_undirected_hypergraph(edges, tab):
    """Draw Undirected Hypergraphs using HypernetX.
       NOTE: This doesn't currently consider duplicates.

    Args:
        edges (list): Edges to include in the graph.
        tab (variable): Variable name of tab the graph should be produced in.
    Returns:
        pyplot: Undirected hypergraph showing the fictious patients graph.
    """
    hnx_edge_list = remove_dup_tuples(edges)

    # Remove tuples with length == 1 to prevent self-connections
    hnx_edge_list = [t for t in hnx_edge_list if len(t) != 1]

    hnx_edge_dict = {v: k for v, k in enumerate(hnx_edge_list)}

    H = hnx.Hypergraph(hnx_edge_dict)
    hnx.draw(H)
    tab.pyplot()
