###############################################################################
# Libraries and Imports
###############################################################################

import matplotlib.pyplot as plt
from string import ascii_uppercase as auc

# import streamlit as st
import numpy as np
import pandas as pd
from src import numpy_utils

##############################################################################
# Successor Diseases
##############################################################################


def generate_forward_prog(disease_set, hyperarc_evc, n, max_degree):
    """
    Given a disease set, generate a tree of likely disease progressions given
    the hyperarc eigenvector centrality values. n decides on the number of
    disease progessions to generate.

    Args:
        disease_set (str) : Observed disease progression. Must be of format
        "DIS1, DIS2, ..., DISn-1"

        hyperarc_evc (pd.DataFrame) : Dataframe of hyperarc eigenvector
            centrality values.

        n (int) : Number of progressions to return.

        max_degree (int) : Maximum degree disease progression to generate.
    """
    pathways = [[] for i in range(n)]
    deg = len(disease_set.split(", ")) + 1
    if deg < max_degree:
        deg_hyperarc_evc = hyperarc_evc[hyperarc_evc.Degree == deg]
        deg_dis = np.array([dis.split(" -> ")[0] for dis in deg_hyperarc_evc.Disease])
        deg_dis_hyperarc_evc = deg_hyperarc_evc.iloc[
            np.where(deg_dis == disease_set)
        ].sort_values(by="Eigenvector Centrality", ascending=False, axis=0)
        deg_progs = list(deg_dis_hyperarc_evc.iloc[:n].Disease)
        for i, prog in enumerate(deg_progs):
            pathways[i].append(prog)
            disease_set = ", ".join(prog.split(" -> "))
            prog_pathway = generate_forward_prog(
                disease_set, hyperarc_evc, 2, max_degree
            )
            if prog_pathway is not None:
                pathways[i].append(prog_pathway)
        deg += 1
    else:
        pathways = None

    return pathways


def np_inc_mat(edge_list, dis_list, tab):
    """Create an incidence matrix for an directed hypergraph.

    Args:
        edge_list (list of tuples): List of edges from the population.
        dis_list (list of strings): List of nodes includes.
        tab (Streamlit tab): Where to print on Streamlit applet.

    Returns:
        dataframe: Index is the node (tail (-) or head(+)),
            the column names are the edges.
    """
    dups_removed = list(set(edge_list))

    num_edges = len(dups_removed)

    # Take the last comma and turn it into an arrow
    # Create a tail list and a head list
    arc_list = []
    tail_list = []
    head_list = []
    for arc in dups_removed:
        string = ", ".join(map(str, arc))
        comma_idx = string.rfind(",")
        if comma_idx != -1:
            tail = string[:comma_idx].replace(" ", "")
            head = string[comma_idx + 1 :].replace(" ", "")
            tail_list.append(tail)
            head_list.append(head)
            string = f"{string[:comma_idx]} ->{string[comma_idx + 1 :]}"
        else:
            tail = string.replace(" ", "")
            head = string.replace(" ", "")
            tail_list.append(tail)
            head_list.append(head)
            string = f"{string} -> {string}"
        arc_list.append(string)

    inc_mat_shape = np.full((len(dis_list), num_edges), 0)
    head_df = pd.DataFrame(inc_mat_shape, columns=arc_list)
    head_df = head_df.set_index(pd.Index(dis_list))
    tail_df = head_df.copy()

    for row_num, row_name in enumerate(head_df.index):
        for i in range(0, len(head_list)):
            if row_name in head_list[i]:
                head_df.iloc[row_num, i] = 1

    head_df.index = [name + "+" for name in head_df.index]

    for row_num, row_name in enumerate(tail_df.index):
        for i in range(0, len(tail_list)):
            if row_name in tail_list[i]:
                tail_df.iloc[row_num, i] = 1

    tail_df.index = [name + "-" for name in tail_df.index]

    dir_inc_mat_df = pd.concat([tail_df, head_df], axis=0)

    return dir_inc_mat_df


def get_node_prev(final_prog_df, dis_list, tab):
    """Given all final progressions count how many times each node appears.
    NOTE: Self-loop nodes are only counted once and here we set them as head
    nodes for simplicity.


    Args:
        final_prog_df (dataframe): Dataframe containing the final progression
            hyperarc for each fictitious patient.

    Returns:
        dict: Dictionary containing the prevalence count for each node.
    """

    all_dir_nodes_list = []
    for string in final_prog_df["Final Pathway"].tolist():
        string = string.replace(", ", "")
        split_string = string.split("->")
        if len(string) > 1:
            tail_nodes = split_string[0]
            tail_mod = "-".join(tail_nodes)

            head_node = split_string[1]
            mod_string = f"{tail_mod},{head_node}+"
        else:
            mod_string = f"{string}+"
        all_dir_nodes_list.append(mod_string)

    # Create a list of tail and head nodes
    head_nodes_list = [dis + "+" for dis in dis_list]
    tail_nodes_list = [dis + "-" for dis in dis_list]
    nodes_list = head_nodes_list + tail_nodes_list

    dir_nodes_string = "".join(all_dir_nodes_list)

    prev_count = {string: dir_nodes_string.count(string) for string in nodes_list}

    return prev_count
