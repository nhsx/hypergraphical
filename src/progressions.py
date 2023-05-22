###############################################################################
# Libraries and Imports
###############################################################################

import matplotlib.pyplot as plt
from string import ascii_uppercase as auc
from itertools import chain, combinations

# import streamlit as st
import math
import numpy as np
import pandas as pd

##############################################################################
# Successor Diseases
##############################################################################


def generate_forward_prog(disease_set, hyperarc_evc, n, max_degree):
    """
    Given a disease set, generate a tree of likely disease progressions given the hyperarc eigenvector
    centrality values. n decides on the number of disease progessions to generate.

    Args:
        disease_set (str) : Observed disease progression. Must be of format
        "DIS1, DIS2, ..., DISn-1"

        hyperarc_evc (pd.DataFrame) : Dataframe of hyperarc eigenvector centrality values.

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
