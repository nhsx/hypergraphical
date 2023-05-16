###############################################################################
# Libraries and Imports
###############################################################################
import os
import streamlit as st
import pandas as pd
import numpy as np
from scipy import linalg

# from sklearn import preprocessing
from string import ascii_uppercase as auc

# local
from src import build_model, centrality, centrality_utils, weight_functions
from src import numpy_utils

##############################################################################


def tab2_directed(tab2, final_prog_df, dis_list, edge_list):
    tab2.header("Directed Hypergraph")
    tab2.subheader("_Page is Work in Progress_ 👷")
    if tab2.checkbox("Show the list of each patient's final hyperarc"):
        tab2.write(final_prog_df)
    tab2.subheader("Visual population representation:")

    # Draw b hypergraph from randomly generated patients
    numpy_utils.draw_b_hypergraph(dis_list, edge_list, tab2)

    pagerank_image = numpy_utils.add_image(
        image_path="images/PageRank.png", width=700, height=380
    )
    tab2.image(
        pagerank_image,
        caption="PageRank overview for hypergraphs.",
    )

    tab2.subheader("Hyperarc weight calcuations:")
    tab2.markdown(
        "Hyperarcs are calculated using the hyperedge calculations from the \
            `Undirected Hypergraph` tab"
    )

    tab2.subheader("RandomWalk Probability Transition Matrix:")

    tab2.write("#### Successor Transition Matrix")

    tab2.write("#### Predecessor Transition Matrix")

    tab2.subheader("PageRank:")

    tab2.write("#### Successor PageRank")
    tab2.text("Disease most likely to be successor diseases...")

    tab2.write("#### Predecessor PageRank")
    tab2.text("Disease most likely to be predecessor diseases...")
