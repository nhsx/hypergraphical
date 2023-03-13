###############################################################################
# Libraries
###############################################################################
import streamlit as st
import base64

# import networkx as nx
# import hypernetx as hnx
# import matplotlib.pyplot as plt
# import random
# import numpy as np
# import numba
# import pandas as pd

# local
from utils import *

###############################################################################
# Configure Page and Format
###############################################################################

st.set_page_config(
    page_title="Multimorbidity Calculations",
    page_icon="https://www.england.nhs.uk/wp-content/themes/nhsengland/static/img/favicon.ico",
    layout="centered",
    initial_sidebar_state="expanded",
    menu_items={
        # "Get Help": "https://www.england.nhs.uk/allocations/",
        # "Report a bug": "<github url>",
        # "About": "[https://www.england.nhs.uk/allocations/]\
        # (https://www.england.nhs.uk/allocations/)",
    },
)
padding = 1
st.markdown(
    f""" <style>
    .reportview-container .main .block-container{{
        padding-top: {padding}rem;
    }} </style> """,
    unsafe_allow_html=True,
)

st.set_option("deprecation.showPyplotGlobalUse", False)

# render svg image
def render_svg(svg):
    """Renders the given svg string."""
    b64 = base64.b64encode(svg.encode("utf-8")).decode("utf-8")
    html = r'<img src="data:image/svg+xml;base64,%s"/>' % b64
    st.write(html, unsafe_allow_html=True)


# NHS Logo
svg = """
    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 300 16">
            <path d="M0 0h40v16H0z" fill="#005EB8"></path>
            <path d="M3.9 1.5h4.4l2.6 9h.1l1.8-9h3.3l-2.8 13H9l-2.7-9h-.1l-1.8 9H1.1M17.3 1.5h3.6l-1 4.9h4L25 1.5h3.5l-2.7 13h-3.5l1.1-5.6h-4.1l-1.2 5.6h-3.4M37.7 4.4c-.7-.3-1.6-.6-2.9-.6-1.4 0-2.5.2-2.5 1.3 0 1.8 5.1 1.2 5.1 5.1 0 3.6-3.3 4.5-6.4 4.5-1.3 0-2.9-.3-4-.7l.8-2.7c.7.4 2.1.7 3.2.7s2.8-.2 2.8-1.5c0-2.1-5.1-1.3-5.1-5 0-3.4 2.9-4.4 5.8-4.4 1.6 0 3.1.2 4 .6" fill="white"></path>
          </svg>
"""
render_svg(svg)


###############################################################################
# SIDEBAR
###############################################################################

st.sidebar.subheader("Random Patient Generator")

view_choice = st.sidebar.selectbox(
    "What would you like to view?",
    (
        "Population hypergraph calculations",
        "Future disease prediction",
        "Most likely cause(s) of disease",
    ),
)


num_patients = st.sidebar.slider(
    "Number of patients to generate", min_value=1, max_value=20
)
num_dis = st.sidebar.slider("Number of diseases to generate", min_value=1, max_value=5)

if st.sidebar.checkbox("Show Maximum Number of Edges"):
    st.sidebar.write(
        # "With ",
        # num_patients,
        # "patients and ",
        "With",
        num_dis,
        "potential diseases there could be up to... ",
    )
    # Calculate the number of possible directed edges
    st.sidebar.subheader("Standard Directed Graph")

    max_edges = num_dis * (num_dis - 1)
    st.sidebar.write(
        max_edges,
        "edges in a standard directed graph (without self-loops)",
    )

    st.sidebar.write(
        max_edges + num_dis,
        "edges in a standard directed graph (with self-loops)",
    )

    # Calculate the number of possible undirected hyperedges
    st.sidebar.subheader("Hyperedges (Undirected Hypergraph)")
    max_hyperedges = N_max_hyperedges(n_diseases=num_dis)
    st.sidebar.write(
        max_hyperedges,
        "hyperedges in an undirected hypergraph (without self-loops)",
    )

    # Calculate the number of possible directed hyperedges (b-hypergraphs)
    st.sidebar.subheader("Hyperarcs (Directed Hypergraph)")
    max_hyperarcs = N_max_hyperarcs(n_diseases=num_dis, b_hyp=True)
    st.sidebar.write(
        max_hyperarcs,
        "hyperarcs in a b-hypergraph (with self-loops)",
    )

edge_list, dis_list, final_prog_df = patient_maker(
    num_dis=num_dis, num_patients=num_patients, max_deg=num_dis
)

binmat, conds_worklist, idx_worklist = create_worklists(len(dis_list), edge_list)

###############################################################################
# POPULATION HYPERGRAPH CALCULATIONS
###############################################################################

if view_choice == "Population hypergraph calculations":

    st.title("Hypergraphs for Multimorbidity")
    st.markdown("Last Updated 13th March 2023")
    st.markdown(
        "_This applet is a prototype which generates fictious patient data to show how hypergraphs can be used to explore multimorbidity._"
    )

    motivation_tab, tab1, tab2 = st.tabs(
        [
            "Why Hypergraphs for Multimorbidity",
            "Undirected Hypergraph",
            "Directed Hypergraph",
        ]
    )

    ###############################################################################
    # MOTIVATION TAB = WHY HYPERGRAPHS FOR MULTIMORBIDITY
    ###############################################################################

    # TODO: Explain multimorbidity and hypergraphs in this section
    # motivation_tab.subheader("Multimorbidity")
    display_markdown_from_file("markdown_text/mm_description.txt", motivation_tab)

    display_markdown_from_file("markdown_text/graphs.txt", motivation_tab)
    col1, col2, col3 = motivation_tab.columns(3)  # to centre image
    dir_graph_image_labelled = add_image(
        image_path="images/graph_labelled.png", width=300, height=300
    )
    col2.image(
        dir_graph_image_labelled,
        caption="Directed graph showing edge and node labelling.",
    )

    col1, col2 = motivation_tab.columns(2)  # to centre image
    undir_graph_image = add_image(
        image_path="images/undirected_graph.png", width=300, height=300
    )
    dir_graph_image = add_image(
        image_path="images/directed_graph.png", width=300, height=300
    )
    col1.image(undir_graph_image, caption="Undirected graph")
    col2.image(dir_graph_image, caption="Directed graph")

    display_markdown_from_file("markdown_text/undir_hypergraphs.txt", motivation_tab)

    col1, col2 = motivation_tab.columns(2)  # to centre image
    elastic = add_image(
        image_path="images/undirected_hyper_elastic.png", width=400, height=400
    )
    col1.image(
        elastic,
        caption="Undirected hypergraph with 'elastic band' hyperedges.",
    )

    non_elastic = add_image(
        image_path="images/undirected_hyper_nonelastic.png", width=400, height=400
    )
    col2.image(
        non_elastic,
        caption="Undirected hypergraph with hyperedges.",
    )

    display_markdown_from_file("markdown_text/dir_hypergraphs.txt", motivation_tab)

    col1, col2 = motivation_tab.columns(2)  # to centre image
    hyperarc = add_image(
        image_path="images/hyperarc_example.png", width=400, height=400
    )
    col1.image(
        hyperarc,
        caption="Labelled directed hypergraph with only one hyperarc.",
    )

    parents_sibs = add_image(
        image_path="images/siblings_parents.png", width=500, height=300
    )
    col2.image(
        parents_sibs,
        caption="",
    )

    display_markdown_from_file("markdown_text/ref_list.txt", motivation_tab)

    ###########################################################################
    # TAB1 = UNDIRECTED HYPERGRAPH
    ###########################################################################

    if tab1.checkbox("Show patients individually as undirected"):
        tab1.write(final_prog_df)

    tab1.header("Undirected Hypergraph")

    tab1.subheader("Visual representation:")
    tab1.write("Note: Self-connections are not considered with undirected.")
    if num_dis == 1:
        tab1.write(
            "There are no possible undirected hypergraphs with only one node/disease present"
        )

    draw_undirected_hypergraph(edge_list, tab1)

    tab1.subheader("Hyperedge weight calculation:")

    tab1.subheader("Eigenvector Centrality:")

    tab1.text("Most important diseases...")

    ###########################################################################
    # TAB2 = DIRECTED HYPERGRAPH
    ###########################################################################

    tab2.header("Directed Hypergraph")
    if tab2.checkbox("Show the list of each patient's final hyperarc"):
        tab2.write(final_prog_df)
    tab2.subheader("Visual representation:")

    # Draw b hypergraph from randomly generated patients
    draw_b_hypergraph(dis_list, edge_list, tab2)

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

    ###########################################################################
    # TODO: TAB3 = DUAL HYPERGRAPH
    ###########################################################################

    # tab3.subheader("Hyperedge weight calculation:")
    # tab3.subheader("Eigenvector Centrality:")
    # tab3.text("Most important sets of diseases...")


elif view_choice == "Future disease prediction":
    st.markdown("_Page under construction_ 👷")
    # TODO: Implement


elif view_choice == "Cause(s) of disease":
    st.markdown("_Page under construction_ 👷")
    # TODO: Implement
