###############################################################################
# Libraries and Imports
###############################################################################
import streamlit as st
from src import numpy_utils


def tab0_motiv(mot_tab):
    numpy_utils.display_markdown_from_file(
        "markdown_text/overview.txt",
        mot_tab,
    )

    with mot_tab.expander("What Can You Use This Applet For?"):
        numpy_utils.display_markdown_from_file("markdown_text/purpose.txt", st)

    numpy_utils.display_markdown_from_file(
        "markdown_text/project_aims.txt",
        mot_tab,
    )
    summary = numpy_utils.add_image(
        image_path="images/summary.png", width=700, height=260
    )
    mot_tab.image(
        summary,
        caption="A summary of the different modules currently available"
        " from this work and what insights they provide"
        " us about the population.",
    )

    with mot_tab.expander("What is Multimorbidity?"):
        numpy_utils.display_markdown_from_file(
            "markdown_text/mm_description.txt",
            st,
        )

    with mot_tab.expander("Graphs Explained"):
        numpy_utils.display_markdown_from_file("markdown_text/graphs.txt", st)
        col1, col2, col3 = st.columns(3)  # to centre image
        dir_graph_image_labelled = numpy_utils.add_image(
            image_path="images/graph_labelled.png", width=300, height=300
        )
        col2.image(
            dir_graph_image_labelled,
            caption="Directed graph showing edge and node labelling.",
        )

        col1, col2 = st.columns(2)  # to centre image
        undir_graph_image = numpy_utils.add_image(
            image_path="images/undirected_graph.png", width=300, height=300
        )
        dir_graph_image = numpy_utils.add_image(
            image_path="images/directed_graph.png", width=300, height=300
        )
        col1.image(undir_graph_image, caption="Undirected graph")
        col2.image(dir_graph_image, caption="Directed graph")

        numpy_utils.display_markdown_from_file(
            "markdown_text/undir_hypergraphs.txt", st
        )

        col1, col2 = st.columns(2)  # to centre image
        elastic = numpy_utils.add_image(
            image_path="images/undirected_hyper_elastic.png",
            width=400,
            height=400,
        )
        col1.image(
            elastic,
            caption="Undirected hypergraph with 'elastic band' hyperedges.",
        )

        non_elastic = numpy_utils.add_image(
            image_path="images/undirected_hyper_nonelastic.png",
            width=400,
            height=400,
        )
        col2.image(
            non_elastic,
            caption="Undirected hypergraph with hyperedges.",
        )

        numpy_utils.display_markdown_from_file(
            "markdown_text/dir_hypergraphs.txt",
            st,
        )

        col1, col2 = st.columns(2)  # to centre image
        hyperarc = numpy_utils.add_image(
            image_path="images/hyperarc_example.png", width=400, height=400
        )
        col1.image(
            hyperarc,
            caption="Labelled directed hypergraph with only one hyperarc.",
        )

        parents_sibs = numpy_utils.add_image(
            image_path="images/siblings_parents.png", width=500, height=300
        )
        col2.image(
            parents_sibs,
            caption="Example hyperarc and corresponding sibling hyperarcs"
            " and parent hyperedge.",
        )

    numpy_utils.display_markdown_from_file(
        "markdown_text/ref_list.txt",
        mot_tab,
    )
