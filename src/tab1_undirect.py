###############################################################################
# Libraries and Imports
###############################################################################
import streamlit as st
import pandas as pd
import numpy as np
from scipy import linalg


# from sklearn import preprocessing
from string import ascii_uppercase as auc

# local
from src import numpy_utils


def tab1_undirected(tab1, final_prog_df, num_dis, edge_list, dis_list):
    tab1.header("Undirected Hypergraph")

    tab1.markdown(
        "This page provides a walkthrough of how to construct undirected "
        "hypergraphs and how to calculate node centrality/ importance "
        "using the fictitious patients generated using the sliders "
        "on the left sidebar as the population."
    )

    tab1.markdown(
        "You can [visualise the undirected hypergraph]"
        "(#undirected-hypergraph) of this generated population. View a "
        "graphical [overview of how Eigenvector centrality]"
        "(#individual-disease-importance) is calculated on undirected graphs. "
        "Explore how to calculate the [unweighted adjacency matrix]"
        "(#hypergraph-unweighted-adjacency-matrix) or the "
        "[weighted adjacency matrix](#weighted-adjacency-matrix) for the "
        "undirected hypergraph. And you can learn how we calculated the node "
        "importance using [Eigenvector centrality](#eigenvector-centrality)."
    )

    tab1.subheader("Visual population representation:")
    tab1.markdown(
        "_Note: Self-edges are not considered with undirected"
        " hypergraphs in this case._"
    )

    if num_dis == 1:
        tab1.markdown(
            "**:red[There are no possible undirected hypergraphs with only"
            " one node/disease present. Try changing the `Number of diseases"
            " to generate` slider on the sidebar.]**"
        )
    col1, col2 = tab1.columns(2)  # to centre image
    with col1:
        numpy_utils.hnx_visual(edge_list, dis_list, col1, weight_labels=False)

    with col2:
        if col2.checkbox("Show patients individual trajectories"):
            col2.write(final_prog_df)

    tab1.subheader("Individual Disease Importance")
    tab1.markdown(
        "Here we will go through how to calculate individual"
        " disease importance to the population using Eigenvector centrality"
        " on the undirected hypergraph. See overview image below."
    )
    eigenvector_cent_image = numpy_utils.add_image(
        image_path="images/eigenvector_centrality.png", width=700, height=230
    )
    tab1.image(
        eigenvector_cent_image,
        caption="Eigenvector centrality overview for hypergraphs.",
    )

    tab1.subheader("Hypergraph Unweighted Adjacency Matrix:")

    tab1.latex("A = MM^{T} - D_n")
    tab1.markdown(
        "Where $M$ in the incidence matrix, and $D_n$ is the"
        " diagonal matrix of the node degree."
    )

    tab1.markdown("* Construct an incidence matrix $M$ for the hypergraph")
    with tab1.expander("What is an incidence matrix?"):
        st.markdown(
            "An incidence matrix is a matrix, which in the context"
            " graphs, shows which nodes $v$ connect to which edge $e$. The"
            " rows index the nodes and the columns index the edges. Where the"
            " element is $1$ if edge $e$ is incident to node $v$, otherwise"
            " the element is equal to 0.\nFor this set of randomly"
            " generated, fictious patients the incidence matrix is:"
        )

        inc_mat = numpy_utils.pandas_inc_mat(edge_list, dis_list)
        st.write(inc_mat)
        st.markdown("_Note that we don't include self edges e.g. ($A, A$_)")

    tab1.markdown(
        "* Construct the diagonal node degree matrix $D_n$ for the hypergraph"
    )
    with tab1.expander("What is a diagonal node degree matrix?"):
        st.markdown(
            "The node degree $d(v)$ defines _the number of edges that"
            " connect to a specific node_. We find the node degree of"
            " each node and create the diagonal matrix from them as"
            " as follows:"
        )
        node_deg_mat = numpy_utils.node_deg_mat(edge_list, dis_list)
        node_deg_df = pd.DataFrame(node_deg_mat, columns=dis_list)
        st.write(node_deg_df.set_index(pd.Index(dis_list)))

    tab1.markdown("* Calculate the adjacency matrix $A$")
    with tab1.expander("How to calculate the adjacency matrix?"):
        st.write("$A = MM^{T} - D_n$")
        col1, col2, col3, col4, col5 = st.columns([1.5, 6, 6, 1, 6])
        col1.write("$A$ = ")
        np_inc_mat = inc_mat.to_numpy()
        col2.write(np_inc_mat)
        col3.write(np_inc_mat.transpose())
        col4.write("$-$")
        col5.write(node_deg_mat)
        inc_matT = np.matmul(np_inc_mat, (np_inc_mat.transpose()))
        adj_mat = inc_matT - node_deg_mat

    tab1.write("$A$:")
    tab1.write(adj_mat)
    tab1.write(
        "Undirected graphs cannot have self-loops, as such the diagonal"
        " of $A$ will always be 0's, $A_{uu}=0$."
    )

    tab1.subheader("Weighted Adjacency Matrix:")
    tab1.markdown("The formula to calculate the weighted adjacency matrix is:")
    tab1.latex("A = MW_{e}M^{T} - D_n")
    tab1.markdown(
        "Where $W_e$ is the diagonal matrix of the hyperedge"
        " weights. In this example we calculate $W_e$"
        " using the Complete Modified Sorensen-Dice coefficient:"
    )

    with tab1.expander(
        "What is the Complete Modified Sorensen-Dice"
        " coefficient and how do we calculate it?"
    ):
        st.markdown(
            "The Complete Set Modified Sorensen-Dice Coefficient"
            " formula is given as:"
        )
        st.markdown(
            r"""
            $$W(e_i) = \frac{C(e_i)}{C(e_i) + \sum_{e_j \in \mathcal{P}(e_i)}w_j C(e_j) + \sum_{e_k \in \mathcal{S}(e_i)}w_k C(e_k)},$$
            """  # noqa: E501
        )
        st.markdown(
            "where\n"
            r"""$\mathcal{S}(e_i) = \{e_k \hspace{2pt} : \hspace{2pt} e_i \subset e_k\}.$"""  # noqa: E501
        )
        st.markdown(
            "For this example when we want to calculate the weight"
            " of a specific hyperedge $e_i$:\n"
        )
        st.markdown(
            "* $\mathcal{P}(e_i)$ is"  # noqa: W605
            " the power set of hyperedges for multimorbidity set"
            " $e_i$ (all subsets} disease sets)."
        )
        st.markdown(
            "* $\mathcal{S}(e_i)$ is the super set of hyperedges for"  # noqa: W605, E501
            " multimorbidity set $e_i$ (all disease sets containing"
            " $e_i$)."
        )
        st.markdown(
            "\n* :red[$C(e_i)$] is the raw"
            " prevalence count for the hyperedge/multimorbidity set $e_i$"
            " in the population\n"
        )
        st.markdown(
            "* :red[$\sum_{e_j \in \mathcal{P}(e_i)} C(e_j)$] is the"  # noqa: W605, E501
            " sum of the power set prevalence"
        )
        st.markdown(
            "* :red[$\sum_{e_k \in \mathcal{S}(e_i)} C(e_k)$] is the"  # noqa: W605, E501
            " sum of the super set prevalence"
        )
        st.markdown(
            "* :green[$w_j$] and :green[$w_k$] are optional weights which"
            " could be included to weight the power and super set"
            " differently, e.g. based on edge degree. But here we"
            " set them to 1."
        )

        st.markdown(
            "The table below shows the power set and super set for"
            " each hyperedge included in the undirected hypergraph:"
        )

        # Table with the hyperedges, power set and super set as column.
        powsupset_tab = numpy_utils.create_powsupset_tab(edge_list, dis_list)
        st.dataframe(powsupset_tab)

        st.markdown(
            "Once we have the power sets and super sets we can then"
            " calculate the weight for each hyperedge. This is done by"
            " by counting the occurence of disease sets in the patient"
            " pathways, relative to the raw prevalences, the power sets and"
            " the super sets as shown in the table below."
        )

        # NOTE: the undirected hypergraph hyperedge calcs don't include
        # single/selfedges however the directed hypergraph hyperedge calcs do
        # but this we'll use the build_model etc files for these instead?
        soren_dice_df = numpy_utils.soren_dice_create_df(edge_list, dis_list)
        st.dataframe(soren_dice_df)

    tab1.markdown(
        "$W_e$ is the diagonal matrix of size (edges, edges) (same size as"
        " the incidence matrix) with"
        " the diagonal numbers supplying the edge weights."
    )
    col1, col2 = tab1.columns(2)  # to centre image
    with col1:
        numpy_utils.hnx_visual(edge_list, dis_list, col1, weight_labels=True)
    with col2:
        col2.markdown("$W_e:$")

        we_df = numpy_utils.create_hyperedge_weight_df(edge_list, dis_list)
        we_df = we_df.round(3)
        col2.dataframe(we_df)  # .style.highlight_max(axis=0, color="grey"))

    tab1.markdown(
        "With $W_e$ we can calculate the weighted adjacency"
        " matrix $A = MW_{e}M^{T} - D_n$:"
    )

    MWe = np.matmul(np_inc_mat, we_df.to_numpy())
    MWeMT = np.matmul(MWe, (np_inc_mat.transpose()))
    weighted_adj_mat = MWeMT - node_deg_mat
    np.fill_diagonal(weighted_adj_mat, 0.0001)
    tab1.write(weighted_adj_mat)

    tab1.subheader("Eigenvector Centrality:")
    # eigenvector notebook in hypergraph-testing
    tab1.write(
        "Now that we have the adjacency matrix we can calculate the"
        " centrality of the nodes. First calculating eigenvalues and"
        " the subsequent eigenvector."
    )

    with tab1.expander("How to calculate the Eigenvector?"):
        st.markdown(
            "To calculate the Eigen values of the adjacency matrix"
            " we need to use the equation:"
        )
        st.latex("det(A - \lambda I) = 0")  # noqa: W605
        st.markdown(
            "Where $I$ is the equivalent order identity matrix"
            " (same shape as the adjacency matrix). Where the Eigen"
            " values are denoted as:"
        )
        st.latex(f"\lambda_1, ..., \lambda_{len(weighted_adj_mat)}")  # noqa: W605, E501
        st.markdown("From the adjacency matrix above we get Eigen values:")
        eigen_vals = linalg.eigvals(a=weighted_adj_mat)
        eigen_vals = np.real(np.round(eigen_vals, 3))
        for i, value in enumerate(eigen_vals):
            st.latex(f"\lambda_{i} = {value}")  # noqa: W605

        maxvalue = max(eigen_vals)
        st.markdown(
            f"Then you take the maximum Eigen value {maxvalue}"
            " and use this to calculate the left Eigenvector. This is"
            " done by substituting $\lambda$ into the equation below:"  # noqa: W605, E501
        )
        st.latex("X^T A = X^T \lambda")  # noqa: W605
        st.markdown("Where $X$ is the vector:")
        X = [*auc][:num_dis]
        st.write(pd.DataFrame(X))

        maxvalue_idx = int(np.where(eigen_vals == maxvalue)[0])

        st.markdown("From this we get the Eigenvector:")
        left_eigvec = linalg.eig(weighted_adj_mat, left=True, right=False)[1][
            :, int(maxvalue_idx)
        ]

        left_eigvec = np.round(left_eigvec, 3)
        st.write(left_eigvec)
        st.markdown("And the normalised Eigenvector:")
        norm_eigenvec = [(v / sum(n)) for n in [list(left_eigvec)] for v in n]
        norm_eigenvec_vec = pd.DataFrame(norm_eigenvec)
        st.write(norm_eigenvec_vec)

    tab1.write("Therefore the most central/important diseases are:")
    norm_eig_df = pd.DataFrame(norm_eigenvec).set_index(pd.Index(dis_list))
    tab1.dataframe(norm_eig_df.style.highlight_max(axis=0, color="pink"))
    max_idx = max(norm_eig_df.idxmax())
    min_idx = min(norm_eig_df.idxmin())

    tab1.markdown(
        f":red[{max_idx}] is the most central disease"
        f" with the most connections and {min_idx} is the least central."
    )

    tab1.markdown(
        "This page has shown how undirected hypergraphs can be "
        "used to find the most central nodes using Eigenvector "
        "centrality. This is useful for multimorbidity applications "
        "such as finding the most central diseases in a population. "
        "Whilst it is useful to see disease connectivity and importance, "
        "we may find more use in looking at which diseases progress to"
        " other diseases. For more on this visit the tab"
        " `Directed Hypergraph` at the top of this page."
    )
