###############################################################################
# Libraries and Imports
###############################################################################
import streamlit as st
import pandas as pd
import numpy as np
from scipy import linalg
import itertools
from string import ascii_uppercase as auc
import networkx as nx
import matplotlib.pyplot as plt


# local
from src import build_model, centrality, centrality_utils, weight_functions
from src import numpy_utils, create_figs

##############################################################################


def tab2_directed(
    tab2, final_prog_df, dis_list, edge_list, binmat, conds_worklist, all_progs, num_dis
):
    tab2.header("Directed Hypergraph")
    tab2.subheader("_Page is Work in Progress_ 👷")
    if tab2.checkbox("Show the list of each patient's final hyperarc"):
        tab2.write(final_prog_df)
    tab2.subheader("Visual population representation:")

    # Draw b hypergraph from randomly generated patients
    numpy_utils.draw_b_hypergraph(dis_list, edge_list, tab2)
    tab2.subheader("Predecessor and Successor Disease Importance")
    tab2.markdown(
        "In this section we will go through how to calculate "
        "the importance of each disease as a predecessor "
        "or a successor, using left Eigenvector centrality on "
        "the directed hypergraph to obtain PageRank scores. "
        "Here we define a predecessor as a disease that is observed "
        "before another, and a successor is a disease that is observed "
        "after another. See overview image below."
    )

    pagerank_image = numpy_utils.add_image(
        image_path="images/PageRank.png", width=700, height=380
    )
    tab2.image(
        pagerank_image,
        caption="PageRank overview for hypergraphs.",
    )

    tab2.subheader("Hyperarc weight calcuations:")
    tab2.markdown(
        "Hyperarcs are calculated using the hyperedge calculations from the "
        "`Undirected Hypergraph` tab. From those calculations we get the "
        "following hyperedge weights (undirected):"
    )

    soren_dice_df = numpy_utils.soren_dice_create_df(
        edge_list, dis_list, undirected=False
    )

    tab2.dataframe(soren_dice_df)

    tab2.markdown(
        "Once we have the hyperedge weights we can calculate the "
        "hyperarc weights. The weighting of the hyperarcs considers disease "
        "prevalence by weighting its prevalence among other children and of "
        "its parenting hyperedge, which overall is weighted by the prevalence "
        " of the hyperedge itself."
    )

    with tab2.expander("How to calculate hyperarc weights?"):
        st.markdown("The equation:")

        st.markdown(
            r"""
                $$\mathcal{K}(h_i) = \{h_j \hspace{2pt} : \hspace{2pt} p(h_j)=p(h_i)\}$$
                """
        )

        st.markdown(
            "is the set of all the hyperarc children of the parent "
            "hyperedge $p(h_{i})$. Where $\mathcal{K}(h_i)$ is the set of "
            "siblings for hyperarc $h_{i}$. And $W(p(h_i))$ is the hyperedge "
            "weight of the parent $e_i$. Then we define the weight for hyperarc "
            "$h_i$ as"
        )

        st.markdown(
            r"""$$w(h_i) = W(p(h_i)) \frac{C(h_i)}{\sum_{h_j \in \mathcal{K}(h_i)}C(h_j)}$$"""
        )

        st.markdown(
            "Note that $W(\cdot)$ represents a _hyperedge_ "
            "weight while $w(\cdot)$ represents a "
            "_hyperarc_ weight."
        )
        examp_hyperedge = soren_dice_df.iloc[0, 0]

        examp_hyperedge_we = soren_dice_df.iloc[0, 4]

        examp_hyperarc = numpy_utils.print_hyperarc_str(examp_hyperedge)
        examp_hyperedge = ", ".join(map(str, examp_hyperedge))

        st.markdown(
            f"As an example let's calculate the hyperarc weight of hyperarc {{{examp_hyperarc}}}:"
        )

        st.markdown(
            f"The weight of the parent hyperedge {{{examp_hyperedge}}} of hyperarc {{{examp_hyperarc}}} is {round(examp_hyperedge_we, 2)}."
        )

        st.markdown(f"$W(p({{{examp_hyperarc}}}))$ = {round(examp_hyperedge_we,2)}.")
        st.markdown(
            "Then we need to count the number of times the hyperarc "
            "occurs (the raw prevalence)."
        )

        # Get a count for the number of times each of the hyperarcs appear
        hyperarcs_dups_list = list()
        for i in range(len(all_progs)):
            ind_progs = all_progs[i]
            for hyparc in range(len(ind_progs)):
                ind_prog_i = ind_progs[hyparc]
                hyperarcs_dups_list.append(tuple(ind_prog_i))

        hyparc_count_dict = {}
        for item in hyperarcs_dups_list:
            if item in hyparc_count_dict:
                hyparc_count_dict[item] += 1
            else:
                hyparc_count_dict[item] = 1

        # Convert the counts to a DataFrame
        hyperarc_count_df = pd.DataFrame(
            list(hyparc_count_dict.items()), columns=["Hyperarc", "Count"]
        )

        for i, row in hyperarc_count_df.iterrows():
            hyperarc_comma = hyperarc_count_df.iloc[i, 0]
            hyperarc_count_df.iloc[i, 0] = numpy_utils.print_hyperarc_str(
                hyperarc_comma
            )

        # Find the number of times the example hyperarc occurs
        hyperarc_count_row = hyperarc_count_df[
            hyperarc_count_df["Hyperarc"] == examp_hyperarc
        ]
        st.dataframe(hyperarc_count_row)
        if hyperarc_count_row.empty:
            examp_hyperarc_count = 0
            new_row = {
                "Hyperarc": examp_hyperarc,
                "Count": examp_hyperarc_count,
            }
            hyperarc_count_df = hyperarc_count_df.append(new_row, ignore_index=True)
        else:
            examp_hyperarc_count = hyperarc_count_row.iloc[0, 1]

        st.markdown(
            f"The raw prevalence of hyperarc {{{examp_hyperarc}}} is "
            f"{examp_hyperarc_count}."
        )

        st.markdown(f"$C({{{examp_hyperarc}}})$ = {examp_hyperarc_count}.")

        st.markdown(
            "Next we need to count the number of times the hyperarc "
            "children of the parent hyperedge occurs. In other words we need "
            "to find the siblings for the hyperarc and the number of "
            "times they occur."
        )

        st.markdown("The sibling hyperarcs are:")
        # Print of siblings
        elems = examp_hyperedge.split(", ")
        arc_siblings = list(itertools.permutations(elems, len(elems)))

        sibs_list = list()
        for hyp in arc_siblings:
            sib = numpy_utils.print_hyperarc_str(hyp)
            sibs_list.append(sib)

            st.write(sib)

        # Count of hyperarc siblings
        hyperedge_par_list = list()
        for i, hyp_edge in hyperarc_count_df.iterrows():
            arc = hyperarc_count_df.iloc[i, 0]
            par_edge = arc.replace("->", ",").replace(" ", "")
            par_edge_list = par_edge.split(",")
            par_edge_set = set(par_edge_list)
            hyperedge_par_list.append(par_edge_set)

        hyperarc_count_df["Parent Hyperedge"] = hyperedge_par_list
        hyperarc_count_df["Siblings Count"] = [None] * len(hyperarc_count_df)
        hyperarc_count_df["Parent Hyperedge Weight"] = [None] * len(hyperarc_count_df)
        hyperarc_count_df["w(h_i)"] = [None] * len(hyperarc_count_df)

        for i, row in hyperarc_count_df.iterrows():
            # Count the sum of the parent edges
            par_edge = hyperarc_count_df.iloc[i, 2]
            matching_par_rows = hyperarc_count_df[
                hyperarc_count_df["Parent Hyperedge"] == par_edge
            ]
            par_sum = matching_par_rows["Count"].sum()
            hyperarc_count_df.iloc[i, 3] = par_sum

            # Get the hyperedge weights from the soren_dice df
            soren_row = soren_dice_df[
                soren_dice_df["Hyperedge"] == tuple(sorted(par_edge))
            ]
            if len(soren_row) > 0:
                hyperarc_count_df.iloc[i, 4] = soren_row.iloc[0, 4]

        # This is the sibling count of the examp hyperarc
        examp_sib_count = hyperarc_count_df[
            hyperarc_count_df["Hyperarc"] == examp_hyperarc
        ].iloc[0, 3]

        st.markdown(f"These siblings occur {examp_sib_count} times.")

        st.markdown(r"""$$\sum_{h_j \in \mathcal{K}(h_i)}C(h_j)$$ is equal to:""")
        st.markdown(f"{examp_sib_count}")

        st.markdown("We can then calculate this hyperarc weight $w(h_{i})$ as:")
        eg_arc_we = examp_hyperedge_we * (examp_hyperarc_count / examp_sib_count)
        st.markdown(
            f"{round(examp_hyperedge_we,2)}({examp_hyperarc_count}/"
            f"{examp_sib_count})={round(eg_arc_we,2)}"
        )

        st.markdown(
            "The Table below gives the counts and weight "
            "values for all of the hyperarcs:"
        )

        hyperarc_count_df["w(h_i)"] = hyperarc_count_df["Parent Hyperedge Weight"] * (
            hyperarc_count_df["Count"] / hyperarc_count_df["Siblings Count"]
        )

        st.dataframe(hyperarc_count_df)

    #######################################################################
    # Weighted Directed Hypergraph
    #######################################################################

    # Take the top n highest hyperarc weights
    # If there are less than n hyperarc, take them all
    hyperarc_weights_df = hyperarc_count_df[["Hyperarc", "w(h_i)"]]

    hyperarc_weights_df = hyperarc_weights_df.sort_values(
        by=["w(h_i)"], ascending=False
    ).reset_index(drop=True)
    n_hyperarcs = tab2.slider(
        "Number of hyperarcs and their corresponding weights to visualise",
        min_value=1,
        max_value=20,
    )

    top_n_hyparcs = hyperarc_weights_df.iloc[:n_hyperarcs, :]
    tab2.dataframe(top_n_hyparcs)

    #######################################################################
    # RandomWalk Probability Transition Matrix
    #######################################################################

    tab2.subheader("RandomWalk Probability Transition Matrix:")

    tab2.markdown(
        "Random walks (a particular case of Markov random chain) "
        "can be used in this application with random steps being "
        "taken from one node to another, where each step is "
        "completely independent for the last step. The behaviour "
        "is determined by a transition probability matrix "
        "$\mathcal{P}$, where the column defines the start node "
        "and the row defines the end node. Here we can use random "
        "walks to find the probability of transitioning to another "
        "disease state based on a transition matrix e.g. "
        "probability of B to follow A."
    )

    with tab2.expander(
        "General equation for calculating the probability of transitioning "
        "from one node to another"
    ):
        st.markdown(
            r"""$$p(u,v) = \sum_{e \in \mathcal{E}} w(e) \frac{h(u,e)}{d(u)} \frac{h(v,e)}{\delta(e)}$$"""
        )

        st.markdown("Where")
        st.markdown("- $h(u,e) = 1$ if hyperedge $e \in E$ connects the nodes")
        st.markdown("- $d$ is the node degree function")
        st.markdown("- $\delta$ the edge degree function")

    tab2.write("#### Successor Transition Matrix")

    tab2.markdown(
        "We can find which disease is likely to be observed "
        "following other diseases. This is done by calculating "
        "the successor transition matrix. In the case of successor "
        "detection we only consider the transition from a tail to "
        "a head."
    )

    with tab2.expander("Successor Transition Matrix Equation"):
        st.markdown(
            "We can use the equation below to calculate the "
            "probability of their being a transition from node $u$ to "
            "node $v$ which can be formulated into a successor "
            "transition matrix."
        )

        st.markdown(
            r"""$$p(u,v) = \sum_{e \in \mathcal{E}} w(e) \frac{m_-(u,e)}{d_-(u)} \frac{m_+(v,e)}{\delta_+(e)}.$$"""
        )
        st.markdown("Where")
        st.markdown("- $u$ is the current node position (tail)")
        st.markdown("- $v$ is the node to be transitioned to (head)")
        st.markdown(
            "- $m_-(u,e)$ = 1 if node $u$ has a edge $e$ stemming from it "
            "(a tail is connected to node $u$)"
        )
        st.markdown(
            "- $m_+(v,e)$ = 1 if node $v$ has the head of an edge $e$ "
            "connected to it"
        )
        st.markdown("- $d_-(u)$ = the sum of all possible contributions to $u$")
        st.markdown(
            "- $\delta_+ (e)$ = the number of nodes connected to the edge "
            "$e$ via the edges head (this will always be 1 in b-hypergraphs)"
        )
        st.markdown(
            r"""- $\frac{m_-(u,e)}{d_-(u)} \frac{m_+(v,e)}{\delta_+(e)}$ is the row normaliser"""
        )

    with tab2.expander("Example - calculate of the successor transition probability"):
        st.markdown(
            "First, we need to calculate the non-normalised "
            "probability of transitioning from one node to another."
        )
        st.markdown(
            r"""This is done by taking the sum of the hyperarc weights $w(e)$ for all possible node pairs $$\sum_{e \in \mathcal{E}} w(e).$$"""
        )

        # If a hyperarc has u in the tail and v in the head then get the sum of those hyperarc weights
        # hyperarc_weights_df = hyperarc_count_df[["Hyperarc", "w(h_i)"]]
        # st.dataframe(hyperarc_weights_df)

        nn_succ_trans_df = pd.DataFrame(columns=dis_list)
        nn_succ_trans_df["Node"] = dis_list
        nn_succ_trans_df.set_index("Node", inplace=True)
        nn_succ_trans_df = nn_succ_trans_df.fillna(0)

        # get a list of all possible pairs
        all_dis_pairs = list(itertools.permutations(dis_list, 2))

        # Create self loop pairs
        for node in dis_list:
            all_dis_pairs.append([node] * 2)
        st.write(hyperarc_weights_df)
        for pair in all_dis_pairs:
            for i, row in hyperarc_weights_df.iterrows():
                tail = hyperarc_weights_df.iloc[i, 0].split("->")[0]
                head = hyperarc_weights_df.iloc[i, 0].split("->")[1]
                if pair[0] in tail and pair[1] in head:
                    nn_succ_trans_df.loc[pair[0], pair[1]] += hyperarc_weights_df.iloc[
                        i, 1
                    ]

        st.markdown("__Example__")
        st.markdown(
            f"As an example let's calculate the probability of transitioning from node {all_dis_pairs[1][0]} to {all_dis_pairs[1][1]}."
        )

        st.markdown(
            f"First we find all the hyperarcs that have {all_dis_pairs[1][0]} in their tail and {all_dis_pairs[1][1]} in their head component and their corresponding weights:"
        )

        examp_succ_hyps = list()
        examp_succ_hyp_weights = list()
        for i, row in hyperarc_weights_df.iterrows():
            tail = hyperarc_weights_df.iloc[i, 0].split("->")[0]
            head = hyperarc_weights_df.iloc[i, 0].split("->")[1]
            if all_dis_pairs[1][0] in tail and all_dis_pairs[1][1] in head:
                examp_succ_hyps.append(hyperarc_weights_df.iloc[i, 0])
                examp_succ_hyp_weights.append(hyperarc_weights_df.iloc[i, 1])

        examp_succ_df = pd.DataFrame(
            {"Hyperarc": examp_succ_hyps, "Weight": examp_succ_hyp_weights}
        )
        st.dataframe(examp_succ_df)

        st.markdown(
            f"The probability of transitioning from {all_dis_pairs[1][0]} to "
            f"{all_dis_pairs[1][1]} is the sum of the weights of these "
            f"hyperarcs: {round(examp_succ_df['Weight'].sum(), 2)}."
        )

        st.markdown("__All node pairs__")
        st.markdown(
            "If we do this for all possible node combinations we get "
            "the non-normalised successor transition matrix:"
        )

        # Make irreducible by changing 0's to 0.0001
        nn_succ_trans_df = nn_succ_trans_df.replace(0, 0.0001)
        st.dataframe(nn_succ_trans_df.round(2))

        succ_trans_df = nn_succ_trans_df.div(nn_succ_trans_df.sum(axis=1), axis=0)
        # succ_trans_df = succ_trans_df.round(2)
        st.markdown(
            "We can then use the following equation to get the "
            "normalised transition matrix:"
        )
        st.markdown(r""" $\frac{m_+(u,e)}{d_+(u)} \frac{m_-(v,e)}{\delta_-(e)}$""")

    tab2.markdown(
        "Following the steps above we get the normalised successor "
        "transition matrix, where each row sums to 1:"
    )
    tab2.dataframe(succ_trans_df.round(2))  # .style.highlight_max(axis=0))

    tab2.markdown(
        "We can show the probability of a condition being observed "
        "before another using a standard directed weighted graph:"
    )

    succ_trans_ar = succ_trans_df.to_numpy()

    col1, col2 = tab2.columns(2)  # to centre image
    with col1:
        node_labels = [*auc][:num_dis]
        numpy_utils.draw_trans_mat_graph(
            node_labels, all_dis_pairs, col1, succ_trans_df
        )

    ##############################################################################
    # PREDECCESOR TRANSITION MATRIX
    ##############################################################################
    tab2.write("#### Predecessor Transition Matrix")

    tab2.markdown(
        "We can find which disease is likely to be observed "
        "prior to other diseases. This is done by calculating "
        "the predecessor transition matrix. In the case of predecessor "
        "detection we only consider the transition from a head to "
        "a tail."
    )

    with tab2.expander("Predecessor Transition Matrix Equation"):
        st.markdown(
            "We can use the equation below to calculate the "
            "probability of their being a transition from node $u$ to "
            "node $v$ which can be formulated into a predecessor "
            "transition matrix. The equation for the predecessor probability "
            " transitions is similar to the successor equation, except the "
            "operation sign is flipped i.e. $+$ becomes $-$ and vice versa."
        )

        st.markdown(
            r"""$$p(u,v) = \sum_{e \in \mathcal{E}} w(e) \frac{m_+(u,e)}{d_+(u)} \frac{m_-(v,e)}{\delta_-(e)}.$$"""
        )
        st.markdown("Where")
        st.markdown("- $u$ is the current node position (head)")
        st.markdown("- $v$ is the node to be transitioned to (tail)")
        st.markdown(
            "- $m_+(u,e)$ = 1 if node $u$ has a edge $e$ stemming from it (a "
            " tail is connected to node $u$))"
        )
        st.markdown(
            "- $m_-(v,e)$ = 1 if node $v$ has the head of an edge $e$ "
            "connected to it"
        )
        st.markdown("- $d_+(u)$ = the sum of all possible contributions to $u$")
        st.markdown(
            "- $\delta_- (e)$ = the number of nodes connected to the edge "
            "$e$ via the edges head (this will always be 1)"
        )
        st.markdown(
            r"""- $\frac{m_+(u,e)}{d_+(u)} \frac{m_-(v,e)}{\delta_-(e)}$ is the row normaliser"""
        )

    with tab2.expander("Example - calculate of the Predecessor transition probability"):
        st.markdown(
            "First, we need to calculate the non-normalised "
            "probability of transitioning from one node to another."
        )
        st.markdown(
            r"""This is done by taking the sum of the hyperarc weights $w(e)$ for all possible node pairs $$\sum_{e \in \mathcal{E}} w(e).$$"""
        )

        nn_pred_trans_df = pd.DataFrame(columns=dis_list)
        nn_pred_trans_df["Node"] = dis_list
        nn_pred_trans_df.set_index("Node", inplace=True)
        nn_pred_trans_df = nn_pred_trans_df.fillna(0)

        for pair in all_dis_pairs:
            for i, row in hyperarc_weights_df.iterrows():
                tail = hyperarc_weights_df.iloc[i, 0].split("->")[0]
                head = hyperarc_weights_df.iloc[i, 0].split("->")[1]
                u = pair[1]
                v = pair[0]
                if u in tail and v in head:
                    nn_pred_trans_df.loc[v, u] += hyperarc_weights_df.iloc[i, 1]

        st.markdown("__Example__")
        st.markdown(
            f"As an example let's calculate the probability of transitioning from node {all_dis_pairs[1][1]} to {all_dis_pairs[1][0]}."
        )

        st.markdown(
            f"First we find all the hyperarcs that have {all_dis_pairs[1][0]}"
            f" in their tail and {all_dis_pairs[1][1]} in their head component"
            f" and their corresponding weights:"
        )

        examp_succ_hyps = list()
        examp_succ_hyp_weights = list()
        for i, row in hyperarc_weights_df.iterrows():
            tail = hyperarc_weights_df.iloc[i, 0].split("->")[0]
            head = hyperarc_weights_df.iloc[i, 0].split("->")[1]
            if all_dis_pairs[1][0] in tail and all_dis_pairs[1][1] in head:
                examp_succ_hyps.append(hyperarc_weights_df.iloc[i, 0])
                examp_succ_hyp_weights.append(hyperarc_weights_df.iloc[i, 1])

        examp_succ_df = pd.DataFrame(
            {"Hyperarc": examp_succ_hyps, "Weight": examp_succ_hyp_weights}
        )
        st.dataframe(examp_succ_df)

        st.markdown(
            f"The probability of transitioning from {all_dis_pairs[1][1]} to "
            f"{all_dis_pairs[1][0]} is the sum of the weights of these "
            f"hyperarcs: {round(examp_succ_df['Weight'].sum(), 2)}."
        )

        st.markdown("__All node pairs__")
        st.markdown(
            "If we do this for all possible node combinations we get "
            "the non-normalised predeccessor transition matrix:"
        )

        # Make irreducible by changing 0's to 0.0001
        nn_pred_trans_df = nn_pred_trans_df.replace(0, 0.0001)

        st.dataframe(nn_pred_trans_df.round(2))

        pred_trans_df = nn_pred_trans_df.div(nn_pred_trans_df.sum(axis=1), axis=0)
        # pred_trans_df = pred_trans_df.round(2)
        st.markdown(
            "We can then use the following equation to get the "
            "normalised transition matrix:"
        )
        st.markdown(r""" $\frac{m_+(u,e)}{d_+(u)} \frac{m_-(v,e)}{\delta_-(e)}$""")

    tab2.markdown(
        "Following the steps above we get the normalised predecessor "
        "transition matrix, where each row sums to 1:"
    )
    tab2.dataframe(pred_trans_df.round(2))

    tab2.markdown(
        "We can show the probability of a condition being observed "
        "before another using a standard directed weighted graph:"
    )

    col1, col2 = tab2.columns(2)  # to centre image
    with col1:
        node_labels = [*auc][:num_dis]
        numpy_utils.draw_trans_mat_graph(
            node_labels, all_dis_pairs, col1, pred_trans_df
        )

    tab2.subheader("PageRank:")

    tab2.markdown(
        "We can calculate the PageRank of the directed hypergraph for "
        "both the predecessor and successor transitions. Similarly to "
        "Eigenvector Centrality calculation in the undirected "
        "hypergraph we use the Eigenvector Centrality again but this "
        "time ensuring we use the left Eigenvector calculation as the "
        "transition matrix is not symmetrical due to the directed "
        "nature of the hypergraph. "
    )

    tab2.markdown(
        "None of the elements in the transition probability matrix "
        "should equal zero as the matrix must be irreducible, but "
        "to resolve this we can set 0 elements to a small value. "
        "Finally, we normalise the Eigenvector to so that the "
        "columns are equal to one."
    )

    tab2.write("#### Successor PageRank")

    with tab2.expander("How to calculate Successor PageRank?"):
        st.markdown(
            "To calculate the Eigen values of the Successor Transiton matrix"
            " we need to use the equation:"
        )
        st.latex("det(A - \lambda I) = 0")
        st.markdown("Where")
        st.markdown("- $A$ is the successor transition matrix.")
        st.markdown(
            "- $I$ is the equivalent order identity matrix "
            "(same shape as the transition matrix)."
        )

        st.markdown("Where the Eigen values are denoted as:")

        st.latex(f"\lambda_1, ..., \lambda_{len(succ_trans_df)}")
        st.markdown("From the transition matrix we get Eigen values:")
        succ_trans_ar = succ_trans_df.to_numpy()
        eigen_vals = linalg.eigvals(a=succ_trans_ar)
        eigen_vals = np.real(np.round(eigen_vals, 3))
        for i, value in enumerate(eigen_vals):
            st.latex(f"\lambda_{i} = {value}")

        maxvalue = max(eigen_vals)
        st.markdown(
            f"Then you take the maximum Eigen value {maxvalue}"
            " and use this to calculate the left Eigenvector. This is"
            " done by substituting $\lambda$ into the equation below:"
        )
        st.latex("X^T A = X^T \lambda")
        st.markdown("Where $X$ is the vector:")
        X = [*auc][:num_dis]
        st.write(pd.DataFrame(X))

        maxvalue_idx = int(np.where(eigen_vals == maxvalue)[0][0])

        st.markdown("From this we get the Eigenvector:")
        left_eigvec = linalg.eig(succ_trans_ar, left=True, right=False)[1][
            :, int(maxvalue_idx)
        ]

        left_eigvec = np.real(np.round(left_eigvec, 3))
        st.write(left_eigvec)
        st.markdown("And the normalised Eigenvector:")
        succ_norm_eigenvec = [(v / sum(n)) for n in [list(left_eigvec)] for v in n]
        succ_norm_eigenvec_vec = pd.DataFrame(succ_norm_eigenvec)
        st.write(succ_norm_eigenvec_vec)

    tab2.write("The PageRanks for these diseases are:")
    succ_norm_eig_df = pd.DataFrame(
        {"Successor PageRank": succ_norm_eigenvec}
    ).set_index(pd.Index(dis_list))
    tab2.dataframe(succ_norm_eig_df.style.highlight_max(axis=0, color="pink"))
    # max_idx = max(norm_eig_df.idxmax())
    # min_idx = min(norm_eig_df.idxmin())

    # tab2.markdown(
    #     f":red[{max_idx}] is the most likely to be a successor disease"
    #     f" and {min_idx} is the least likely."
    # )

    tab2.write("#### Predecessor PageRank")

    with tab2.expander("How to calculate Predecessor PageRank?"):
        st.markdown(
            "To calculate the Eigen values of the Predecessor Transiton matrix"
            " we need to use the equation:"
        )
        st.latex("det(A - \lambda I) = 0")
        st.markdown("Where")
        st.markdown("- $A$ is the predecessor transition matrix.")
        st.markdown(
            "- $I$ is the equivalent order identity matrix "
            "(same shape as the transition matrix)."
        )

        st.markdown("Where the Eigen values are denoted as:")

        st.latex(f"\lambda_1, ..., \lambda_{len(pred_trans_df)}")
        st.markdown("From the transition matrix we get Eigen values:")
        pred_trans_ar = pred_trans_df.to_numpy()
        eigen_vals = linalg.eigvals(a=pred_trans_ar)
        eigen_vals = np.real(np.round(eigen_vals, 3))
        for i, value in enumerate(eigen_vals):
            st.latex(f"\lambda_{i} = {value}")

        maxvalue = max(eigen_vals)
        st.markdown(
            f"Then you take the maximum Eigen value {maxvalue}"
            " and use this to calculate the left Eigenvector. This is"
            " done by substituting $\lambda$ into the equation below:"
        )
        st.latex("X^T A = X^T \lambda")
        st.markdown("Where $X$ is the vector:")
        X = [*auc][:num_dis]
        st.write(pd.DataFrame(X))

        maxvalue_idx = int(np.where(eigen_vals == maxvalue)[0][0])

        st.markdown("From this we get the Eigenvector:")
        left_eigvec = linalg.eig(pred_trans_ar, left=True, right=False)[1][
            :, int(maxvalue_idx)
        ]

        left_eigvec = np.real(np.round(left_eigvec, 3))
        st.write(left_eigvec)
        st.markdown("And the normalised Eigenvector:")
        pred_norm_eigenvec = [(v / sum(n)) for n in [list(left_eigvec)] for v in n]
        pred_norm_eigenvec_vec = pd.DataFrame(pred_norm_eigenvec)
        st.write(pred_norm_eigenvec_vec)

    tab2.write("The Predecessor PageRanks for these diseases are:")
    pred_norm_eig_df = pd.DataFrame(
        {"Predecessor PageRank": pred_norm_eigenvec}
    ).set_index(pd.Index(dis_list))
    tab2.dataframe(pred_norm_eig_df.style.highlight_max(axis=0, color="pink"))
    # max_idx = max(norm_eig_df.idxmax())
    # min_idx = min(norm_eig_df.idxmin())

    # tab2.markdown(
    #     f":red[{max_idx}] is the most likely to be a predecessor disease"
    #     f" and {min_idx} is the least likely."
    # )

    tab2.subheader("Successor vs Predecessor Condition")
    tab2.markdown(
        "We can compare successor and predecessor PageRank to find "
        "out which conditions are more likely to be observed before "
        "or after another condition or whether they are transitive conditions."
    )

    col1, col2 = tab2.columns(2)  # to centre image
    with col1:
        create_figs.pagerank_scatter(
            succ_norm_eig_df["Successor PageRank"],
            pred_norm_eig_df["Predecessor PageRank"],
            pred_norm_eig_df.index,
            col1,
        )
