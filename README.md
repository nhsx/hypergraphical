# Hypergraphs

A library of animations explaining hypergraphs using manim.


## Installation Instructions

A detailed tutorial for installing manim can be found here:

[Manim Docs](https://docs.manim.community/en/stable/installation.html)

## Quick Guide

1. Clone the repository locally
2. Install Manim in a virtual environment (use the instructions found above)
3. Run virtual environment
   
   ```
      $ source venv/bin/activate venv
   ```
   
4. Run the following command to create the Hypergraph Example

    ```
    $ manim -qm src/hypergraphs/hypergraphs.py HypergraphExample
    ```
   
5. The mp4 will autoplay. If not please find the video in the following directory:

   ``` 
   $ media/videos/hypergraphs/720p30/
   
   ```

## TL;DC (Too Long; Didn't Clone)

Please find a colab workbook here:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/nhsx/hypergraphical/blob/workbooks/hypergraph-animations.ipynb)

----------------------

_The section below probably needs integrating in better with the animation stuff, but I've kept it seperate for now_

# Hypergraphs for Multimorbidity

This tool has been built using Streamlit, a Python app framework that can be used to create web apps. This Streamlit app allows users to explore hypergraphs in the context of multimoribidity. 

The Streamlit app explains what multimorbidity is, what hypergraphs are and why hypergraphs are useful for modelling multimoribidity.

Users can input the number of fictious patients and diseases to randomly generate hypergraphs (undirected and directed) to represent the population.

Users can follow through the examples to find out how the hypergraph-mm works by generating hyperedge and hyperarc weights for the population, followed by calculating centrality to show the importance of different diseases within the population. 

![Hypergraphs for Multimoribidity Tool](/images/streamlit_screenshot.PNG)

## Deployment (local) Instructions

To deploy the streamlit app locally we advise following the instructions below:

To clone the repository:

`git clone https:github.com/nhsx/hypergraphical`

To create a suitable environment, in the terminal run the following command:

* Build conda environment via `conda create --name hg-streamlit python=3.8`

* Activate environment `conda activate hg-streamlit`

* Install requirements via `python -m pip install -r ./requirements.txt`

To run the tool locally, open a terminal whilst in the directory containing the app and run

```bash
streamlit run streamlit_hypergraphs.py
```

Streamlit will then render the tool and display it in your default web browser at

```bash
http://localhost:8501/
```

_In the future we could deploy this app so that GitHub isn't required_