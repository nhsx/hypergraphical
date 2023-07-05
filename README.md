# Hypergraphical

## About the project

[![status: experimental](https://github.com/GIScience/badges/raw/master/status/experimental.svg)](https://github.com/GIScience/badges#experimental) [![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)


This repository contains two components:
 - _Hypergraph Animations_
 - _Hypergraphs for Multimorbidity_

The first components contains the code to create hypergraph animations using `manim`, explaining how
hypergraphs are constructed.

The second component is a streamlit applet which provides an interactive walkthrough of calculating node centrality and
PageRank scores of hypergraphs. This code supports the Transforming Healthcare
Data with Graph-based Techniques Using SAIL DataBank project and a link to the
original project proposal can be found [here](https://nhsx.github.io/nhsx-internship-projects/).

_**Note:** Only public or fake data are shared in this repository._

⚠️ The repository uses [pre-commit](https://pre-commit.com) hooks to enforce code style using [black](https://github.com/psf/black), follows [flake8](https://github.com/PyCQA/flake8), and performs a few other checks.  See `.pre-commit-config.yaml` for more details. These hooks will also need installing locally via:

```{bash}
pre-commit autoupdate
pre-commit install
```

and then will be checked on commit.


# Hypergraph Animations

A library of animations explaining hypergraphs using `manim`.

### Installation Instructions

A detailed tutorial for installing `manim` can be found here:

[Manim Docs](https://docs.manim.community/en/stable/installation.html)

### Quick Guide

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

### TL;DC (Too Long; Didn't Clone)

Please find a colab workbook here:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/nhsx/hypergraphical/blob/workbooks/hypergraph-animations.ipynb)

----------------------


# Hypergraphs for Multimorbidity

This __prototype__ tool has been built using Streamlit, a Python app framework that can be used to create web apps. This Streamlit app allows users to explore hypergraphs in the context of multimoribidity.

> This is not an official NHS England product or service but instead is an interactive applet prototype containing ongoing work.
This __prototype__ aims only to demonstrate work that may be of interest to others.
Opinions expressed in this applet are not representative of the views of NHS England
and any content here should __not__ be regarded as official output in any form.
For more information about NHS England please visit our official
[website](https://www.england.nhs.uk/).

The Streamlit applet explains what multimorbidity is, what hypergraphs are and why hypergraphs are useful for modelling multimoribidity. This applet randomly generates a set of _fictious_ 'patients' and their disease pathways to demonstrate the use of hypergraphs in understanding multimorbidity. The
sidebar on the left of this page can be used to change the number of 'patients' to
generate and the maximum number of diseases to include in their pathways.
In changing the number of patients and diseases, the hypergraph outputs will change
and this is purposeful to enable you to observe how population alterations result in different outcomes.

Users can follow through the examples to find out how the hypergraph-mm works by generating hyperedge and hyperarc weights for the population, followed by calculating centrality to show the importance of different diseases within the population.

![Hypergraphs for Multimoribidity Tool](/images/streamlit_screenshot.PNG)

### Deployment (local) Instructions

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

### Testing

Run tests by using `pytest test_streamlit/test_edge_count.py`.

---
### Roadmap

See the repo [Issues](./Issues/) for a list of proposed features (and known problems).

### Contributing

Contributions are what make the open source community such an amazing place to learn, inspire, and create. Any contributions you make are **greatly appreciated**.

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

_See [CONTRIBUTING.md](./CONTRIBUTING.md) for detailed guidance._

### License

Unless stated otherwise, the codebase is released under [the MIT Licence][mit].
This covers both the codebase and any sample code in the documentation.

_See [LICENSE](./LICENSE) for more information._

The project specific documentation is [© Crown copyright][copyright] and available under the terms
of the [Open Government 3.0][ogl] licence.

[mit]: LICENCE
[copyright]: http://www.nationalarchives.gov.uk/information-management/re-using-public-sector-information/uk-government-licensing-framework/crown-copyright/
[ogl]: http://www.nationalarchives.gov.uk/doc/open-government-licence/version/3/

### Contact

To find out more about [DART](https://www.nhsx.nhs.uk/key-tools-and-info/nhsx-analytics-unit/) visit our [project website](https://nhsx.github.io/AnalyticsUnit/projects.html) or get in touch [here](mailto:england.tdau@nhs.net).
