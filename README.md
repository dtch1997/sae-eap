# sae-eap
![Github Actions](https://github.com/dtch1997/sae-eap/actions/workflows/tests.yaml/badge.svg)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![pdm-managed](https://img.shields.io/badge/pdm-managed-blueviolet)](https://pdm-project.org)
[![Checked with pyright](https://microsoft.github.io/pyright/img/pyright_badge.svg)](https://microsoft.github.io/pyright/)

Exploratory analysis with edge attribution patching in SAEs. 

# Quickstart

## Installation

First, clone the repository: 
```bash
git clone git@github.com:dtch1997/sae-eap.git
```

Next, install PDM if you have not done so before. For detailed instructions, see [official docs](https://pdm-project.org/en/latest/#installation). 
```bash
# Install python 3.11
apt update && apt install -y python3.11 vim
# Install pdm
curl -sSL https://pdm-project.org/install-pdm.py | python3 -
echo "export PATH=/root/.local/bin:$PATH" >> ~/.bashrc
```

Then, use PDM to install dependencies. 
```
pdm install
# If you intend to develop, also install the pre-commit hooks
pdm run pre-commit install
```

Activate the virtual environment.
```
source .venv/bin/activate
```

As a sanity-check, run the tests.
```
pytest tests
```

## Usage

For an overview of current functionality, see our [walkthrough notebook](examples/walkthrough.ipynb). 

## Roadmap

Currently, the library is a work-in-progress. Here's an overview of current tasks. 

- [ ] Basic functionality
    - [x] Implement basic graph operations (`sae_eap.graph`)
    - [x] Implement basic caching operations (`sae_eap.cache`)
    - [ ] Implement graph visualization
    - [x] Implement building graph from a model (`sae_eap.graph.build`) 
    - [ ] Implement building graph from a model + SAE
- [ ] Implement circuit discovery (EAP)
    - [x] Implement computing attribution scores (`sae_eap.attribute`)
    - [ ] Test that attribution scores match original (`notebooks/test_our_attrib_matches_original.ipynb`)
    - [x] Implement pruning edges by score (`sae_eap.prune`)
- [ ] Implement circuit evaluation
    - [ ] Implement ablating a circuit (`sae_eap.ablate`)
    - [ ] Test that recovered circuits match reference implementation

# Detailed User Guide

This codebase assumes familiarity with TransformerLens. 

## Graph
For any TransformerLens model, we represent the model's computational graph as a DAG
- Each node in a graph references a TransformerLens `HookPoint`. (`sae_eap.graph.node.TensorNode`)
- There are directed edges from upstream to downstream nodes. (`sae_eap.graph.edge.TensorEdge`)
- A circuit is defined as a subgraph of the full graph. (`sae_eap.graph.graph.TensorGraph`)

### Nodes vs hooks
Note that there is not a 1-1 correspondence between nodes and hook points. Each hook point could have multiple nodes associated with it; for example, there are 8 nodes representing different attention heads, but all of these reference the same hook point: `blocks.{layer}.attn.hook_result`. As a result, each node defines a `get_act()` method which computes its activation given the corresponding model's activations. 

In general, it could also be the case that a single node needs to reference multiple hook points to compute its activation. We do not currently have any examples of this, but this could be supported in the future. 

## EAP
To compute edge scores using EAP, we do the following: 

1. Start with a clean input, corrupt input, and metric `m`. 
2. Cache the clean and corrupt activations, as well as the clean gradients. 
3. Compute all edge scores as `Attrib(Src, Dest) = (dm / dDest) * (dDest / dSrc) * (Src_corr - Src_clean)`. 
    - Remark: to compute edge scores in parallel, `dDest / dSrc` must be analytically known.  
    - For edges that start and end at "model nodes" (e.g. MLP or attention nodes), `dDest / dSrc = I`. 
    - For edges that start and end at SAE feature nodes, `dDest / dSrc` can be computed via matrix multiplication. 

# Acknowledgements

This codebase is inspired by Michael Hanna's [EAP-IG](https://github.com/hannamw/EAP-IG) repository. 