# CHANGELOG

## v0.2.1 (2024-07-02)

### Fix

* fix: attribution calculation (#14)

* add test for pruner

* move walkthrough to top-level directory

* ignore pickle files

* test caches match TL caches

* add motivation for cache dict

* fix viz

* delete incorrect test

We will produce one hook for every hook point in the model which needs to be cached
This is different from the number of src nodes in the graph

* rename file to avoid pytest issue

---------

Co-authored-by: Daniel CH Tan &lt;dtch1997@users.noreply.github.com&gt; ([`8dc8e5a`](https://github.com/dtch1997/sae-eap/commit/8dc8e5a9672690121c95336e6616b8f9ddbb4a9e))

### Unknown

* improve docs, readme ([`8bbd72e`](https://github.com/dtch1997/sae-eap/commit/8bbd72ec134a4a13cabcadb26f4b2c9a36ef29b5))

* refactor nodes ([`f4fb9f1`](https://github.com/dtch1997/sae-eap/commit/f4fb9f166822345454ebb61cd1021cb835f3304f))

## v0.2.0 (2024-06-27)

### Chore

* chore: mark test as xfail ([`2a9a15f`](https://github.com/dtch1997/sae-eap/commit/2a9a15f1d5c16c5b7cfd99adfc99297ff215cc91))

### Feature

* feat: pruning, evaluation (#11)

* wip: single-batch debugging

* single-batch in original notebook

* fix: scores not added inside loop

* wip add pruning algorithms

* fix formatting

* delete pickle files

* rafactor

* refactor

* refactor

* wip walkthrough

---------

Co-authored-by: Daniel Tan &lt;dtch1997@users.noreply.github.com&gt; ([`51d8223`](https://github.com/dtch1997/sae-eap/commit/51d8223ddc81942619192c20adb23abb6890b7ec))

### Refactor

* refactor: attribute ([`397b366`](https://github.com/dtch1997/sae-eap/commit/397b3667113616232de87c6257310cf59594ccd7))

* refactor: var names ([`a7e1904`](https://github.com/dtch1997/sae-eap/commit/a7e190450f7dcadbd3cc5dac5737771ce08d4777))

### Unknown

* add code to export attrib scores in example ([`fc84984`](https://github.com/dtch1997/sae-eap/commit/fc84984a97b8ea651b7d3567e4ba9489e9fa777a))

* mwe for run_attribution function ([`34943c1`](https://github.com/dtch1997/sae-eap/commit/34943c1ae09df3b09a68b1bd09ef4d17aa350d98))

## v0.1.1 (2024-06-24)

### Fix

* fix: attentionode get_act ([`8f69328`](https://github.com/dtch1997/sae-eap/commit/8f693287615af057cecc43dca4e63946395860c7))

### Refactor

* refactor: separate model and node cache

fix: tests, types ([`ae8b6be`](https://github.com/dtch1997/sae-eap/commit/ae8b6bead66a477bc86f7693835a8042d8df502a))

* refactor: node (#4)

* refactor graph to generic

* refactor build

* minor

* refactor index

* add data handler

* attribute  tests passing

* fix tests

---------

Co-authored-by: Daniel Tan &lt;dtch1997@users.noreply.github.com&gt; ([`44b89c7`](https://github.com/dtch1997/sae-eap/commit/44b89c73d33b1c96bb907ca303ecd345908672cc))

## v0.1.0 (2024-06-19)

### Feature

* feat: attribute (#2)

* add unit tests for graph

* minor

* WIP: add notebook to do attribution

* write test for graph indexing

* working tests for indexer

* refactor indexer: names, method names

* implement attribution function

* fix types

---------

Co-authored-by: Daniel Tan &lt;dtch1997@users.noreply.github.com&gt; ([`623412d`](https://github.com/dtch1997/sae-eap/commit/623412de7611302c4819ee7119a1d4131c271247))

### Unknown

* refactor to MultiDiGraph ([`8c479d3`](https://github.com/dtch1997/sae-eap/commit/8c479d36972fea1c3d9bc7701a4d2cdb94b488d0))

## v0.0.0 (2024-06-15)

### Chore

* chore: disable pypi ci ([`6ed0e1e`](https://github.com/dtch1997/sae-eap/commit/6ed0e1e3e7c134ff6365dc8b71982971dbc08a03))

* chore: update README ([`6b79576`](https://github.com/dtch1997/sae-eap/commit/6b79576f0249704445b560774d9fe5c59d1b5146))

### Refactor

* refactor: graph (#1)

* refactor graph

* Add minimal node, edge dataclasses

* implement minimal graph

* working build_graph

* setup graphviz in github actions

* remove py312 which has no tensordict

* fix python req

* move eap code out of library

* remove unused deps

---------

Co-authored-by: Daniel Tan &lt;dtch1997@users.noreply.github.com&gt; ([`890228b`](https://github.com/dtch1997/sae-eap/commit/890228b18c06beca68f4e2e27e27535881bb9898))

### Unknown

* minor ([`e83a1a6`](https://github.com/dtch1997/sae-eap/commit/e83a1a6ef132f425366d5637c6f639f5a3115e0e))

* EAP working ([`ee6d7cb`](https://github.com/dtch1997/sae-eap/commit/ee6d7cbee50835a80f848f69c62e67d6edc3577a))

* WIP add old SAEs ([`5ceb1a0`](https://github.com/dtch1997/sae-eap/commit/5ceb1a0a6df99cdbffd110d3ade62492d6dc6958))

* initial commit ([`92ebb7b`](https://github.com/dtch1997/sae-eap/commit/92ebb7b01d2e88014cc8c33445ce0f8a19186885))
