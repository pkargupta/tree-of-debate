# Tree-of-Debate: Multi-Persona Debate Trees Elicit Critical Thinking for Scientific Comparative Analysis
<p align="center"><img src="https://github.com/pkargupta/tree-of-debate/blob/main/figs/example_hierarchy.png" alt="Tree-of-Debate Example" width="200"/></p>
<p align="center"><img src="https://github.com/pkargupta/tree-of-debate/blob/main/figs/tods_framework.png" alt="Framework Diagram of Tree-of-Debate" width="350"/></p>

This repository contains the source code for **Tree-of-Debate: Multi-Persona Debate Trees Elicit Critical Thinking for Scientific Comparative Analysis**.

## Links

- [Installation](#installation)
- [Quick Start](#quick-start)

## Installation
The code is written in Python 3.8.10. The Python dependencies are summarized in the file `requirements.txt`. You can install them like this:
```
pip install -r requirements.txt
```

## Quick Start
In order to run Tree-of-Debate, we provide the `run.sh` script. We have provided the full dataset for Tree-of-Debate in `dataset/tree_of_debate_dataset.tsv`, with a corresponding README specifying the file format. We note that the introductions for each paper are optional-- only necessary for the baseline experiments.

The following are the primary arguments for Tree-of-Debate:

- `--tsv_file` $\rightarrow$ the path to the dataset in tsv file format.
- `--log` $\rightarrow$ the output directory where all logs and final output will be saved.
- `--experiment` $\rightarrow$ the type of experiment to be performed. Select one of the experiment settings: `tod` for our full Tree-of-Debate method, `single` for the single stage baseline, `two` for the two-stage baseline, `no-tree` for the No-Tree ablation study, or `no-delib` for the No-Self-Deliberation ablation study.

We provide our human evaluation instructions under the `evaluation` directory.