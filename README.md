# Tree-of-Debate: Multi-Persona Debate Trees Elicit Critical Thinking for Scientific Comparative Analysis
<p align="center"><img src="https://github.com/pkargupta/tree-of-debate/blob/main/figs/tods_framework.png" alt="Framework Diagram of Tree-of-Debate"/></p>

This repository contains the source code for **Tree-of-Debate: Multi-Persona Debate Trees Elicit Critical Thinking for Scientific Comparative Analysis**. This work has been accepted at [ACL 2025](https://2025.aclweb.org/).

We introduce Tree-of-Debate (ToD), a framework which converts scientific papers into LLM personas that debate their respective novelties. To emphasize structured, critical reasoning rather than focusing solely on outcomes, ToD dynamically constructs a debate tree, enabling fine-grained analysis of independent novelty arguments within scholarly articles. Through experiments on scientific literature across various domains, evaluated by expert researchers, we demonstrate that ToD generates informative arguments, effectively contrasts papers, and supports researchers in their literature review.

## Links

- [Paper](https://arxiv.org/abs/2502.14767)
- [Installation](#installation)
- [Quick Start](#quick-start)

<p align="center"><img src="https://github.com/pkargupta/tree-of-debate/blob/main/figs/example_hierarchy.png" alt="Tree-of-Debate Example" width="400"/></p>

## Installation
The code is written in Python 3.8.10. The Python dependencies are summarized in the file `requirements.txt`. You can install them like this:
```
pip install -r requirements.txt
```

## Quick Start
In order to run Tree-of-Debate, we provide the `run.sh` script. We have provided the full dataset for Tree-of-Debate in `dataset/tree_of_debate_dataset.tsv`, with a corresponding README specifying the file format. We note that the introductions for each paper are optional (only necessary for the baseline experiments).

The following are the primary arguments for Tree-of-Debate:

- `--tsv_file` $\rightarrow$ the path to the dataset in tsv file format.
- `--log` $\rightarrow$ the output directory where all logs and final output will be saved.
- `--experiment` $\rightarrow$ the type of experiment to be performed. Select one of the experiment settings: `tod` for our full Tree-of-Debate method, `single` for the single stage baseline, `two` for the two-stage baseline, `no-tree` for the No-Tree ablation study, or `no-delib` for the No-Self-Deliberation ablation study.

We provide our human evaluation instructions under the `evaluation` directory.

## 📖 Citations
Please cite the paper and star this repo if you use Tree-of-Debate and find it interesting/useful, thanks! Feel free to open an issue if you have any questions.

```bibtex
@article{kargupta2025tree,
  title={Tree-of-Debate: Multi-Persona Debate Trees Elicit Critical Thinking for Scientific Comparative Analysis},
  author={Kargupta, Priyanka and Agarwal, Ishika and August, Tal and Han, Jiawei},
  journal={arXiv preprint arXiv:2502.14767},
  year={2025}
}
```
