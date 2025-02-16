Tree-of-Debate's dataset contains 100 samples. Each sample contains the following:
- Topic: a short, vague description of the theme of the two papers
- Paper #1 arXiv Link
- Paper #1 Title
- Paper #1 Abstract
- Paper #1 Introduction
- Paper #2 arXiv Link
- Paper #2 Title
- Paper #2 Abstract
- Paper #2 Introduction
- Method or Task: 0 if the papers differ in methodology (but have the same task) and 1 if the papers differ in the task (but the methodology is generally the same)
- No cite or cite: 0 if the papers do not cite each other, and 1 if the papers cite each other.

The dataset is in `tree_of_debate_dataset.tsv` and is a tab-separated file.