# Molecular Multimodal Foundation Models

This repository contains the code for an exploration of molecular multimodal foundation models for molecule generation from natural language.


## Directory Structure

```
--MoMu
	--base-contrast   # original MoMu pre-training implementation
	--base-downstream   # original MoMu downstream tasks
		-- graph-retrieval   # graph retrieval task
		-- molecule-caption   # molecular captioning task
		-- molecule-generation   # molecular generation task
		-- molecule-prediction   # properties prediction task
	--data   # datasets to be downloaded
  		--contast-pretrain
			--S   # small dataset: 89 molecules <- in repo
			--XL   # full size base dataset: 15,613 molecules
				--text   # text corpus from S2ORC
				--graph   # molecule graphs from PubChem
		-- graph-retrieval   # graph retrieval datasets
		-- molecule-caption   # molecular captioning datasets
		-- molecule-generation   # molecular generation datasets
	--new-contrast   # new MoMu contrastive pre-training (WIP)
	--new-downstream   # new MoMu downstream tasks benchmarking (WIP)
	--text-preprocess   # relevance scoring to improve text retrieval (WIP)
```

## Models

**`base-MoMu`**: original model trained on contrast-XL base dataset.
- `base-contrast`: code for contrastive pre-training
- `base-downstream`: code for fine-tuning on downstream tasks

**`new-MoMu`** [WIP]: lightweight model family trained for experimental purposes. 
- `new-contrast` [WIP]: code to perform contrastive pre-training with mini-MoMum on the smaller contrast-pretrain datasets
- `new-downstream` [WIP]: code to evaluate performance on a downstream task.


## Tasks

### Contrastive pre-training

The multimodal models are pre-trained on a joint molecular graph-text corpus dataset through contrastive learning.
We then benchmark the performance of these pre-trained models by fine-tuning them on downstream tasks below.

### Graph retrieval
Task: given the name of a molecule, generating the corresponding molecular graph.

### Molecular caption
Task: given a molecule graph, generate natural language describing the molecule.

### Molecular prediction
Task: given a molecule, predict its properties.

### Molecular generation
Task: given a natural language input, generate a molecule that fits the description.


## Data

We are working on providing online access to the various datasets on HuggingFace.


## Acknowledgments

This repository builds on the original [MoMu implementation](https://github.com/ddz16/MoMu/) from Su et al. 2022 available on GitHub. Thanks to the original authors for their work!

The original implementation also uses some code from [Graphormer](https://github.com/microsoft/Graphormer/).