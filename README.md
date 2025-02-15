# Stay Focused: Problem Drift in Multi-Agent Debate

This repository contains the code for the paper "Stay Focused: Problem Drift in Multi-Agent Debate".

This repository is still under construction and subject to change.

## DRIFTEval Dataset

The human dataset DRIFTEval is available at [DriftEval.json](data/DRIFTEval/DriftEval.json). It includes both the labels and the explanations of the labels for 170 discussion excerpts.

## Usage

### Install dependencies

```bash
conda env create -f environment.yaml
```

### Run experiments

To run the code, you need the MALLM framework which is available [here](https://github.com/Multi-Agent-LLMs/mallm) and have it running. 

Experiment 1 concerns the investigation of multi-agent debate. Experiment 2 concerns the DRIFTJudge and DRIFTPolicy.

First, you need to download the datasets:

```bash
python data/data_download.py
```

Then, you can run this code with the following commands:

Run experiments:
```bash
python batch_mallm.py exp1/exp1_batch.json
python batch_mallm.py exp2/exp2_batch.json
```

Run evaluations:
```bash
python exp1_evaluation.py
python exp2_evaluation.py
```

Create figures:
```bash
python exp1_create_figures.py
python exp2_create_figures.py
```

## Citation:
```bibtex   
    comming soon
```
