# CAVACHON (Under Development)
**C**ell cluster **A**nalysis with **V**ariational **A**utoencoder using **C**onditional **H**ierarchy **Of** latent representio**N** is the Tensorflow implementation of the research "_Using hierarchical variational autoencoders to incorporate conditional independent priors for paired single-cell multi-omics data integration_" by PH Hsieh, RX Hsiao, T Belova, KT Ferenc, A Mathelier, R Burkholz, CY Chen, GK Sandve, ML Kuijjer. (NeurIPS LMRL Workshop 2022 Under Review)

## Installation
```
pip install -r requirements.txt
pip install -e .
```

## Input Preparation
Please refer to [config template](./sample_data/config_templates/README.md) for detail specification.

## Perform Analysis
### Model Training
```python
from cavachon.workflow import Workflow

filename = os.path.realpath('config.yaml')
workflow = Workflow(filename)
workflow.run()
```

### Cluster Analysis
```python
from cavachon.tools import ClusterAnalysis

analysis = ClusterAnalysis(workflow.multi_modalities, workflow.model)
logpy_z = analysis.compute_cluster_probability('RNA_Modality', 'RNA_Component')
knn = analysis.compute_neighbors_with_same_annotations('RNA_Modality', 'RNA_Component_cluster', 'z_RNA_Component')
```

### Differential Gene Analysis
```python
from cavachon.tools import DifferentialAnalysis

obs = workflow.multi_modalities['RNA_Component'].obs
index_a = obs[obs['RNA_Component_cluster'] == 'Cluster 1']
index_b = obs[obs['RNA_Component_cluster'] == 'Cluster 2']

analysis = DifferentialAnalysis(workflow.multi_modalities, workflow.model)
degs = analysis(index_a, index_b, 'RNA_Component', 'RNA_Modality')
```

### Interactive Visualization
```python
from cavachon.tools import InteractiveVisualization

InteractiveVisualization.latent_space(workflow.multi_modalities['RNA_Modality'], use_rep='z_RNA_Component')
InteractiveVisualization.neighbors_with_same_annotations(
  workflow.multi_modalities, workflow.multi_model, 'RNA_Modality', 'RNA_Component_cluster', 'z_RNA_Component', [5]
)
```