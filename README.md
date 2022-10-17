# CAVACHON (Under Development)
**C**ell cluster **A**nalysis with **V**ariational **A**utoencoder using **C**onditional **H**ierarchy **Of** latent representio**N** is the Tensorflow implementation of the research "_Using hierarchical variational autoencoders to incorporate conditional independent priors for paired single-cell multi-omics data integration_" by PH Hsieh, RX Hsiao, T Belova, KT Ferenc, A Mathelier, R Burkholz, CY Chen, GK Sandve, ML Kuijjer. (NeurIPS LMRL Workshop 2022 Under Review)

## Installation
```
pip install -r requirements.txt
pip install -e .
```

## Perform Analysis
### Input Preparation
Please refer to [config template](./sample_data/config_templates/README.md) for detail specification.

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

### Differentially Expressed Gene Analysis
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

## For Developers
### Implement Custom Data Likelihood
1. The custom data distribution needs to be implemented by inherent the class `cavachon.distributions.Distribution`. The custom data distribution can also inherent from the `tensorflow_probability.distributions.Distribution`. In both cases, the class function `from_parameterizer_output` will need to be implemented. This function takes an Tensor as inputs (usually the output from a neural network that outputs the parameters for the distribution) and creates a  `tensorflow_probability.distributions.Distribution` object.
2. (Optional) Implement modifiers **before** loading batch of the data. This preprocessing step is particularlly important to avoid gradient overflow if the input is not bounded (see `cavachon.dataloader.modifiers`).
3. Implement modifiers for the preprocessor **after** loading batch of the data. This preprocessing step is particularlly important to avoid gradient overflow if the input is not bounded (see `cavachon.modules.preprocessors.modifiers`).
2. (Optional) Implement the parameterizer neural network (`tf.keras.layers.Layer`)(see `cavachon.layers.parameterizers`).
3. Implement the parameterizer neural network (`tf.keras.Model`) that can optionally used the layers created from step 2. to output the parameters for the custom distributions, including all the necessary modification. This postprocessing step is particularlly important to avoid gradient overflow if the input is not bounded. The class name needs to be the same name as the class name of the custom distribution for the `cavachon.utils.ReflectionHandler` to work (see `cavachon.modules.parameterizers`)


### Implement Custom Neural Network Architecture
Create a new class that inherent either the `cavachon.model.Model` or `cavachon.modules.components.Component`. Overwrite the builder class functions with custom neural networks. Currently support the following:
* `cavachon.model.Model`
  * `setup_inputs`: inputs for Tensorflow functional API. 
  * `setup_components`: components
  * `setup_outputs`: outputs for Tensorflow functional API. 
* `cavachon.module.components.Component`
  * `setup_inputs`: inputs for Tensorflow functional API.
  * `setup_preprocessor`: preprocessor to modify the inputs before encoding.
  * `setup_encoder`: encoder networks.
  * `setup_hierarchical_encoder`: hierarchical encoder that combine latent space of ancestors (or parents).
  * `setup_z_prior_parameterizer`: parameterizer for z prior (see Jiang et al., 2016 and Falck et al., 2021)
  * `setup_z_sampler`: reparameterization for z.
  * `setup_decoders` decoder networks.
  * `setup_outputs`: outputs for Tensorflow functional API. 

### Implement Custom Losses
1. Create custom losses (`tf.keras.losses.Loss`) (see `cavachon.losses`).
2. Modify the `train_step` and `compile` function of `cavachon.models.Model` and/or `cavachon.modules.components.Component`.