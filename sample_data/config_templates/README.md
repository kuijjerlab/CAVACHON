# Preparing Config File
The input for the CAVACHON model needs to be provided as a list of config files. An example of the minimum specification of the config files is decribed as follows:

```yaml
io:
  datadir: ${DATASET DIRECTORY}
modalities:
  - name: RNA Modality
    type: RNA
    h5ad: ${RNA H5AD File}
  - name: ATAC Modality
    type: ATAC
    h5ad: ${ATAC H5AD File}
model:
  name: CAVACHON
  components:
    - name: ATAC Component
      modalities:
        - name: ATAC
    - name: RNA Component
      conditioned_on_z_hat: 
        - ATAC Component
      modalities:  
        - name: RNA
```
Some other example templates can be found in `sample_data/config_templates`. To use the template, simply replace `${VARIABLE}` with the custom values.

&nbsp;
# Config Specification
## Inputs and Outputs
The configs for inputs and outputs are specified under the field `io`:
* `checkpointdir`:
  * required: `False`.
  * defaults: `./`
  * description: the directory for the pretrained checkpoint and the save model weights.
* `datadir`:
  * required: `False`.
  * defaults: `./`
  * description: the directory of the input datasets.
* `outdir`:
  * required: `False`.
  * defaults: `./`
  * description: the output directory.

&nbsp;
## Modalities
The configs for modalities (or data views) are specified under the field `modalities`. This is used to specified the data distribution and type of the modalities.
* `name`:
  * required: `False`.
  * defaults: `modality/{i:02d}`
  * type: `str`
  * description: the name of the modality.
* `type`:
  * required: `True`.
  * type: `str`
  * description: the type of the modality. Currently supports `'atac'` and `'rna'`.
* `dist`:
  * required: `False`.
  * type: `str`
  * defaults:
    1. `'IndependentBernoulli'` for `type:atac`.
    2. `'IndependentZeroInflatedNegativeBinomial'` for `type:rna`.
  * description: the data distribution of the modality. Currently supports `'IndependentBernoulli'` and `'IndependentZeroInflatedNegativeBinomial'` (see `cavachon/distributions` for more details).
* `h5ad`:
  * required: `False`.
  * type: `str`.
  * description: the `h5ad` file name corresponding to the modality in directory `io/datadir` (see [Inputs and Outputs](#inputs-and-outputs)). Alternatively, the data can be loaded with `mtx`, `features` and `barcodes` files specified in [Samples](#samples).
* `filters`:
  * required: `False`.
  * type: `List[FilterConfig]`
  * defaults: `List[]`
  * description: see [Filters](#filters) and `cavachon/config/FilterConfig.py` for more details.

## Filters
The filter applied to each modality. Should be provided as a list of `FilterConfig` specification. The filtering steps will be executed sequentially based on the provided order in the list. The FilterConfig specification is described as follows: 
* `step`:
  * required: `True`
  * type: `str`
  * description: type of the filtering steps. Currently supports `FilterCells`, `FilterGenes`, and `FilterQC`. (see `cavachon/filter/` for more details)
* `**kwargs`:
  * description: please replace `kwargs` with the arguments passed to `scanpy.pp.filter_cells` (for `FilterCells`), `scanpy.pp.filter_genes` (for `FilterGenes`). For `FilterQC`, please see the following example.
### Filters Examples
```yaml
filters:
  - step: FilterQC 
    qc_vars:
      - ERCC
      - MT
    filter_threshold:
      - field: n_genes_by_counts
        operator: ge
        threshold: 500
      - field: pct_counts_ERCC
        operator: le
        threshold: 0.2
      - field: pct_counts_MT
        operator: le
        threshold: 0.2
  - step: FilterGenes
    min_counts: 25
  - step: FilterGenes
    min_cells: 10
  - step: FilterCells
    min_counts: 5  
```
&nbsp;
## Samples
The configs for the samples (or experiments) files. Note that one samples can have multiple modalities (e.g. from single-cell multi-omics technology), the files of every samples will be merged into multiple modalities. 
* `name`:
  * required: `False`.
  * defaults: `sample/{i:02d}`
  * type: `str`
  * description: the name of the sample
* `modalities`:
  * required: `True`.
  * type: `List[ModalityFileConfig]`.
  * description: see [Modality Files](#modality-files) for more details.

## Modality Files
The configs for the files of a modality from **one sample**:
* `name`:
  * required: `True`.
  * type: `str`
  * description: the name of the corresponding modality. Must match one of the name specified in [Modalities](#modalities).
* `matrix`:
  * required: `True`.
  * type: `str`.
  * description: the matrix file corresponding to the modality in directory io/datadir (see [Inputs and Outputs](#inputs-and-outputs)).
* `barcodes`:
  * required: `True`.
  * type: `str`.
  * description: the barcodes file (for the anchor indices) corresponding to the modality in directory io/datadir (see [Inputs and Outputs](#inputs-and-outputs)).
* `features`:
  * required: `True`.
  * type: `str`.
  * description: the features file (e.g. gene annotations) corresponding to the modality in directory io/datadir (see [Inputs and Outputs](#inputs-and-outputs)).
* `has_headers`:
  * required: `False`
  * defaults: `False`
  * type: `bool`
  * description: whether or not the `features` and `barcodes` files have headers.
* `barcodes_colnames`:
  * required: `False`
  * type: `List[str]`.
  * description: the column names of the barcodes files (if `has_headers=False`)
* `feature_colnames`:
  * required: `False`
  * type: `List[str]`.
  * description: the column names of the features files (if `has_headers=False`)
* `batch_effect_colnames`:
  * required: `False`
  * type: `List[str]`
  * description: the column names of the batch effects that needs to be corrected.

&nbsp;
## Model
The configs for the model are specified under the field `model`:
* `name`:
  * required: `False`.
  * defaults: `CAVACHON`
  * type: `str`
  * description: the name of the model.
* `load_weights`:
  * required: `True`.
  * type: `bool`
  * description: whether or not to load the pretrained weights. If `True`, the checkpoint of the pretrained in `checkpoiontdir/model_name` will be load to the Model. See [config for IO](#inputs-and-outputs).
* `save_weights`:
  * required: `False`.
  * type: `bool`
  * description: whether or not to save the weights. If `True`, the weights will be save to `checkpoiontdir/model_name`. See [config for IO](#inputs-and-outputs).
* `components`:
  * required: `True`.
  * type: `List[ComponentConfig]`
  * description: see [Components](#components) and `cavachon/config/ComponentConfig.py` for more details.
* `training`:
  * required: `False`.
  * type: `TrainingConfig`
  * description: see [Training](#training) and `cavachon/config/TrainingConfig.py` for more details.
* `dataset`:
  * required: `False`.
  * type: `DastasetConfig`
  * description: see [Dataset](#dataset) and `cavachon/config/DatasetConfig.py` for more details.

## Components
The configs for the components in the model:
* `name`:
  * required: `False`.
  * defaults: `component/{i:02d}`
  * type: `str`
  * description: the name of the component.
* `n_encoder_layers`:
  * required: `False`.
  * defaults: `3`
  * type: `int`.
  * description: the number of hidden layers used in the encoder neural network.
* `n_latent_dims`:
  * required: `False`.
  * defaults: `5`.
  * type: `int`.
  * description: the dimensionality of the latent space.
* `n_priors`:
  * required: `False`.
  * defaults: `n_latent_dims * 2 + 1`
  * type: `int`.
  * description: the number of components of Gaussian-mixture priors used to compute KL-divergence and perform online clustering.
* `n_progressive_epochs`:
  * required: `False`.
  * defaults: `500`.
  * type: `int`.
  * description: number of progressive epochs used during the training process. The weight of the data likelihood will be scaled linearly with `epoch/n_progressive_epochs`.
* `modalities`:
  * required: `True`.
  * type: `List[Config]`
  * description: see [Modalities (in Component)](#modalities-in-component)

## Modalities (in Component)
The configs for the modalities in the component.
* `n_decoder_layers`:
  * required: `False`.
  * defaults: `3`
  * type: `int`.
  * description: the number of hidden layers used in the decoder neural network.
* `conditioned_on_z`:
  * required: `False`.
  * defaults: `List[]`
  * type: `List[str]`
  * description: the provided string in the list needs to be the name that matched to one of the specified [Components](#components). The current component will be conditionally independent with the specified components on the latent representation of the later one (exclude its ancestors). Note that the conditional independent relationships between components needs to be a **directed acyclic graph**.
* `conditioned_on_z_hat`:
  * required: `False`.
  * defaults: `List[]`
  * type: `List[str]`
  * description: the provided string in the list needs to be the name that matched to one of the specified [Components](#components). The current component will be conditionally independent with the specified components on the latent representation of the later one (include its ancestors). Note that the conditional independent relationships between components needs to be a **directed acyclic graph**.
* `save_z`:
  * required: `False`.
  * defaults: `True`.
  * type: `bool`.
  * description: whether or not to save the predicted `z` and `z_hat` to `obsm` of the modality.
* `save_x`:
  * required: `False`.
  * defaults: `True`.
  * type: `bool`.
  * description: whether or not to save the predicted `x_parameters` to `obsm` of the modality. Note that `x_parameters` will not be predicted by defaults if none of the modalities in the component set `save_x`.

## Training
The configs for the training process.
* `train`:
  * required: `False`.
  * defaults: `True`.
  * type: `bool`.
  * description: whether or not to train or finetune the model.
* `early_stop`:
  * required: `False`.
  * defaults: `True`.
  * type: `bool`.
  * description: whether or not to use early stopping when training the model. Ignored if `train=False`.
* `max_n_epochs`:
  * required: `False`.
  * defaults: `1000`.
  * type: `int`.
  * description: maxmimum number of epochs used during the training process (set globally for all components).
* `optimizer`:
  * required: `False`.
  * defaults: `Adam(learning_rate=1e-3)`
  * type: `OptimizerConfig`
  * description: see [Optimizer](#optimizer) and `cavachon/config/OptimizerConfig` for more details.

## Optimizer
The configs for the optimizer.
* `name`:
  * required: `False`.
  * defaults: `adam`
  * type: `str`
  * description: string representation for the Tensorflow Keras optimizer. See [tf.keras.optimizers](https://www.tensorflow.org/api_docs/python/tf/keras/optimizers) for more details.
* `learning_rate`:
  * required: `False`.
  * defaults: `1e-3`
  * type: `float`
  * description: learning rate for the specified optimizers .

## Dataset
The configs for the dataset.
* `batch_size`:
  * required: `False`.
  * deafults: `128`.
  * type: `int`
  * description: batch size used to train and evaluate the model. The higher the value, the more efficient the training process will be but more memory will be used.