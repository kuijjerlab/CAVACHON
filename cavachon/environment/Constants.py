class Constants:

  TENSOR_NAME_X = 'matrix'
  TENSOR_NAME_ORIGIONAL_X = '_matrix'
  TENSOR_NAME_LIBSIZE = 'libsize'
  TENSOR_NAME_BATCH = 'batch_effect'

  TENSORFLOW_NAME_REGEX = r'^[A-Za-z0-9.][A-Za-z0-9_.\\/>-]*$'
  TENSORFLOW_NAME_START_REGEX = r'^[A-Za-z0-9.]'

  MODEL_INPUTS_Z_CONDITIONAL_DIMS = 'z_conditional_dims'
  MODEL_INPUTS_Z_HAT_CONDITIONAL_DIMS = 'z_hat_conditional_dims'
  MODEL_OUTPUTS_Z_HAT = 'z_hat'
  MODEL_OUTPUTS_Z = 'z'
  MODEL_OUTPUTS_Z_PARAMS = 'z_parameters'
  MODEL_OUTPUTS_X_PARAMS = 'x_parameters'
  MODEL_LOSS_KL_POSTFIX = 'kl_divergence'
  MODEL_LOSS_DATA_POSTFIX = 'negative_log_data_likelihood'

  MODULE_INPUTS_CONDITIONED_Z = 'z_conditional'
  MODULE_INPUTS_CONDITIONED_Z_HAT = 'z_hat_conditional'
  MODULE_BACKBONE = 'backbone_network'
  MODULE_R_NETWORK = 'r_network'
  MODULE_B_NETWORK = 'b_network'
  MODULE_X_PARAMETERIZER = 'x_parameterizer'
  MODULE_Z_PARAMETERIZER = 'z_parameterizer'

  SUPPORTED_MODALITY_TYPES = set(['atac', 'rna'])
  DEFAULT_MODALITY_DISTRIBUTION = dict([
      ('atac', 'IndependentBernoulli'),
      ('rna', 'IndependentZeroInflatedNegativeBinomial')
  ])

  CONFIG_FIELD_IO = 'io'
  CONFIG_FIELD_IO_CHECKPOINTDIR = 'checkpointdir'
  CONFIG_FIELD_IO_DATADIR = 'datadir'
  CONFIG_FIELD_IO_OUTDIR = 'outdir'

  CONFIG_FIELD_MODALITY = 'modalities'
  CONFIG_FIELD_MODALITY_TYPE = 'type'
  CONFIG_FIELD_MODALITY_DIST = 'dist'
  CONFIG_FIELD_MODALITY_H5AD = 'h5ad' 
  CONFIG_FIELD_MODALITY_FILTER = 'filters'
  CONFIG_FIELD_MODALITY_BATCH_COLNAMES = 'batch_effect_colnames'

  CONFIG_FIELD_MODALITY_REQUIRED = [
      'name', CONFIG_FIELD_MODALITY_TYPE
  ]

  CONFIG_FIELD_SAMPLE = 'samples'
  CONFIG_FIELD_SAMPLE_DESCRIPTION = 'description'
  CONFIG_FIELD_SAMPLE_REQUIRED = [
      'name', CONFIG_FIELD_MODALITY
  ]
  CONFIG_FIELD_SAMPLE_MODALITY_MTX = 'matrix'
  CONFIG_FIELD_SAMPLE_MODALITY_MTX_FILENAME = 'filename'
  CONFIG_FIELD_SAMPLE_MODALITY_MTX_TRANSPOSE = 'transpose'
  CONFIG_FIELD_SAMPLE_MODALITY_OBS = 'barcodes'
  CONFIG_FIELD_SAMPLE_MODALITY_VAR = 'features'
  CONFIG_FIELD_SAMPLE_MODALITY_FEATURE_FILENAME = 'filename'
  CONFIG_FIELD_SAMPLE_MODALITY_FEATURE_HAS_HEADERS = 'has_headers'
  CONFIG_FIELD_SAMPLE_MODALITY_FEATURE_HAS_HEADERS_COLNAMES = 'colnames'

  CONFIG_FIELD_SAMPLE_MODALITY_REQUIRED = [
    CONFIG_FIELD_SAMPLE_MODALITY_MTX,
    CONFIG_FIELD_SAMPLE_MODALITY_OBS,
    CONFIG_FIELD_SAMPLE_MODALITY_VAR,
  ]

  CONFIG_FIELD_MODEL = 'model'
  CONFIG_FIELD_MODEL_LOAD_WEIGHTS = 'load_weights'
  CONFIG_FIELD_MODEL_SAVE_WEIGHTS = 'save_weights'
  CONFIG_FIELD_MODEL_TRAINING = 'training'
  CONFIG_FIELD_MODEL_TRAINING_EARLY_STOPPING = 'early_stopping'
  CONFIG_FIELD_MODEL_TRAINING_OPTIMIZER = 'optimizer'
  CONFIG_FIELD_MODEL_TRAINING_N_EPOCHS = 'max_n_epochs'
  CONFIG_FIELD_MODEL_TRAINING_LEARNING_RATE = 'learning_rate'
  CONFIG_FIELD_MODEL_TRAINING_TRAIN = 'train'
  CONFIG_FIELD_MODEL_DATASET = 'dataset'
  CONFIG_FIELD_MODEL_DATASET_SHUFFLE = 'shuffle'
  CONFIG_FIELD_MODEL_DATASET_BATCHSIZE = 'batch_size'
  CONFIG_FIELD_MODEL_COMPONENT = 'components'
  CONFIG_FIELD_MODEL_REQUIRED = [
    CONFIG_FIELD_MODEL_COMPONENT
  ]
  CONFIG_FIELD_COMPONENT_MODALITY_NAMES = 'modality_names'
  CONFIG_FIELD_COMPONENT_MODALITY_DIST_NAMES = 'distribution_names'
  CONFIG_FIELD_COMPONENT_MODALITY_SAVE_X = 'save_x'
  CONFIG_FIELD_COMPONENT_MODALITY_SAVE_Z = 'save_z'
  CONFIG_FIELD_COMPONENT_CONDITION_Z = 'conditioned_on_z'
  CONFIG_FIELD_COMPONENT_CONDITION_Z_HAT = 'conditioned_on_z_hat'
  CONFIG_FIELD_COMPONENT_N_LATENT_DIMS = 'n_latent_dims'
  CONFIG_FIELD_COMPONENT_N_VARS = 'n_vars'
  CONFIG_FIELD_COMPONENT_N_VARS_BATCH = 'n_vars_batch_effect'
  CONFIG_FIELD_COMPONENT_N_PRIORS = 'n_priors'
  CONFIG_FIELD_COMPONENT_N_ENCODER_LAYERS = 'n_encoder_layers'
  CONFIG_FIELD_COMPONENT_N_DECODER_LAYERS = 'n_decoder_layers'
  CONFIG_FIELD_COMPONENT_N_PROGRESSIVE_EPOCHS = 'n_progressive_epochs'
  CONFIG_FIELD_COMPONENT_REQUIRED = [
    'name', CONFIG_FIELD_MODALITY
  ]
  CONFIG_FIELD_COMPONENT_MODALITIES_REQUIRED = [
    'name'
  ]