class Constants:
  MODALITY_TYPES = set(['atac', 'rna'])
  DEFAULT_DIST = dict([
      ('atac', 'IndependentBernoulliWrapper'),
      ('rna', 'IndependentZeroInflatedNegativeBinomialWrapper')
  ])