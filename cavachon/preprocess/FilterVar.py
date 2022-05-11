import scanpy
from anndata import AnnData
from cavachon.preprocess.PreprocessAnnData import PreprocessAnnData

class FilterVar(PreprocessAnnData):

  @staticmethod
  def execute(adata: AnnData, **kwargs) -> AnnData:
    scanpy.pp.filter_genes(adata, **kwargs)
    return adata