import tensorflow as tf

from anndata import AnnData
from typing import Dict
from abc import ABC, abstractstaticmethod

class PreprocessAnnData(ABC):

  @abstractstaticmethod
  def execute(anndata: AnnData, **kwargs) -> AnnData:
    pass