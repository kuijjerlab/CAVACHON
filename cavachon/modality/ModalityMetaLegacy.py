from __future__ import annotations

import os
import yaml

from cavachon.environment.Constants import Constants
from cavachon.utils.ReflectionHandler import ReflectionHandler

class ModalityMetaLegacy:
  def __init__(self, name, dist_cls, preprocess_adata, preprocess_batch, data_type, layer_no):
    self.data_type = data_type
    self.dist_cls = dist_cls
    self.preprocess_adata = preprocess_adata
    self.preprocess_batch = preprocess_batch
    self.layer_no = layer_no
    self.name = name

  def __lt__(self, other: ModalityMetaLegacy) -> bool:
    return self.layer_no < other.layer_no

  def __eq__(self, other: ModalityMetaLegacy) -> bool:
    return self.layer_no == other.layer_no

  @classmethod
  def from_default_dist(cls, name, preprocess_adata, preprocess_batch, data_type, layer_no) -> ModalityMetaLegacy:
    dist_cls_name = Constants.DEFAULT_DIST[data_type]
    dist_cls = ReflectionHandler.get_class_by_name(dist_cls_name)
    return cls(name, dist_cls, preprocess_adata, preprocess_batch, data_type, layer_no)

  @classmethod
  def from_dist_name(cls, name, dist_cls_name, preprocess_adata, preprocess_batch, data_type, layer_no) -> ModalityMetaLegacy:
    if not dist_cls_name.endswith('Wrapper'):
      dist_cls_name += 'Wrapper'
    dist_cls = ReflectionHandler.get_class_by_name(dist_cls_name)
    return cls(name, dist_cls, preprocess_adata, preprocess_batch, data_type, layer_no)

  @classmethod
  def from_meta(cls, meta: dict) -> ModalityMetaLegacy:
    name = meta.get('name', default='name')
    data_type = meta.get('data_type', default='rna')
    layer_no = meta.get('layer_no', default=1)
    dist_cls_name = meta.get('dist', default='IndependentNormalWrapper')
    preprocess_adata = meta.get('preprocess_adata', default=None)
    preprocess_batch = meta.get('preprocess_batch', default=None)
    return cls.from_dist_name(name, dist_cls_name, preprocess_adata, preprocess_batch, data_type, layer_no)