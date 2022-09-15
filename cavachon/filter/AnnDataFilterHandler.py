from cavachon.filter.AnnDataFilter import AnnDataFilter
from cavachon.config.Config import Config
from cavachon.utils.ReflectionHandler import ReflectionHandler
from collections.abc import Callable
from typing import Any, Dict, List, Mapping

import anndata

class AnnDataFilterHandler(Callable):
  """AnnDataFilterHandler

  Handler for AnnDataFilter used to create a series of filter steps to
  Mapping of AnnData.

  Attributes
  ----------
  steps: Dict[str, List[AnnDataFilter]]
      filtering steps stored in dictionary. The keys are the names for
      the modality, and the values is the list the AnnDataFilters used 
      to filter the corresponding AnnData.

  """
  def __init__(self, steps: Dict[str, List[AnnDataFilter]]):
    """Constructor for AnnDataFilterHandler

    Attributes
    ----------
    steps: Dict[str, List[AnnDataFilter]]
        filtering steps stored in dictionary. The keys are the names
        for the modality, and the values is the list the AnnDataFilters 
        used to filter the corresponding AnnData.

    """
    self.steps: Dict[str, List[AnnDataFilter]] = steps

  @classmethod
  def from_config(cls, config: Mapping[str, Any]):
    """Create AnnDataFilterHandler from the config.modality_filter.

    Parameters
    ----------
    config: Mapping[str, Any]
        config.modality_filter used to create AnnDataFilterHandler.

    Returns
    -------
    AnnDataFilterHandler:
        AnnDataFilterHandler created from the config.modality_filter.

    """
    steps = dict()
    for modality_name, modality_filter_steps in config.filter.items():
      step_runners = []
      for filter_step in modality_filter_steps:
        step_runner_class = ReflectionHandler.get_class_by_name(filter_step.get('step'))
        step_runner = step_runner_class(name=filter_step.get('step'), **filter_step)
        step_runners.append(step_runner)

      steps.setdefault(modality_name, step_runners)

    return cls(steps)

  def __call__(self, target: Mapping[str, anndata.AnnData]):
    """Perform preprocessing to all AnnDatas in the provided target.

    Parameters
    ----------
    target: Mapping[str, anndata.AnnData]
        Mapping of AnnData to preprocessed.

    """
    for modality_name, modality in target.items():
      for step_runner in self.steps.get(modality_name):
        step_runner(modality)
    return