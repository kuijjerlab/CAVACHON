from cavachon.filter.AnnDataFilter import AnnDataFilter
from cavachon.config.Config import Config
from cavachon.utils.ReflectionHandler import ReflectionHandler
from collections.abc import Callable
from typing import Dict, List, Mapping

import anndata

class AnnDataFilterHandler(Callable):
  def __init__(self, steps: Dict[str, List[AnnDataFilter]]):
    self.steps: Dict[str, List[AnnDataFilter]] = steps

  @classmethod
  def from_config(cls, config: Config):
    steps = dict()
    for modality_name, modality_filter_steps in config.modality_filter.items():
      step_runners = []
      for filter_step in modality_filter_steps:
        step_runner_class = ReflectionHandler.get_class_by_name(filter_step.get('step'))
        step_runner = step_runner_class(name=filter_step.get('step'), **filter_step)
        step_runners.append(step_runner)

      steps.setdefault(modality_name, step_runners)

    return cls(steps)

  def __call__(self, target: Mapping[str, anndata.AnnData]):
    for modality_name, modality in target.items():
      for step_runner in self.steps.get(modality_name):
        step_runner(modality)
    return