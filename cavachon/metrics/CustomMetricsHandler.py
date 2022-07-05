from typing import List, Tuple

import tensorflow as tf

class CustomMetricsHandler:
  def __init__(self, metrics_list):
    self.metrics_list = metrics_list

  def __iter__(self):
    return iter(self.metrics_list)

  def update_state(self, y_true, y_pred, module, modality_ordered_map):
    kwargs = {
      'y_true': y_true,
      'y_pred': y_pred,
      'module': module,
      'modality_ordered_map': modality_ordered_map
    }
    for metrics in self.metrics_list:
      metrics.update_state(**kwargs)