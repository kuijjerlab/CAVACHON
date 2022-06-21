from collections.abc import Mapping
from typing import Any, List, Tuple
import copy

class GeneralUtils:
  
  @staticmethod
  def are_all_fields_in_mapping(
      key_list: List[Any],
      mapping: Mapping) -> Tuple[bool, List[Any]]:
    """Check if all the keys are in the Mapping

    Args:
        key_list (List[Any]): the keys to be evaluated.
        
        mapping (Mapping): the mapping to be evaluated.

    Returns:
        Tuple[bool, List[Any]]: the first element is whether the all the
        keys are in the Mapping. The second element is a list of keys 
        that do not exist in the Mappipng.
    """
    key_not_exist = []
    for key in key_list:
      if key not in mapping:
        key_not_exist.append(key)
    
    return len(key_not_exist) == 0, key_not_exist

  @staticmethod
  def duplicate_obj_to_list(obj: Any, n_objs: int) -> List[Any]:
    """Duplicate any object into a list of the same duplicated objects.

    Args:
        obj (Any): object to be duplicated.

        n_objs (int): number of duplication.

    Returns:
        List[Any]: list of duplicated objects.
    """
    res = list()
    for i in range(0, n_objs):
      res.append(copy.deepcopy(obj))
    return res