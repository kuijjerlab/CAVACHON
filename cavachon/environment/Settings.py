import os
from pathlib import Path

class Settings:
  root_path = Path(os.path.realpath(
      os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../")))
  test_path = Path(os.path.realpath(os.path.join(root_path, 'test')))
  src_path = Path(os.path.realpath(os.path.join(root_path, 'cavachon')))
  tmp_path = Path(os.path.realpath(os.path.join(root_path, 'tmp')))