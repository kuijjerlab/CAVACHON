class ConcatedTensorSpec:
  def __init__(self, name, start, end):
    self.name = name
    self.start = start
    self.end = end

  @property
  def size(self):
    return self.end - self.start