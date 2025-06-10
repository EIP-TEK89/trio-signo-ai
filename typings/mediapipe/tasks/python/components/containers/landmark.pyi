import dataclasses
from typing import Any

@dataclasses.dataclass
class Landmark:
  """A landmark that can have 1 to 3 dimensions.

  Use x for 1D points, (x, y) for 2D points and (x, y, z) for 3D points.

  Attributes:
    x: The x coordinate.
    y: The y coordinate.
    z: The z coordinate.
    visibility: Landmark visibility. Should stay unset if not supported. Float
      score of whether landmark is visible or occluded by other objects.
      Landmark considered as invisible also if it is not present on the screen
      (out of scene bounds). Depending on the model, visibility value is either
      a sigmoid or an argument of sigmoid.
    presence: Landmark presence. Should stay unset if not supported. Float score
      of whether landmark is present on the scene (located within scene bounds).
      Depending on the model, presence value is either a result of sigmoid or an
      argument of sigmoid function to get landmark presence probability.
    """
  x: float | None = None
  y: float | None = None
  z: float | None = None
  visibility: float | None = None
  presence: float | None = None
    
  def __init__(self, *args: Any, **kwargs: Any) -> None: ...

@dataclasses.dataclass
class NormalizedLandmark:
  """A normalized version of above Landmark proto.

  All coordinates should be within [0, 1].

  Attributes:
    x: The normalized x coordinate.
    y: The normalized y coordinate.
    z: The normalized z coordinate.
    visibility: Landmark visibility. Should stay unset if not supported. Float
      score of whether landmark is visible or occluded by other objects.
      Landmark considered as invisible also if it is not present on the screen
      (out of scene bounds). Depending on the model, visibility value is either
      a sigmoid or an argument of sigmoid.
    presence: Landmark presence. Should stay unset if not supported. Float score
      of whether landmark is present on the scene (located within scene bounds).
      Depending on the model, presence value is either a result of sigmoid or an
      argument of sigmoid function to get landmark presence probability.
  """

  x: float | None = None
  y: float | None = None
  z: float | None = None
  visibility: float | None = None
  presence: float | None = None

  def __init__(self, *args: Any, **kwargs: Any) -> None: ...

