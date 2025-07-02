from typing import Optional, Any
import dataclasses

@dataclasses.dataclass
class Category:
  """A classification category.

  Category is a util class, contains a label, its display name, a float
  value as score, and the index of the label in the corresponding label file.
  Typically it's used as the result of classification tasks.

  Attributes:
    index: The index of the label in the corresponding label file.
    score: The probability score of this label category.
    display_name: The display name of the label, which may be translated for
      different locales. For example, a label, "apple", may be translated into
      Spanish for display purpose, so that the `display_name` is "manzana".
    category_name: The label of this category object.
  """

  index: Optional[int] = None
  score: Optional[float] = None
  display_name: Optional[str] = None
  category_name: Optional[str] = None

  def __init__(self, *args: Any, **kwargs: Any) -> None: ...

