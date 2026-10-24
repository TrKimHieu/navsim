"""Datasets: find and read them (``dataset``), download them (``download``), Colab / Kaggle
token (``platform``)."""

from .dataset import SPLITS, Dataset, list_datasets, load_dataset  # noqa: F401
from .download import setup  # noqa: F401
