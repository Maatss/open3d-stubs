"""
This type stub file was generated by pyright.
"""

from abc import ABC, abstractmethod

log = ...
class BaseDataset(ABC):
    """The base dataset class that is used by all other datasets.

    All datasets must inherit from this class and implement the functions in order to be
    compatible with pipelines.

    Args:
        **kwargs: The configuration of the model as keyword arguments.

    Attributes:
        cfg: The configuration file as Config object that stores the keyword
            arguments that were passed to the constructor.
        name: The name of the dataset.

    **Example:**
        This example shows a custom dataset that inherit from the base_dataset class:

            from .base_dataset import BaseDataset

            class MyDataset(BaseDataset):
            def __init__(self,
                 dataset_path,
                 name='CustomDataset',
                 cache_dir='./logs/cache',
                 use_cache=False,
                 num_points=65536,
                 class_weights=[],
                 test_result_folder='./test',
                 val_files=['Custom.ply'],
                 **kwargs):
    """
    def __init__(self, **kwargs) -> None:
        """Initialize the class by passing the dataset path."""
        ...
    
    @staticmethod
    @abstractmethod
    def get_label_to_names(): # -> None:
        """Returns a label to names dictonary object.

        Returns:
            A dict where keys are label numbers and
            values are the corresponding names.
        """
        ...
    
    @abstractmethod
    def get_split(self, split): # -> None:
        """Returns a dataset split.

        Args:
            split: A string identifying the dataset split that is usually one of
            'training', 'test', 'validation', or 'all'.

        Returns:
            A dataset split object providing the requested subset of the data.
        """
        ...
    
    @abstractmethod
    def is_tested(self, attr): # -> Literal[False]:
        """Checks whether a datum has been tested.

        Args:
            attr: The attributes associated with the datum.

        Returns:
            This returns True if the test result has been stored for the datum with the
            specified attribute; else returns False.
        """
        ...
    
    @abstractmethod
    def save_test_result(self, results, attr): # -> None:
        """Saves the output of a model.

        Args:
            results: The output of a model for the datum associated with the attribute passed.
            attr: The attributes that correspond to the outputs passed in results.
        """
        ...
    


class BaseDatasetSplit(ABC):
    """The base class for dataset splits.

    This class provides access to the data of a specified subset or split of a dataset.

    Args:
        dataset: The dataset object associated to this split.
        split: A string identifying the dataset split, usually one of
            'training', 'test', 'validation', or 'all'.

    Attributes:
        cfg: Shortcut to the config of the dataset object.
        dataset: The dataset object associated to this split.
        split: A string identifying the dataset split, usually one of
            'training', 'test', 'validation', or 'all'.
    """
    def __init__(self, dataset, split=...) -> None:
        ...
    
    @abstractmethod
    def __len__(self): # -> Literal[0]:
        """Returns the number of samples in the split."""
        ...
    
    @abstractmethod
    def get_data(self, idx): # -> dict[Unknown, Unknown]:
        """Returns the data for the given index."""
        ...
    
    @abstractmethod
    def get_attr(self, idx): # -> dict[Unknown, Unknown]:
        """Returns the attributes for the given index."""
        ...
    

