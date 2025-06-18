#! /usr/bin/env python3

from abc import ABC, abstractmethod
class Constraint(ABC):
    """
    Abstract base class for constraints.
    """
    def __init__(self, *args, **kwargs):
        """
        Initialize the constraint with optional arguments.
        :param args: Positional arguments.
        :param kwargs: Keyword arguments.
        """
        self.args = args
        self.kwargs = kwargs

    @abstractmethod
    def is_satisfied(self, *args, **kwargs) -> bool:
        """
        Check if the constraint is satisfied.
        :param args: Positional arguments.
        :param kwargs: Keyword arguments.
        :return: True if the constraint is satisfied, False otherwise.
        """
        pass

    @abstractmethod
    def __str__(self) -> str:
        """
        String representation of the constraint.
        :return: String representation of the constraint.
        """
        pass
