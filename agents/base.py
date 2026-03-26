"""
Base agent class for all CognitionHive agents.
Provides shared config, thresholds, and logging setup.
"""

import logging


class BaseAgent:
    """Base class that all CognitionHive agents inherit from."""

    def __init__(self, config: dict = None, thresholds: dict = None, **kwargs):
        self.config = config or {}
        self.thresholds = thresholds or {}
        self.name = self.__class__.__name__
        self.logger = logging.getLogger(f"cognition-hive.{self.name.lower()}")

    def __repr__(self):
        return f"<{self.name}>"
