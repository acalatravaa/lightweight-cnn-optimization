"""
Configuration module initialization.
Dynamically maps uppercase constants from global_settings into a unified configuration object.
"""

from typing import Any
import stage1_classification.conf.global_settings as default_settings


class Settings:
    """
    Configuration wrapper class.
    Extracts all public, uppercase constants from the settings module.
    """
    def __init__(self, config_module: Any) -> None:
        for attr in dir(config_module):
            if attr.isupper():
                setattr(self, attr, getattr(config_module, attr))


# Initialize the singleton settings object to be imported across the application
settings = Settings(default_settings)