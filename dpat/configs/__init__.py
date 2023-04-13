"""Apply configuration settings."""
from .config import get_default_config_by_name
from .omegaconf import register_conf_resolvers

__all__ = ["get_default_config_by_name", "register_conf_resolvers"]
