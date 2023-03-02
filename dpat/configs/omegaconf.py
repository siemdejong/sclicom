"""Configure settings for Omegaconf."""
from datetime import datetime

from omegaconf import OmegaConf


def register_conf_resolvers():
    """Register configuration resolvers.

    Examples
    --------
    The snippet
    ```yaml
    key: ${now:%Y-%m-%d}
    ```
    sets `key` to the current year-month-day (e.g. 2023-03-02)
    """
    OmegaConf.register_new_resolver("now", lambda x: datetime.now().strftime(x))
