# %%
import copy
import importlib.util
import pathlib
from timeit import default_timer as timer

from typing import TypedDict, Literal, Union, Any, cast


class Config(TypedDict):
    display_as: Literal["widget", "html"]
    dev: bool
    defaults: dict[Any, Any]


CONFIG: Config = {"display_as": "widget", "dev": False, "defaults": {}}


def configure(options: dict[str, Any] = {}, **kwargs: Any) -> None:
    CONFIG.update(cast(Config, {**options, **kwargs}))


def get_config(k: str) -> Union[str, None]:
    return CONFIG.get(k)


class benchmark(object):
    """
    A context manager for simple benchmarking.

    Usage:
        with benchmark("My benchmark"):
            # Code to be benchmarked
            ...

    Args:
        msg (str): The message to display with the benchmark result.
        fmt (str, optional): The format string for the time display. Defaults to "%0.3g".

    http://dabeaz.blogspot.com/2010/02/context-manager-for-timing-benchmarks.html
    """

    def __init__(self, msg, fmt="%0.3g"):
        self.msg = msg
        self.fmt = fmt

    def __enter__(self):
        self.start = timer()
        return self

    def __exit__(self, *args):
        t = timer() - self.start
        print(("%s : " + self.fmt + " seconds") % (self.msg, t))
        self.time = t


try:
    PARENT_PATH = pathlib.Path(importlib.util.find_spec("genstudio.util").origin).parent  # type: ignore
except AttributeError:
    raise ImportError("Cannot find the genstudio.util module")
# %%


def deep_merge(dict1: dict[str, Any], dict2: dict[str, Any]) -> dict[str, Any]:
    """
    Recursively merge two dictionaries. Mutates dict1.
    Values in dict2 overwrite values in dict1. If both values are dictionaries, recursively merge them.
    """

    for k, v in dict2.items():
        if k in dict1 and isinstance(dict1[k], dict) and isinstance(v, dict):
            dict1[k] = deep_merge(dict1[k], v)
        elif isinstance(v, dict):
            dict1[k] = copy.deepcopy(v)
        else:
            dict1[k] = v
    return dict1
