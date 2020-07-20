import inspect
import os
import os.path
import re
import numpy as np
from inspect import Parameter
from typing import Any, Callable, Dict, Iterable, List, Tuple, Type

import yaml
from allennlp.common import util
from elasticsearch import Elasticsearch

from . import environment

_INVALID_TAG_CHARACTERS = re.compile(r"[^-/\w\.]")


def yaml_to_dict(filepath: str) -> Dict[str, Any]:
    """Loads a yaml file into a data dictionary

    Parameters
    ----------
    filepath
        Path to the yaml file

    Returns
    -------
    dict
    """
    with open(filepath) as yaml_content:
        config = yaml.safe_load(yaml_content)
    return config


def get_compatible_doc_type(client: Elasticsearch) -> str:
    """Find a compatible name for doc type by checking the cluster info

    Parameters
    ----------
    client
        The elasticsearch client

    Returns
    -------
    name
        A compatible name for doc type in function of cluster version
    """
    es_version = int(client.info()["version"]["number"].split(".")[0])
    return "_doc" if es_version >= 6 else "doc"


def get_env_cuda_device() -> int:
    """Gets the cuda device from an environment variable.

    This is necessary to activate a GPU if available

    Returns
    -------
    cuda_device
        The integer number of the CUDA device
    """
    cuda_device = int(os.getenv(environment.CUDA_DEVICE, "-1"))

    return cuda_device


def update_method_signature(
    signature: inspect.Signature, to_method: Callable
) -> Callable:
    """Updates the signature of a method

    Parameters
    ----------
    signature
        The signature with which to update the method
    to_method
        The method whose signature will be updated

    Returns
    -------
    updated_method
    """

    def wrapper(*args, **kwargs):
        return to_method(*args, **kwargs)

    wrapper.__signature__ = signature
    return wrapper


def isgeneric(class_type: Type) -> bool:
    """Checks if a class type is a generic type (List[str] or Union[str, int]"""
    return hasattr(class_type, "__origin__")


def is_running_on_notebook() -> bool:
    """Checks if code is running inside a jupyter notebook"""
    try:
        import IPython

        return IPython.get_ipython().has_trait("kernel")
    except (AttributeError, NameError, ModuleNotFoundError):
        return False


def split_signature_params_by_predicate(
    signature_function: Callable, predicate: Callable
) -> Tuple[List[Parameter], List[Parameter]]:
    """Splits parameters signature by defined boolean predicate function"""
    signature = inspect.signature(signature_function)
    parameters = list(
        filter(
            lambda p: p.name != "self"
            and p.kind not in [Parameter.VAR_KEYWORD, Parameter.VAR_POSITIONAL],
            signature.parameters.values(),
        )
    )
    matches_group = list(filter(lambda p: predicate(p), parameters))
    non_matches_group = list(filter(lambda p: not predicate(p), parameters))

    return matches_group, non_matches_group


def sanitize_metric_name(name: str) -> str:
    """Sanitizes the name to comply with tensorboardX conventions when logging.

    Parameter
    ---------
    name
        Name of the metric

    Returns
    -------
    sanitized_name
    """
    if not name:
        return name
    new_name = _INVALID_TAG_CHARACTERS.sub("_", name)
    new_name = new_name.lstrip("/")
    return new_name


def save_dict_as_yaml(dictionary: dict, path: str) -> str:
    """Save a cfg dict to path as yaml

    Parameters
    ----------
    dictionary
        Dictionary to be saved
    path
        Filesystem location where the yaml file will be saved

    Returns
    -------
    path
        Location of the yaml file
    """
    dir_name = os.path.dirname(path)
    # Prevent current workdir relative routes
    # `save_dict_as_yaml("just_here.yml")
    if dir_name:
        os.makedirs(dir_name, exist_ok=True)

    with open(path, "w") as yml_file:
        yaml.dump(dictionary, yml_file, default_flow_style=False, allow_unicode=True)

    return path


def get_full_class_name(the_class: Type) -> str:
    """Given a type class return the full qualified class name """
    # o.__module__ + "." + o.__class__.__qualname__ is an example in
    # this context of H.L. Mencken's "neat, plausible, and wrong."
    # Python makes no guarantees as to whether the __module__ special
    # attribute is defined, so we take a more circumspect approach.
    # Alas, the module name is explicitly excluded from __qualname__
    # in Python 3.

    module = the_class.__module__
    if module is None or module == str.__class__.__module__:
        return the_class.__name__  # Avoid reporting __builtin__
    else:
        return module + "." + the_class.__name__


def stringify(value: Any) -> Any:
    """Creates an equivalent data structure representing data values as string

    Parameters
    ----------
    value
        Value to be stringified

    Returns
    -------
    stringified_value
    """
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        return {key: stringify(value) for key, value in value.items()}
    if isinstance(value, Iterable):
        return [stringify(v) for v in value]
    return str(value)


def sanitize_for_params(x: Any) -> Any:
    """Sanitizes the input for a more flexible usage with AllenNLP's `.from_params()` machinery.

    For now it is mainly used to transform numpy numbers to python types

    Parameters
    ----------
    x
        The parameter passed on to `allennlp.common.FromParams.from_params()`

    Returns
    -------
    sanitized_x
    """
    # AllenNLP has a similar method (allennlp.common.util.sanitize) but it does not work for my purpose, since
    # numpy types are checked only after the float type check, and:
    # isinstance(numpy.float64(1), float) == True !!!
    if isinstance(x, util.numpy.number):
        return x.item()
    elif isinstance(x, util.numpy.bool_):
        # Numpy bool_ need to be converted to python bool.
        return bool(x)
    if isinstance(x, (str, float, int, bool)):
        return x
    elif isinstance(x, dict):
        # Dicts need their values sanitized
        return {key: sanitize_for_params(value) for key, value in x.items()}
    # Lists and Tuples need their values sanitized
    elif isinstance(x, list):
        return [sanitize_for_params(x_i) for x_i in x]
    elif isinstance(x, tuple):
        return tuple(sanitize_for_params(x_i) for x_i in x)

    return x


def moving_average(values: 'np.array_like', window: int = 3) -> np.array:
    """Returns a simple moving average of the values.

    It tries to take an equal number of data on either side of the central value by padding the `values`
    on the left with `values[0]` and on the right with `values[-1]`.
    If `window` is an even number, it takes one more value to the right than to the left of the central value.

    Parameters
    ----------
    values
        An `np.array_like` input of values to be averaged.
    window
        Size of the window.

    Returns
    -------
    averages
    """
    # TODO: I would much rather use the pandas implementation, but it seems to be broken at the moment:
    #       https://github.com/pandas-dev/pandas/issues/11704
    pad_left = int((window - 1) / 2)
    pad_right = int(window / 2)

    values = np.concatenate([[values[0]] * pad_left, values, [values[-1]] * pad_right])
    averages = np.convolve(np.array(values), np.ones(window) / window, mode="valid")

    return averages
