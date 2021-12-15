from typing import Any, Dict, Optional

import yaml


class PrettySafeLoader(yaml.SafeLoader):
    def construct_python_tuple(self, node):
        return tuple(self.construct_sequence(node))


PrettySafeLoader.add_constructor(
    "tag:yaml.org,2002:python/tuple", PrettySafeLoader.construct_python_tuple
)


def yaml_parser(
    path: Optional[str] = None,
    data: Optional[str] = None,
    loader: Any = PrettySafeLoader,
) -> Dict[str, Any]:
    if path:
        with open(r"{}".format(path)) as file:
            return yaml.load(file, Loader=loader)

    elif data:
        return yaml.load(data, Loader=loader)

    else:
        raise ValueError("Either a path or data should be defined as input")
