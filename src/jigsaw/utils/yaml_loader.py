import yaml
import os
from ensure import ensure_annotations

class YAMLoader(yaml.SafeLoader):

    def __init__(self, stream):

        self._root = os.path.split(stream.name)[0]

        super().__init__(stream)

    def include(self, node: yaml.SequenceNode):
        sequence = self.construct_sequence(node)
        filename = sequence[0]

        with open(filename, 'r') as f:
            return yaml.load(f, YAMLoader)[sequence[1]]

yaml.add_constructor('!include', YAMLoader.include, Loader = YAMLoader)
