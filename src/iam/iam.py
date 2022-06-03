from typing import Union
from pathlib import Path

from attrdict import AttrDict


class IamPipeline:
    STAGES = [

    ]

    def __init__(self, conf: AttrDict, source_dir: Union[Path, str], store_dir: Union[Path, str]):
        self.conf = conf
        self.store_dir = Path(store_dir).resolve()
        self.source_dir = Path(source_dir).resolve()

    def pipeline(self):
        pass

    def _get_lines(self):


