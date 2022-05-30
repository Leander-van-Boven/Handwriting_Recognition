from typing import Union
from pathlib import Path


class DssCharDataset:
    def __init__(self, store_dir: Union[Path, str]):
        self.store_dir = store_dir
