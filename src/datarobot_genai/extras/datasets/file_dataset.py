from kedro.io import AbstractDataset
from typing import Any
import os


class AnyFileDataset(AbstractDataset):
    def __init__(self, filepath: str):
        self._filepath = filepath

    def _load(self) -> bytes:
        with open(self._filepath, "rb") as f:
            return f.read()

    def _save(self, data: bytes) -> None:
        with open(self._filepath, "wb") as f:
            f.write(data)

    def _describe(self) -> dict[str, Any]:
        return dict(filepath=self._filepath)

    def _exists(self) -> bool:
        return os.path.isfile(self._filepath)
