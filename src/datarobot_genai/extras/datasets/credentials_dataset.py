from typing import Any
from kedro.io import AbstractDataset


class CredentialsDataset(AbstractDataset):
    def __init__(self, credentials: dict[str, Any]) -> None:
        self.credentials = credentials

    def _load(self) -> dict[str, Any]:
        return self.credentials

    def _save(self, credentials: dict[str, Any]) -> None:
        raise NotImplementedError("Saving not implemented for CredentialsDataset")

    def _describe(self) -> dict[str, Any]:
        return dict()

    def _exists(self) -> bool:
        return True
