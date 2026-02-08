from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import pandas as pd

from ddl69.core.settings import Settings

@dataclass
class ParquetArtifact:
    uri: str
    sha256: str
    rows: int

class ParquetStore:
    """Local Parquet artifact store. Swap out ARTIFACT_ROOT later to S3/R2."""

    def __init__(self, settings: Settings):
        self.root = Path(settings.artifact_root)
        self.root.mkdir(parents=True, exist_ok=True)

    def write_df(self, df: pd.DataFrame, kind: str, name: str) -> ParquetArtifact:
        path = self.root / kind
        path.mkdir(parents=True, exist_ok=True)
        file_path = path / f"{name}.parquet"
        df.to_parquet(file_path, index=False)
        sha = _sha256_file(file_path)
        return ParquetArtifact(uri=str(file_path), sha256=sha, rows=len(df))


def _sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()
