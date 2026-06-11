import hashlib
import json
import os
from datetime import datetime, timezone
from typing import Any

import pandas as pd


def ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def ts_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def hash_df(df: pd.DataFrame) -> str:
    b = pd.util.hash_pandas_object(df.fillna(0), index=True).values.tobytes()
    return hashlib.sha256(b).hexdigest()


def write_json(path: str, obj: dict[str, Any]) -> None:
    tmp = f"{path}.tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


def snapshot_run_dir(base_out: str, run_id: str) -> str:
    return ensure_dir(os.path.join(base_out, "phase6", run_id))


def save_artifacts(
    run_dir: str,
    artifacts: dict[str, pd.DataFrame],
    parquet: bool = False,
) -> dict[str, str]:
    """Persist DataFrames to CSV or Parquet and return a path map."""
    out: dict[str, str] = {}
    for name, df in artifacts.items():
        ext = "parquet" if parquet else "csv"
        fp = os.path.join(run_dir, f"{name}.{ext}")
        if parquet:
            df.to_parquet(fp, index=False)
        else:
            df.to_csv(fp, index=False)
        out[name] = fp
    return out


def manifest(run_dir: str, data: dict[str, Any]) -> None:
    write_json(os.path.join(run_dir, "run_manifest.json"), data)
