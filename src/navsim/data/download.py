"""``navsim.setup``: get datasets ready in one call, on Colab, Kaggle or your computer.

    navsim.setup()                                   # the demo inside the package: nothing to download
    navsim.setup("isb-svg-v1")                       # download once, reuse afterwards
    navsim.setup("isb-svg-v1", data_dir="/content/drive/MyDrive/navsim")   # keep it on Drive

A dataset already on this machine is used as it is. Otherwise it is downloaded from the
Hugging Face repository below, which is gated: request access on its page, then give
NavSim a *Read* token (see :mod:`navsim.data.platform`).
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

from .dataset import Dataset, download_dir, find_dataset, validate_dataset
from .platform import detect, hf_token, token_help

REPO_ID = "PhysicalSI/2D_Navigation_Sim"
REPO_URL = f"https://huggingface.co/datasets/{REPO_ID}"


def setup(datasets="demo-svg", token: str | None = None, data_dir: str | os.PathLike | None = None,
          force: bool = False, repo_id: str = REPO_ID, quiet: bool = False):
    """Make datasets ready and return them (a ``Dataset``, or a list when given a list).

    ``data_dir``: where to download (default ~/.navsim/datasets). ``token``: Hugging Face
    token (default: $HF_TOKEN, or the Colab / Kaggle secret HF_TOKEN). ``force``: download
    again even if the dataset is already here."""
    names = [datasets] if isinstance(datasets, (str, os.PathLike)) else list(datasets)
    dest = Path(data_dir).expanduser().resolve() if data_dir else None
    if dest is not None:
        dest.mkdir(parents=True, exist_ok=True)
        _add_search_dir(dest)
    out = []
    for name in names:
        found = None if force else find_dataset(name, dest)
        if found is None:
            if Path(name).expanduser().is_absolute() or os.sep in str(name) or "/" in str(name):
                raise FileNotFoundError(f"no dataset at {name} (no manifest.json)")
            tok, source = hf_token(token)
            if not quiet:
                print(f"navsim: downloading {name} from Hugging Face ({detect()}, token: {source}) ...",
                      flush=True)
            found = download(name, dest, token=tok, force=force, repo_id=repo_id)[0]
        ds = Dataset(found)
        if not quiet:
            print(f"navsim: {ds.name} ready ({ds.env_type}, robots {ds.robots}) at {ds.root}")
        out.append(ds)
    return out[0] if isinstance(datasets, (str, os.PathLike)) else out


def _add_search_dir(folder: Path) -> None:
    """Put ``folder`` first in $NAVSIM_DATASETS, so that worker processes (vector
    environments) find the datasets too: they inherit the environment variables."""
    current = [p for p in os.environ.get("NAVSIM_DATASETS", "").split(os.pathsep) if p]
    if str(folder) not in current:
        os.environ["NAVSIM_DATASETS"] = os.pathsep.join([str(folder)] + current)


# ------------------------------------------------------------------ Hugging Face

def _hub():
    try:
        import huggingface_hub
        import huggingface_hub.errors  # noqa: F401
    except ImportError:
        raise ImportError('downloading datasets needs huggingface_hub: pip install "navsim[hub]"') from None
    return huggingface_hub


def _access_error(e: Exception) -> PermissionError:
    return PermissionError(
        f"no access to {REPO_URL} ({type(e).__name__}). Request access on that page and wait for "
        f"the approval e-mail, then give NavSim a token: {token_help()}")


def _network_error(e: Exception) -> ConnectionError:
    return ConnectionError(
        f"cannot reach Hugging Face ({type(e).__name__}: {e}). Check the internet connection"
        " (Kaggle: Settings > Internet on), or add the dataset folder as a Kaggle / Drive input "
        "and pass its path.")


def available(token: str | None = None, repo_id: str = REPO_ID) -> list[str]:
    """Names of the datasets in the repository (its top-level folders)."""
    hub = _hub()
    try:
        tree = hub.HfApi(token=token).list_repo_tree(repo_id, repo_type="dataset")
        return sorted(e.path for e in tree if isinstance(e, hub.hf_api.RepoFolder)
                      and not e.path.startswith("."))
    except (hub.errors.GatedRepoError, hub.errors.RepositoryNotFoundError) as e:
        raise _access_error(e) from None
    except (OSError, hub.errors.HfHubHTTPError) as e:
        raise _network_error(e) from None


def download(names, dest: str | os.PathLike | None = None, token: str | None = None,
             force: bool = False, repo_id: str = REPO_ID) -> list[Path]:
    """Download datasets by name into ``dest`` (default ~/.navsim/datasets), check them
    against their manifest and return their folders. Datasets already there are kept
    unless ``force``."""
    hub = _hub()
    names = [names] if isinstance(names, str) else list(names)
    dest = Path(dest).expanduser() if dest else download_dir()
    dest.mkdir(parents=True, exist_ok=True)
    todo = []
    for name in names:
        if (dest / name / "manifest.json").exists() and not force:
            continue
        if force and (dest / name).exists():
            shutil.rmtree(dest / name)
        todo.append(name)
    if todo:
        try:
            hub.snapshot_download(repo_id, repo_type="dataset", local_dir=dest, token=token,
                                  allow_patterns=[f"{n}/*" for n in todo])
        except (hub.errors.GatedRepoError, hub.errors.RepositoryNotFoundError) as e:
            for name in todo:                     # nothing was downloaded: no empty folders
                if (dest / name).is_dir() and not any(p.is_file() for p in (dest / name).rglob("*")):
                    shutil.rmtree(dest / name)
            raise _access_error(e) from None
        except (OSError, hub.errors.HfHubHTTPError) as e:
            raise _network_error(e) from None
    out = []
    for name in names:
        root = dest / name
        if not (root / "manifest.json").exists():
            raise FileNotFoundError(f"dataset '{name}' is not in {REPO_URL}; available: "
                                    f"{available(token, repo_id)}")
        problems = validate_dataset(root)
        if problems:
            raise ValueError(f"{name}: downloaded files do not match the manifest: {problems[:5]}")
        out.append(root.resolve())
    return out
