"""Where the code runs (Colab, Kaggle or your computer) and the Hugging Face token.

The token is looked for in this order (the first one found is used):

    1. the ``token`` argument
    2. the environment variable HF_TOKEN (or HUGGING_FACE_HUB_TOKEN)
    3. Colab: the secret HF_TOKEN (key icon in the left bar, "Notebook access" on)
    4. Kaggle: the secret HF_TOKEN (Add-ons > Secrets, attached to the notebook)
    5. none: huggingface_hub then uses a saved ``hf auth login`` if there is one
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

SECRET = "HF_TOKEN"


def detect() -> str:
    """"colab", "kaggle" or "local"."""
    if "google.colab" in sys.modules or os.environ.get("COLAB_RELEASE_TAG"):
        return "colab"
    if os.environ.get("KAGGLE_KERNEL_RUN_TYPE") or Path("/kaggle/working").is_dir():
        return "kaggle"
    return "local"


def _colab_secret(name: str) -> str | None:
    try:
        from google.colab import userdata
        return userdata.get(name) or None
    except Exception:                     # not on Colab, secret missing, no notebook access
        return None


def _kaggle_secret(name: str) -> str | None:
    try:
        from kaggle_secrets import UserSecretsClient
        return UserSecretsClient().get_secret(name) or None
    except Exception:                     # not on Kaggle, or secret not attached
        return None


def hf_token(token: str | None = None) -> tuple[str | None, str]:
    """(token, where it came from): "argument", "env:<VARIABLE>", "colab secret",
    "kaggle secret" or "none"."""
    if token:
        return token, "argument"
    for var in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
        if os.environ.get(var):
            return os.environ[var], f"env:{var}"
    for source, read in (("colab secret", _colab_secret), ("kaggle secret", _kaggle_secret)):
        value = read(SECRET)
        if value:
            return value, source
    return None, "none"


def token_help(where: str | None = None) -> str:
    """How to give NavSim the token on this platform."""
    where = where or detect()
    if where == "colab":
        return ("Colab: click the key icon in the left bar, add a secret named HF_TOKEN with a "
                "Hugging Face *Read* token, and switch on 'Notebook access'.")
    if where == "kaggle":
        return ("Kaggle: Add-ons > Secrets > Add a new secret named HF_TOKEN with a Hugging Face "
                "*Read* token, tick it for this notebook, and turn Internet on (Settings).")
    return ("set the HF_TOKEN environment variable to a Hugging Face *Read* token, or run "
            "`hf auth login` once.")
