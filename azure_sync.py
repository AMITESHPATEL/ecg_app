"""
Azure Blob Storage synchronization helper for Streamlit ECG Annotation App
===========================================================================

Provides automatic sync of annotations.csv to an Azure Blob Storage container.

Supports SAS URL configuration via:
1. Streamlit secrets (`st.secrets["AZURE_SAS_URL"]` or `st.secrets["azure_sas_url"]`)
2. Environment variable `AZURE_SAS_URL`
3. Fallback default SAS URL
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

from azure.core.exceptions import ResourceNotFoundError
from azure.storage.blob import BlobClient, ContainerClient

logger = logging.getLogger("azure_sync")

DEFAULT_AZURE_SAS_URL = (
    "https://rpmsmlsinblob.blob.core.windows.net/vt-svt-annotation?"
    "sp=racwl&st=2026-08-18T10:16:08Z&se=2026-08-18T18:31:08Z&spr=https&sv=2026-02-06&sr=c&"
    "sig=ukJ4JR%2Fk5ngyH3FVkzR8fq9lUMqTg7xVFZZJob09LM8%3D"
)


def get_azure_sas_url() -> str:
    """Retrieve Azure Container SAS URL from secrets, environment, or default."""
    try:
        import streamlit as st

        if hasattr(st, "secrets"):
            if "AZURE_SAS_URL" in st.secrets:
                return str(st.secrets["AZURE_SAS_URL"]).strip()
            if "azure_sas_url" in st.secrets:
                return str(st.secrets["azure_sas_url"]).strip()
            if "azure" in st.secrets and isinstance(st.secrets["azure"], dict):
                return str(st.secrets["azure"].get("sas_url", "")).strip()
    except Exception as exc:
        logger.debug("Could not read st.secrets for Azure SAS URL: %s", exc)

    return os.environ.get("AZURE_SAS_URL", DEFAULT_AZURE_SAS_URL).strip()


def get_container_client(sas_url: str | None = None) -> ContainerClient | None:
    """Create and return an Azure ContainerClient, or None if URL is invalid."""
    url = sas_url or get_azure_sas_url()
    if not url:
        return None
    try:
        return ContainerClient.from_container_url(url)
    except Exception as exc:
        logger.error("Failed to initialize Azure ContainerClient: %s", exc)
        return None


def download_blob_from_azure(
    client: ContainerClient, blob_name: str, local_dest: Path
) -> bool:
    """Download a blob from Azure Blob Storage to local_dest. Returns True on success."""
    try:
        blob_client = client.get_blob_client(blob_name)
        if not blob_client.exists():
            return False

        local_dest.parent.mkdir(parents=True, exist_ok=True)
        with open(local_dest, "wb") as fh:
            data = blob_client.download_blob().readall()
            fh.write(data)
        logger.info("Successfully downloaded '%s' from Azure Blob Storage.", blob_name)
        return True
    except ResourceNotFoundError:
        return False
    except Exception as exc:
        logger.warning("Error downloading '%s' from Azure Blob Storage: %s", blob_name, exc)
        return False


def upload_blob_to_azure(
    client: ContainerClient, local_path: Path, blob_name: str
) -> bool:
    """Upload or overwrite a local file to Azure Blob Storage. Returns True on success."""
    if not local_path.exists():
        logger.warning("Local file '%s' does not exist for Azure upload.", local_path)
        return False

    try:
        blob_client = client.get_blob_client(blob_name)
        with open(local_path, "rb") as data:
            blob_client.upload_blob(data, overwrite=True)
        logger.info("Successfully uploaded '%s' to Azure Blob Storage.", blob_name)
        return True
    except Exception as exc:
        logger.warning("Error uploading '%s' to Azure Blob Storage: %s", blob_name, exc)
        return False
