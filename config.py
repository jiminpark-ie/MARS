from __future__ import annotations

import argparse
import os
from typing import Any, Optional


def add_auth_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument(
        "--openai-api-key",
        default=None
    )
    parser.add_argument(
        "--hf-token",
        default=None
    )
    parser.add_argument(
        "--hf-user",
        default=None
    )
    parser.add_argument(
        "--hf-project",
        default=None
    )
    parser.add_argument(
        "--wandb-api-key",
        default=None
    )
    return parser


def resolve_openai_api_key(value: Optional[str] = None, required: bool = False) -> Optional[str]:
    key = value or os.environ.get("OPENAI_API_KEY")
    if required and not key:
        raise RuntimeError("Provide --openai-api-key or set OPENAI_API_KEY.")
    return key


def resolve_hf_token(value: Optional[str] = None, required: bool = False) -> Optional[str]:
    token = value or os.environ.get("HF_TOKEN")
    if required and not token:
        raise RuntimeError("Provide --hf-token or set HF_TOKEN.")
    return token


def resolve_hf_project(value: Optional[str] = None, required: bool = False) -> Optional[str]:
    project = value or os.environ.get("HF_PROJECT")
    if required and not project:
        raise RuntimeError("Provide --hf-project or set HF_PROJECT.")
    return project


def resolve_hf_username(value: Optional[str] = None, required: bool = False) -> Optional[str]:
    name = value or os.environ.get("HF_USER")
    if required and not name:
        raise RuntimeError("Provide --hf-user or set HF_USER.")
    return name


def resolve_wandb_api_key(value: Optional[str] = None) -> Optional[str]:
    return value or os.environ.get("WANDB_API_KEY")


def apply_auth_args_to_env(args: Any) -> None:
    openai_key = resolve_openai_api_key(getattr(args, "openai_api_key", None))
    hf_token = resolve_hf_token(getattr(args, "hf_token", None))
    hf_project = resolve_hf_project(getattr(args, "hf_project", None))
    hf_username = resolve_hf_username(getattr(args, "hf_user", None))
    wandb_key = resolve_wandb_api_key(getattr(args, "wandb_api_key", None))

    if openai_key:
        os.environ["OPENAI_API_KEY"] = openai_key
    if hf_token:
        os.environ["HF_TOKEN"] = hf_token
    if hf_project:
        os.environ["HF_PROJECT"] = hf_project
    if hf_username:
        os.environ["HF_USER"] = hf_username
    if wandb_key:
        os.environ["WANDB_API_KEY"] = wandb_key


def configure_openai(openai_module: Any, api_key: Optional[str] = None, required: bool = False) -> Optional[str]:
    key = resolve_openai_api_key(api_key, required=required)
    if key:
        openai_module.api_key = key
    return key

