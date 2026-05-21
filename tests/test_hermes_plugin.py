"""Validate the Hermes provider profile against the real ProviderProfile shape.

We don't have the Hermes ``providers`` package installed, so we stub it with a
dataclass that mirrors ``providers/base.py`` from nousresearch/hermes-agent.
Importing the plugin against this stub catches field-name typos and confirms
``register_provider`` is called with the values Hermes needs.
"""

import importlib.util
import sys
import types
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

PLUGIN = Path(__file__).resolve().parent.parent / "hermes_plugin" / "gigachat" / "__init__.py"


@dataclass
class ProviderProfile:  # mirrors hermes-agent providers/base.py
    name: str
    api_mode: str = "chat_completions"
    aliases: tuple = ()
    display_name: str = ""
    description: str = ""
    signup_url: str = ""
    env_vars: tuple = ()
    base_url: str = ""
    models_url: str = ""
    auth_type: str = "api_key"
    supports_health_check: bool = True
    fallback_models: tuple = ()
    hostname: str = ""
    default_headers: dict = field(default_factory=dict)
    fixed_temperature: Any = None
    default_max_tokens: int = None
    default_aux_model: str = ""


@pytest.fixture
def loaded_profile(monkeypatch):
    registered = []

    providers_pkg = types.ModuleType("providers")
    providers_pkg.register_provider = lambda p: registered.append(p)
    base_mod = types.ModuleType("providers.base")
    base_mod.ProviderProfile = ProviderProfile
    base_mod.OMIT_TEMPERATURE = object()

    monkeypatch.setitem(sys.modules, "providers", providers_pkg)
    monkeypatch.setitem(sys.modules, "providers.base", base_mod)

    spec = importlib.util.spec_from_file_location("hermes_gigachat_plugin", PLUGIN)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    assert len(registered) == 1
    return registered[0]


def test_profile_registers_with_expected_identity(loaded_profile):
    assert loaded_profile.name == "gigachat"
    assert loaded_profile.api_mode == "chat_completions"
    assert "giga" in loaded_profile.aliases


def test_profile_points_at_proxy_and_declares_env(loaded_profile):
    assert loaded_profile.base_url.endswith("/v1")
    assert loaded_profile.auth_type == "api_key"
    # first env var holds the key sent to the proxy, second overrides base_url
    assert len(loaded_profile.env_vars) == 2
    assert loaded_profile.env_vars[0].endswith("_API_KEY")


def test_profile_lists_gigachat_models(loaded_profile):
    assert "GigaChat" in loaded_profile.fallback_models
    assert loaded_profile.default_aux_model in loaded_profile.fallback_models
