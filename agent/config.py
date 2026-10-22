# build-id: oa-4042bb34
"""Environment -> Config dataclass (DESIGN §4). Secrets are masked in repr."""

from __future__ import annotations

import dataclasses
import os
from dataclasses import dataclass
from typing import Any, Mapping, Optional

from .errors import ConfigError, InvalidRequest

DEFAULT_BASE_URL = "http://127.0.0.1:8088/v1"
_TRUE = {"true", "1", "yes", "on"}
_FALSE = {"false", "0", "no", "off"}
_LEVELS = ("DEBUG", "INFO", "WARNING", "ERROR")
_PLACEHOLDERS = {"", "rt-local", "your-key-here", "changeme"}


def _raw(env: Mapping[str, str], *names: str) -> Optional[str]:
    for n in names:
        v = env.get(n)
        if v is not None and str(v).strip() != "":
            return str(v).strip()
    return None


def _bool(env: Mapping[str, str], name: str, default: bool) -> bool:
    v = _raw(env, name)
    if v is None:
        return default
    low = v.lower()
    if low in _TRUE:
        return True
    if low in _FALSE:
        return False
    raise ConfigError(f"Invalid boolean for {name}", details={"var": name})


def _int(env: Mapping[str, str], name: str, default: int, minimum: int, maximum: Optional[int] = None) -> int:
    v = _raw(env, name)
    if v is None:
        return default
    try:
        n = int(v)
    except ValueError:
        raise ConfigError(f"Invalid integer for {name}", details={"var": name}) from None
    if n < minimum or (maximum is not None and n > maximum):
        raise ConfigError(f"Value for {name} out of range", details={"var": name})
    return n


def _float(env: Mapping[str, str], name: str, default: float, minimum: float) -> float:
    v = _raw(env, name)
    if v is None:
        return default
    try:
        n = float(v)
    except ValueError:
        raise ConfigError(f"Invalid number for {name}", details={"var": name}) from None
    if not (n >= minimum) or n != n or n == float("inf"):
        raise ConfigError(f"Value for {name} out of range", details={"var": name})
    return n


def _mask(v: Optional[str]) -> Optional[str]:
    return "***" if v else None


@dataclass(frozen=True, repr=False)
class Config:
    sample_mode: bool = False
    log_level: str = "INFO"
    port: int = 8000
    log_dir: str = "logs"
    data_dir: str = "data/sample"
    engine: str = "stub"
    live_provider: str = "meterless"  # LIVE_PROVIDER: model provider used when SAMPLE_MODE=false
    hmem_path: str = ".hmem/store.json"
    # raw value (normalised later by provider.normalize_base_url); None -> autodetect
    meterless_base_url: Optional[str] = DEFAULT_BASE_URL
    meterless_runtime_token: Optional[str] = None
    meterless_autodetect: bool = False
    model_id: str = "meterless/auto"
    provider_timeout_s: float = 180.0
    source_timeout_s: float = 20.0
    zetaris_base_url: Optional[str] = None  # direct REST base, from ZETARIS_REST_URL
    zetaris_org_id: str = "1"
    zetaris_api_key: Optional[str] = None
    zetaris_table_orders: str = "orders"
    zetaris_table_tickets: str = "tickets"
    zetaris_table_shipments: str = "shipments"
    nvidia_api_key: Optional[str] = None
    run_timeout_s: float = 300.0
    max_refinements: int = 2
    max_steps: int = 60
    max_model_calls: int = 12
    memory_enabled: bool = True

    # ---- construction -------------------------------------------------
    @classmethod
    def from_env(cls, env: Optional[Mapping[str, str]] = None) -> "Config":
        env = os.environ if env is None else env

        log_level = (_raw(env, "LOG_LEVEL") or "INFO").upper()
        if log_level not in _LEVELS:
            raise ConfigError("Invalid LOG_LEVEL", details={"var": "LOG_LEVEL"})

        engine = (_raw(env, "ENGINE") or "stub").lower()
        if engine not in ("stub", "meterless"):
            raise ConfigError("Invalid ENGINE (expected 'stub' or 'meterless')", details={"var": "ENGINE"})

        live_provider = (_raw(env, "LIVE_PROVIDER") or "meterless").lower()
        if live_provider not in ("meterless", "sample"):
            raise ConfigError("Invalid LIVE_PROVIDER (expected 'meterless' or 'sample')",
                              details={"var": "LIVE_PROVIDER"})

        autodetect = _bool(env, "METERLESS_AUTODETECT", False)
        base = _raw(env, "METERLESS_BASE_URL", "MODEL_BASE_URL")
        if base is None and not autodetect:
            base = DEFAULT_BASE_URL

        return cls(
            sample_mode=_bool(env, "SAMPLE_MODE", False),
            log_level=log_level,
            port=_int(env, "PORT", 8000, 1, 65535),
            log_dir=_raw(env, "LOG_DIR") or "logs",
            data_dir=_raw(env, "DATA_DIR") or "data/sample",
            engine=engine,
            live_provider=live_provider,
            hmem_path=_raw(env, "HMEM_PATH") or ".hmem/store.json",
            meterless_base_url=base,
            meterless_runtime_token=_raw(env, "METERLESS_API_KEY", "MODEL_API_KEY"),
            meterless_autodetect=autodetect,
            model_id=_raw(env, "MODEL_ID") or "meterless/auto",
            provider_timeout_s=_float(env, "PROVIDER_TIMEOUT_S", 180.0, 0.001),
            source_timeout_s=_float(env, "SOURCE_TIMEOUT_S", 20.0, 0.001),
            zetaris_base_url=_raw(env, "ZETARIS_REST_URL", "ZETARIS_BASE_URL"),
            zetaris_org_id=_raw(env, "ZETARIS_ORG_ID") or "1",
            zetaris_api_key=_raw(env, "ZETARIS_API_KEY"),
            zetaris_table_orders=_raw(env, "ZETARIS_TABLE_ORDERS") or "orders",
            zetaris_table_tickets=_raw(env, "ZETARIS_TABLE_TICKETS") or "tickets",
            zetaris_table_shipments=_raw(env, "ZETARIS_TABLE_SHIPMENTS") or "shipments",
            nvidia_api_key=_raw(env, "NVIDIA_API_KEY"),
            run_timeout_s=_float(env, "RUN_TIMEOUT_S", 300.0, 0.001),
            max_refinements=_int(env, "MAX_REFINEMENTS", 2, 0, 20),
            max_steps=_int(env, "MAX_STEPS", 60, 1),
            max_model_calls=_int(env, "MAX_MODEL_CALLS", 12, 0),
        )

    def replace(self, **changes: Any) -> "Config":
        return dataclasses.replace(self, **changes)

    def with_options(self, options: Optional[dict]) -> "Config":
        """Apply request options (DESIGN §12.2). Raises InvalidRequest / ConfigError."""
        if not options:
            return self
        if not isinstance(options, dict):
            raise InvalidRequest("'options' must be an object")
        changes: dict[str, Any] = {}

        if "sample_mode" in options and options["sample_mode"] is not None:
            sm = options["sample_mode"]
            if not isinstance(sm, bool):
                raise InvalidRequest("options.sample_mode must be a boolean")
            changes["sample_mode"] = sm

        if "max_refinements" in options and options["max_refinements"] is not None:
            mr = options["max_refinements"]
            if isinstance(mr, bool) or not isinstance(mr, int) or mr < 0 or mr > 20:
                raise InvalidRequest("options.max_refinements must be an integer between 0 and 20")
            changes["max_refinements"] = mr

        mem = options.get("memory")
        if mem is not None:
            if not isinstance(mem, dict):
                raise InvalidRequest("options.memory must be an object")
            if "enabled" in mem and mem["enabled"] is not None:
                if not isinstance(mem["enabled"], bool):
                    raise InvalidRequest("options.memory.enabled must be a boolean")
                changes["memory_enabled"] = mem["enabled"]

        new = self.replace(**changes) if changes else self
        if changes.get("sample_mode") is False and self.sample_mode:
            new.validate_live()
        return new

    # ---- live-mode credentials ---------------------------------------
    @staticmethod
    def _is_placeholder(v: Optional[str]) -> bool:
        return v is None or v.strip().lower() in _PLACEHOLDERS

    def missing_live_credentials(self) -> list[str]:
        missing: list[str] = []
        if self._is_placeholder(self.zetaris_base_url) or "example" in (self.zetaris_base_url or "").lower().split("/")[2:3][0:1].__str__():
            missing.append("ZETARIS_REST_URL")
        if self._is_placeholder(self.zetaris_api_key):
            missing.append("ZETARIS_API_KEY")
        # A placeholder meterless token (e.g. rt-local) is reported by the
        # provider as ProviderUnauthorized, so only emptiness counts here.
        if self.live_provider == "meterless" and not self.meterless_runtime_token:
            missing.append("METERLESS_API_KEY")
        return missing

    def validate_live(self) -> None:
        missing = self.missing_live_credentials()
        if missing:
            raise ConfigError(
                "Live mode requires credentials: " + ", ".join(missing),
                details={"missing": missing},
            )

    # ---- helpers ------------------------------------------------------
    @property
    def zetaris_tables(self) -> dict[str, str]:
        return {
            "orders": self.zetaris_table_orders,
            "tickets": self.zetaris_table_tickets,
            "shipments": self.zetaris_table_shipments,
        }

    def secret_values(self) -> tuple[str, ...]:
        vals = (self.meterless_runtime_token, self.zetaris_api_key, self.nvidia_api_key)
        return tuple(v for v in vals if v)

    def public_dict(self) -> dict[str, Any]:
        d = dataclasses.asdict(self)
        d["meterless_runtime_token"] = _mask(self.meterless_runtime_token)
        d["zetaris_api_key"] = _mask(self.zetaris_api_key)
        d["nvidia_api_key"] = _mask(self.nvidia_api_key)
        return d

    def __repr__(self) -> str:
        d = self.public_dict()
        inner = ", ".join(f"{k}={v!r}" for k, v in d.items())
        return f"Config({inner})"

    __str__ = __repr__
