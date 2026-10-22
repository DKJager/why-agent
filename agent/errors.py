"""Structured error hierarchy and the error JSON contract (DESIGN §13)."""

from __future__ import annotations

from typing import Any, Callable, Optional


class WhyAgentError(Exception):
    """Base class of all expected, structured errors."""

    type: str = "WhyAgentError"
    recoverable: bool = True

    def __init__(
        self,
        message: str = "",
        *,
        details: Optional[dict] = None,
        recoverable: Optional[bool] = None,
    ) -> None:
        super().__init__(message)
        self.message = str(message)
        self.details: dict = dict(details) if details else {}
        if recoverable is not None:
            self.recoverable = bool(recoverable)

    def to_dict(self, redact: Optional[Callable[[str], str]] = None) -> dict:
        """The ``error`` object of the error contract."""
        msg = self.message
        if redact is not None:
            msg = redact(msg)
        return {
            "type": self.type,
            "message": msg,
            "recoverable": bool(self.recoverable),
            "details": dict(self.details),
        }

    def __str__(self) -> str:  # pragma: no cover - trivial
        return self.message


class InvalidRequest(WhyAgentError):
    type = "InvalidRequest"
    recoverable = True


class QuestionNotSupported(WhyAgentError):
    type = "QuestionNotSupported"
    recoverable = True


class ConfigError(WhyAgentError):
    type = "ConfigError"
    recoverable = True


class ProviderUnauthorized(WhyAgentError):
    type = "ProviderUnauthorized"
    recoverable = True


class ProviderUnavailable(WhyAgentError):
    type = "ProviderUnavailable"
    recoverable = True


class SourceUnauthorized(WhyAgentError):
    type = "SourceUnauthorized"
    recoverable = True


class SourceUnavailable(WhyAgentError):
    type = "SourceUnavailable"
    recoverable = True


class SourceQueryError(WhyAgentError):
    type = "SourceQueryError"
    recoverable = False


class UpstreamTimeout(WhyAgentError):
    type = "UpstreamTimeout"
    recoverable = True


class EngineNotConfigured(WhyAgentError):
    type = "EngineNotConfigured"
    recoverable = True


class NoData(WhyAgentError):
    type = "NoData"
    recoverable = True


class IterationLimit(WhyAgentError):
    type = "IterationLimit"
    recoverable = True


class RunTimeout(WhyAgentError):
    type = "RunTimeout"
    recoverable = True


class InternalError(WhyAgentError):
    type = "InternalError"
    recoverable = False

    def __init__(
        self,
        message: str = "Internal error",
        *,
        details: Optional[dict] = None,
        recoverable: Optional[bool] = None,
        cause_type: Optional[str] = None,
    ) -> None:
        super().__init__(message, details=details, recoverable=recoverable)
        # Exception class name: goes into the trace details only, never the body.
        self.cause_type = cause_type


# Error type -> HTTP status (DESIGN §12.2).
HTTP_STATUS: dict[str, int] = {
    "InvalidRequest": 400,
    "QuestionNotSupported": 422,
    "NoData": 422,
    "ProviderUnauthorized": 502,
    "SourceUnauthorized": 502,
    "ProviderUnavailable": 503,
    "SourceUnavailable": 503,
    "EngineNotConfigured": 503,
    "UpstreamTimeout": 504,
    "RunTimeout": 504,
    "ConfigError": 500,
    "InternalError": 500,
    "IterationLimit": 500,
    "SourceQueryError": 500,
}


def http_status_for(error_type: str) -> int:
    return HTTP_STATUS.get(error_type, 500)


def wrap_unexpected(exc: BaseException) -> WhyAgentError:
    """Convert any exception into a WhyAgentError (no traceback, no secrets)."""
    if isinstance(exc, WhyAgentError):
        return exc
    return InternalError(
        "Internal error while processing the request.",
        cause_type=type(exc).__name__,
    )


def error_body(
    exc: BaseException,
    *,
    run_id: Optional[str],
    trace_id: Optional[str],
    log_file: Optional[str],
    execution_time_seconds: float,
    redact: Optional[Callable[[str], str]] = None,
) -> dict[str, Any]:
    """Build the structured error JSON of DESIGN §13."""
    err = wrap_unexpected(exc)
    return {
        "run_id": run_id,
        "status": "error",
        "error": err.to_dict(redact),
        "trace_id": trace_id,
        "log_file": log_file,
        "execution_time_seconds": round(float(execution_time_seconds), 3),
    }
