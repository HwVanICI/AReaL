"""Utilities for SWE-bench agent training with AReaL."""

from dataclasses import dataclass, field
from typing import Any

from areal.api.cli_args import PPOConfig


@dataclass
class SWEEnvConfig:
    """Environment configuration for AReaL-SWEAgent-backed SWE-bench training.

    Attributes:
        dataset_path: Path to the SWE-bench JSONL dataset file.
        agent_type: AReaL-SWEAgent agent type to train: ``swe``, ``cc``,
            ``codex``, or ``opencode``.
        sandbox_backend: Remote sandbox provider, ``aenv`` or ``e2b``.
        agent_config: Generic AReaL-SWEAgent config name. When set, this overrides
            the compatibility fields below.
        swe_agent_config: Compatibility config field for ``agent_type=swe``.
        cc_agent_config: Compatibility config field for ``agent_type=cc``.
        codex_agent_config: Config field for ``agent_type=codex``.
        opencode_agent_config: Config field for ``agent_type=opencode``.
        agent_root: Root directory of the bundled or external AReaL-SWEAgent checkout.
        swe_agent_root: Legacy alias for ``agent_root``.
        llm_model: Optional LLM model override for OH/OpenCode/Codex agents.
        opencode_provider: Optional OpenCode provider override.
        opencode_config: Additional OpenCode configuration merged into opencode.json.
        codex_provider: Optional Codex provider override.
        step_limit: Maximum number of agent interaction steps per episode.
        max_tokens: Maximum context window advertised to the agent LLM.
        max_completion_tokens: Maximum completion tokens for the agent LLM.
        timeout: Hard wall-clock deadline for the complete episode.
        harness_timeout: Optional wall-clock override for a delegated E2B harness.
        eval_timeout: Wall-clock limit for fresh-sandbox reward evaluation.
        session_idle_timeout: Kill a delegated E2B harness after this much proxy
            inactivity.
        session_poll_interval: Seconds between proxy session status checks.
        terminal_exit_grace: Grace after a terminal model response for harnesses
            that declare such a response final.
    """

    dataset_path: str = field(
        default="",
        metadata={"help": "Path to the SWE-bench JSONL dataset file."},
    )
    agent_type: str = field(
        default="swe",
        metadata={
            "help": (
                "AReaL-SWEAgent agent type to run: 'swe', 'cc', 'codex', or "
                "'opencode'. Codex and OpenCode require sandbox_backend='e2b'."
            )
        },
    )
    sandbox_backend: str = field(
        default="aenv",
        metadata={
            "help": (
                "Sandbox backend used by AReaL-SWEAgent. E2B supports Claude Code, "
                "Codex, and OpenCode; AEnvironment supports SWE and Claude Code."
            )
        },
    )
    agent_config: str = field(
        default="",
        metadata={
            "help": (
                "Generic AReaL-SWEAgent YAML config name. When non-empty, overrides "
                "swe_agent_config / cc_agent_config / codex_agent_config / "
                "opencode_agent_config."
            )
        },
    )
    swe_agent_config: str = field(
        default="1_0_0/min-swe-agent-train-top1",
        metadata={
            "help": (
                "Name of the AReaL-SWEAgent YAML config under the external "
                "AReaL-SWEAgent checkout. Defaults to the Qwen SWE-RL training config."
            )
        },
    )
    cc_agent_config: str = field(
        default="train_cc_time3600",
        metadata={
            "help": (
                "Name of the AReaL-SWEAgent YAML config used when agent_type='cc'. "
                "Kept separate for compatibility with swe/main configs."
            )
        },
    )
    codex_agent_config: str = field(
        default="train_codex_time3600",
        metadata={
            "help": (
                "Name of the AReaL-SWEAgent YAML config used when agent_type='codex'."
            )
        },
    )
    opencode_agent_config: str = field(
        default="train_opencode_time3600",
        metadata={
            "help": (
                "Name of the AReaL-SWEAgent YAML config used when "
                "agent_type='opencode'."
            )
        },
    )
    agent_root: str = field(
        default="",
        metadata={
            "help": (
                "Root directory of the AReaL-SWEAgent checkout. Defaults to a bundled "
                "checkout, then ../AReaL-SWEAgent, when unset."
            )
        },
    )
    swe_agent_root: str = field(
        default="",
        metadata={
            "help": (
                "Legacy alias for agent_root / AWEAGENT_ROOT. Kept so older "
                "SWE launch scripts keep working."
            )
        },
    )
    llm_model: str = field(
        default="",
        metadata={"help": "Optional model name override for OH/OpenCode/Codex agents."},
    )
    opencode_provider: str = field(
        default="",
        metadata={"help": "Optional provider override for OpenCode agents."},
    )
    opencode_config: dict[str, Any] = field(
        default_factory=dict,
        metadata={
            "help": "Additional OpenCode configuration merged into opencode.json."
        },
    )
    codex_provider: str = field(
        default="",
        metadata={"help": "Optional provider override for Codex agents."},
    )
    step_limit: int = field(
        default=100,
        metadata={"help": "Maximum number of agent interaction steps per episode."},
    )
    max_tokens: int = field(
        default=32768,
        metadata={"help": "Maximum context window advertised to the agent LLM."},
    )
    max_completion_tokens: int = field(
        default=16384,
        metadata={"help": "Maximum completion tokens for the agent LLM."},
    )
    timeout: float = field(
        default=1800.0,
        metadata={"help": "Hard wall-clock deadline for the complete episode."},
    )
    harness_timeout: float | None = field(
        default=None,
        metadata={
            "help": (
                "Optional wall-clock override for a delegated E2B harness process; "
                "the selected agent config supplies the default when unset."
            )
        },
    )
    eval_timeout: float = field(
        default=900.0,
        metadata={"help": "Wall-clock limit for fresh-sandbox evaluation."},
    )
    session_idle_timeout: float | None = field(
        default=1200.0,
        metadata={
            "help": "Delegated E2B proxy-activity watchdog timeout; null disables it."
        },
    )
    session_poll_interval: float = field(
        default=30.0,
        metadata={"help": "Delegated E2B session-status polling interval."},
    )
    terminal_exit_grace: float | None = field(
        default=15.0,
        metadata={
            "help": (
                "Grace after a harness-declared terminal model response; null "
                "disables terminal-response teardown."
            )
        },
    )


@dataclass
class SWEPPOConfig(PPOConfig):
    """PPO configuration with SWE-bench-specific settings."""

    econfig: SWEEnvConfig = field(default_factory=SWEEnvConfig)
    should_accept_fn: str | None = field(
        default=None,
        metadata={
            "help": "Import path of the filter function for accepting rollout samples."
        },
    )
