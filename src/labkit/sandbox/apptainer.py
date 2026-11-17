"""Apptainer sandbox for Inspect (`sandbox="apptainer"`), for clusters without Docker.

One persistent `apptainer instance` per sample, started with
  --contain --cleanenv            no host $HOME / /tmp / cwd, no host env vars (API keys stay out)
  --no-mount bind-paths,cwd       no site-configured host binds (/projects, /scratch, ...)
  --net --network none            no network unless the config says network: true
  --overlay <host dir>            the image is writable (disk-backed on node-local scratch)
  --bind <host dir>:/workspace    roomy scratch dir
  (both live under one per-sample dir in $LABKIT_SANDBOX_TMP or /tmp, deleted after the sample)

Config (task `sandbox=("apptainer", <config>)` or CLI `--sandbox apptainer:<config>`):
  None                        default image (images/python-3.12-slim.sif)
  "path/to/image.sif"         a local SIF
  "docker://python:3.12"      pulled once into images/ as a SIF
  "compose.yaml"              a Docker compose file of an existing benchmark: the `default`
                              (or first) service's `image:` is pulled; `build:` is not supported
                              (build a SIF yourself and pass it). network_mode != none enables network.
  "config.yaml"               labkit config file: {image: ..., network: false, binds: [...]}
  ApptainerConfig(...)        the same, in Python
"""
from __future__ import annotations

import asyncio
import base64
import os
import re
import shlex
import shutil
import tempfile
import uuid
from pathlib import Path
from typing import Any, Literal, overload

import yaml
from pydantic import BaseModel
from inspect_ai.util import (
    ExecResult,
    OutputLimitExceededError,
    SandboxConnection,
    SandboxEnvironment,
    SandboxEnvironmentConfigType,
    SandboxEnvironmentLimits,
    sandboxenv,
    subprocess,
)

ROOT = Path(__file__).resolve().parents[3]
IMAGES = Path(os.environ.get("LABKIT_IMAGES", ROOT / "images"))
DEFAULT_IMAGE = IMAGES / "python-3.12-slim.sif"
INSTANCE_PREFIX = "labkit-"
WORKSPACE = "/workspace"

# Apptainer passes the host's proxy variables into the container even with --cleanenv. Clear them:
# the sandbox must not know the cluster proxy, and HTTP clients (e.g. Codex) would otherwise send
# even localhost traffic to it.
NO_PROXY_ENV = {k: "" for k in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY",
                                "all_proxy", "ALL_PROXY", "ftp_proxy", "FTP_PROXY")}
NO_PROXY_ENV.update({"no_proxy": "*", "NO_PROXY": "*"})


class ApptainerConfig(BaseModel, frozen=True):
    image: str = str(DEFAULT_IMAGE)
    network: bool = False
    binds: tuple[str, ...] = ()  # extra "host:container[:ro]" binds, e.g. read-only task data
    writable_tmpfs: bool = True


def _apptainer_env() -> dict[str, str]:
    env = dict(os.environ)
    env.setdefault("APPTAINER_CACHEDIR", str(ROOT / ".apptainer_cache"))
    return env


def _parse_config(config: SandboxEnvironmentConfigType | None) -> ApptainerConfig:
    if config is None:
        return ApptainerConfig()
    if isinstance(config, ApptainerConfig):
        return config
    if isinstance(config, BaseModel):
        return ApptainerConfig(**config.model_dump())
    if not isinstance(config, str):
        raise ValueError(f"apptainer sandbox: unsupported config {config!r}")
    if config.startswith(("docker://", "library://", "oras://")) or config.endswith(".sif"):
        return ApptainerConfig(image=config)
    path = Path(config)
    if path.name.lower() in ("dockerfile",) or path.suffix.lower() == ".dockerfile":
        raise ValueError("apptainer sandbox: Dockerfiles are not supported; build a SIF and pass its path.")
    data = yaml.safe_load(path.read_text()) or {}
    if "services" in data:  # docker compose file
        services = data["services"]
        svc = services.get("default") or next(iter(services.values()))
        if "image" not in svc:
            raise ValueError(f"apptainer sandbox: {config} builds its image; build a SIF and pass it instead.")
        image = svc["image"]
        network = svc.get("network_mode", "none") != "none"
        return ApptainerConfig(image=image if "://" in image else f"docker://{image}", network=network)
    return ApptainerConfig(**data)


_pull_lock = asyncio.Lock()


async def _ensure_sif(image: str) -> str:
    """Local SIF path for an image reference, pulling it once if needed."""
    if image.endswith(".sif") and "://" not in image:
        if not Path(image).exists():
            raise FileNotFoundError(f"apptainer sandbox: image not found: {image}")
        return image
    name = re.sub(r"[^A-Za-z0-9._-]+", "_", image.split("://", 1)[-1]) + ".sif"
    sif = IMAGES / name
    async with _pull_lock:
        if not sif.exists():
            IMAGES.mkdir(parents=True, exist_ok=True)
            tmp = sif.with_suffix(".sif.partial")
            r = await subprocess(["apptainer", "pull", "-F", str(tmp), image], env=_apptainer_env(), timeout=1800)
            if not r.success:
                raise RuntimeError(f"apptainer pull {image} failed:\n{r.stderr}")
            tmp.rename(sif)
    return str(sif)


@sandboxenv(name="apptainer")
class ApptainerSandboxEnvironment(SandboxEnvironment):
    def __init__(self, instance: str, workdir: str) -> None:
        super().__init__()
        self.instance = instance
        self.workdir = workdir

    # ---- lifecycle ---------------------------------------------------------------------------
    @classmethod
    def config_files(cls) -> list[str]:
        return ["compose.yaml", "compose.yml", "docker-compose.yaml", "docker-compose.yml"]

    @classmethod
    def default_concurrency(cls) -> int | None:
        return max(1, (os.cpu_count() or 4))

    @classmethod
    def config_deserialize(cls, config: dict[str, Any]) -> BaseModel:
        return ApptainerConfig(**config)

    @classmethod
    async def task_init(cls, task_name: str, config: SandboxEnvironmentConfigType | None) -> None:
        if shutil.which("apptainer") is None:
            raise RuntimeError("apptainer sandbox: `apptainer` is not on PATH")
        await _ensure_sif(_parse_config(config).image)

    @classmethod
    async def sample_init(
        cls, task_name: str, config: SandboxEnvironmentConfigType | None, metadata: dict[str, str]
    ) -> dict[str, SandboxEnvironment]:
        cfg = _parse_config(config)
        sif = await _ensure_sif(cfg.image)
        base = os.environ.get("LABKIT_SANDBOX_TMP") or os.environ.get("TMPDIR") or "/tmp"
        workdir = tempfile.mkdtemp(prefix="labkit-ws-", dir=base)
        dirs = {k: os.path.join(workdir, k) for k in ("ws", "overlay", "tmp", "vartmp", "home")}
        for d in dirs.values():
            os.makedirs(d)
        ws, overlay = dirs["ws"], dirs["overlay"]
        instance = f"{INSTANCE_PREFIX}{uuid.uuid4().hex[:12]}"
        # --contain puts /tmp, /var/tmp and $HOME on 64 MB session tmpfs mounts; back them with
        # per-sample host dirs instead (agent CLIs stage 100+ MB archives there)
        home_in = os.path.expanduser("~")
        # `--no-mount bind-paths` also drops the site's /etc/hosts bind, which leaves `localhost`
        # unresolvable (agent bridges listen on localhost); provide a minimal loopback-only one
        hosts = os.path.join(workdir, "hosts")
        with open(hosts, "w") as f:
            f.write("127.0.0.1\tlocalhost\n::1\tlocalhost ip6-localhost ip6-loopback\n")
        args = ["apptainer", "instance", "start", "--contain", "--cleanenv", "--no-mount", "bind-paths,cwd",
                "--bind", f"{ws}:{WORKSPACE}", "--bind", f"{dirs['tmp']}:/tmp",
                "--bind", f"{dirs['vartmp']}:/var/tmp", "--bind", f"{hosts}:/etc/hosts:ro",
                "--home", f"{dirs['home']}:{home_in}"]
        if not cfg.network:
            args += ["--net", "--network", "none"]
        if cfg.writable_tmpfs:
            # disk-backed writable layer on node-local scratch (a tmpfs is capped at 64 MB, too small
            # for agent binaries such as Codex's 200 MB package); deleted with the sample
            args += ["--overlay", overlay]
        for b in cfg.binds:
            args += ["--bind", b]
        r = await subprocess(args + [sif, instance], env=_apptainer_env(), timeout=300)
        if not r.success:
            shutil.rmtree(workdir, ignore_errors=True)
            raise RuntimeError(f"apptainer instance start failed:\n{r.stderr}")
        return {"default": cls(instance, workdir)}

    @classmethod
    async def sample_cleanup(
        cls, task_name: str, config: SandboxEnvironmentConfigType | None,
        environments: dict[str, SandboxEnvironment], interrupted: bool,
    ) -> None:
        for env in environments.values():
            assert isinstance(env, ApptainerSandboxEnvironment)
            await subprocess(["apptainer", "instance", "stop", "-t", "10", env.instance],
                             env=_apptainer_env(), timeout=60)
            shutil.rmtree(env.workdir, ignore_errors=True)

    @classmethod
    async def task_cleanup(cls, task_name: str, config: SandboxEnvironmentConfigType | None, cleanup: bool) -> None:
        pass

    @classmethod
    async def cli_cleanup(cls, id: str | None) -> None:
        """`inspect sandbox cleanup apptainer [instance]`: stop leftover labkit instances."""
        if id:
            names = [id]
        else:
            r = await subprocess(["apptainer", "instance", "list"], env=_apptainer_env())
            names = [line.split()[0] for line in r.stdout.splitlines()[1:]
                     if line.split() and line.split()[0].startswith(INSTANCE_PREFIX)]
        for name in names:
            await subprocess(["apptainer", "instance", "stop", name], env=_apptainer_env())
            print(f"stopped {name}")

    # ---- operations --------------------------------------------------------------------------
    async def exec(
        self,
        cmd: list[str],
        input: str | bytes | None = None,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        user: str | None = None,
        timeout: int | None = None,
        timeout_retry: bool = True,
        concurrency: bool = True,
    ) -> ExecResult[str]:
        if user is not None and user not in ("root", os.environ.get("USER", "")):
            raise PermissionError(f"apptainer sandbox runs as the host user; cannot run as {user!r}")
        args = ["apptainer", "exec", "--cleanenv", "--pwd", cwd or WORKSPACE]
        for k, v in {**(env or {}), **NO_PROXY_ENV}.items():  # ours last: always wins
            args += ["--env", f"{k}={v}"]
        args += [f"instance://{self.instance}", *cmd]
        result = await subprocess(
            args, input=input, timeout=timeout, env=_apptainer_env(),
            output_limit=SandboxEnvironmentLimits.MAX_EXEC_OUTPUT_SIZE,
        )
        return result

    async def write_file(self, file: str, contents: str | bytes) -> None:
        data = contents.encode() if isinstance(contents, str) else contents
        path = file if file.startswith("/") else f"{WORKSPACE}/{file}"
        script = 'mkdir -p "$(dirname "$1")" && base64 -d > "$1"'
        r = await self.exec(["sh", "-c", script, "sh", path], input=base64.b64encode(data))
        if not r.success:
            if "Permission denied" in r.stderr:
                raise PermissionError(r.stderr)
            if "Is a directory" in r.stderr:
                raise IsADirectoryError(r.stderr)
            raise RuntimeError(f"write_file {path} failed: {r.stderr}")

    @overload
    async def read_file(self, file: str, text: Literal[True] = True) -> str: ...
    @overload
    async def read_file(self, file: str, text: Literal[False]) -> bytes: ...
    async def read_file(self, file: str, text: bool = True) -> str | bytes:
        path = file if file.startswith("/") else f"{WORKSPACE}/{file}"
        script = ('[ -d "$1" ] && { echo "Is a directory" >&2; exit 21; }; '
                  '[ -e "$1" ] || { echo "No such file" >&2; exit 2; }; base64 "$1"')
        r = await self.exec(["sh", "-c", script, "sh", path])
        if r.returncode == 2:
            raise FileNotFoundError(path)
        if r.returncode == 21:
            raise IsADirectoryError(path)
        if not r.success:
            if "Permission denied" in r.stderr:
                raise PermissionError(path)
            raise RuntimeError(f"read_file {path} failed: {r.stderr}")
        data = base64.b64decode(r.stdout)
        if len(data) > SandboxEnvironmentLimits.MAX_READ_FILE_SIZE:
            raise OutputLimitExceededError(limit_str=SandboxEnvironmentLimits.MAX_READ_FILE_SIZE_STR,
                                           truncated_output=None)
        return data.decode("utf-8") if text else data

    async def connection(self, *, user: str | None = None) -> SandboxConnection:
        return SandboxConnection(
            type="apptainer",
            command=f"apptainer shell instance://{self.instance}",
            vscode_command=None,
        )
