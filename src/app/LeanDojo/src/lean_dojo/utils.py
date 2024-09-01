"""Utility functions used internally by LeanDojo.
"""
import re
import os
import sys
import ray
import time
import urllib
import typing
import hashlib
import tempfile
import subprocess
from pathlib import Path
from loguru import logger
from datetime import datetime
from contextlib import contextmanager
from ray.util.actor_pool import ActorPool
from typing import Tuple, Union, List, Generator, Optional

from .constants import NUM_PROCS, NUM_WORKERS, TMP_DIR, LEAN4_DEPS_DIR


@contextmanager
def working_directory(
    path: Optional[Union[str, Path]] = None
) -> Generator[Path, None, None]:
    """Context manager setting the current working directory (CWD) to ``path`` (or a temporary directory if ``path`` is None).

    The original CWD is restored after the context manager exits.

    Args:
        path (Optional[Union[str, Path]], optional): The desired CWD. Defaults to None.

    Yields:
        Generator[Path, None, None]: A ``Path`` object representing the CWD.
    """
    origin = Path.cwd()
    if path is None:
        tmp_dir = tempfile.TemporaryDirectory(dir=TMP_DIR)
        path = tmp_dir.__enter__()
        is_temporary = True
    else:
        is_temporary = False

    path = Path(path)
    if not path.exists():
        path.mkdir(parents=True)
    os.chdir(path)

    try:
        yield path
    finally:
        os.chdir(origin)
        if is_temporary:
            tmp_dir.__exit__(None, None, None)


@contextmanager
def ray_actor_pool(
    actor_cls: type, *args, **kwargs
) -> Generator[ActorPool, None, None]:
    """Create a pool of Ray Actors of class ``actor_cls``.

    Args:
        actor_cls (type): A Ray Actor class (annotated by ``@ray.remote``).
        *args: Position arguments passed to ``actor_cls``.
        **kwargs: Keyword arguments passed to ``actor_cls``.

    Yields:
        Generator[ActorPool, None, None]: A :class:`ray.util.actor_pool.ActorPool` object.
    """
    assert not ray.is_initialized()
    ray.init(num_cpus=NUM_PROCS, num_gpus=0)
    pool = ActorPool([actor_cls.remote(*args, **kwargs) for _ in range(NUM_WORKERS)])
    try:
        yield pool
    finally:
        ray.shutdown()


@contextmanager
def report_critical_failure(msg: str) -> Generator[None, None, None]:
    """Context manager logging ``msg`` in case of any exception.

    Args:
        msg (str): The message to log in case of exceptions.

    Raises:
        ex: Any exception that may be raised within the context manager.
    """
    try:
        yield
    except Exception as ex:
        logger.error(msg)
        raise ex


def execute(
    cmd: Union[str, List[str]], capture_output: bool = False
) -> Optional[Tuple[str, str]]:
    """Execute the shell command ``cmd`` and optionally return its output.

    Args:
        cmd (Union[str, List[str]]): The shell command to execute.
        capture_output (bool, optional): Whether to capture and return the output. Defaults to False.

    Returns:
        Optional[Tuple[str, str]]: The command's output, including stdout and stderr (None if ``capture_output == False``).
    """
    try:
        res = subprocess.run(cmd, shell=True, capture_output=capture_output, check=True)
    except subprocess.CalledProcessError as ex:
        if capture_output:
            logger.info(ex.stdout.decode())
            logger.error(ex.stderr.decode())
        raise ex
    if not capture_output:
        return None
    output = res.stdout.decode()
    error = res.stderr.decode()
    return output, error


def compute_md5(path: Path) -> str:
    """Return the MD5 hash of the file ``path``."""
    # The file could be large
    # See: https://stackoverflow.com/questions/48122798/oserror-errno-22-invalid-argument-when-reading-a-huge-file
    hasher = hashlib.md5()
    with path.open("rb") as inp:
        while True:
            block = inp.read(64 * (1 << 20))
            if not block:
                break
            hasher.update(block)
    return hasher.hexdigest()


_CAMEL_CASE_REGEX = re.compile(r"(_|-)+")


def camel_case(s: str) -> str:
    """Convert the string ``s`` to camel case."""
    return _CAMEL_CASE_REGEX.sub(" ", s).title().replace(" ", "")


def set_lean_dojo_logger(verbose: bool) -> None:
    """Set loguru's logging level.

    The effect of this function is global, and it should be called only once in the main function.

    Args:
        verbose (bool): Whether to print debug information.
    """
    logger.remove()
    if verbose:
        logger.add(sys.stderr, level="DEBUG")
    else:
        logger.add(sys.stderr, level="INFO")


def get_repo_info(path: Path) -> Tuple[str, str]:
    """Get the URL and commit hash of the Git repo at ``path``.

    Args:
        path (Path): Path to the Git repo.

    Returns:
        Tuple[str, str]: URL and (most recent) hash commit
    """
    with working_directory(path):
        # Get the URL.
        url_msg, _ = execute(f"git remote get-url origin", capture_output=True)
        url = url_msg.strip()
        # Get the commit.
        commit_msg, _ = execute(f"git log -n 1", capture_output=True)
        m = re.search(r"(?<=^commit )[a-z0-9]+", commit_msg)
        assert m is not None
        commit = m.group()

    if url.startswith("git@"):
        assert url.endswith(".git")
        url = url[: -len(".git")].replace(":", "/").replace("git@", "https://")

    return url, commit


_SPACES_REGEX = re.compile(r"\s+", re.DOTALL)


def normalize_spaces(s: str) -> str:
    """Replace any consecutive block of whitespace characters in ``s`` with a single whitespace."""
    return _SPACES_REGEX.sub(" ", s).strip()


def is_optional_type(tp: type) -> bool:
    """Test if ``tp`` is Optional[X]."""
    if typing.get_origin(tp) != Union:
        return False
    args = typing.get_args(tp)
    return len(args) == 2 and args[1] == type(None)


def remove_optional_type(tp: type) -> type:
    """Given Optional[X], return X."""
    if typing.get_origin(tp) != Union:
        return False
    args = typing.get_args(tp)
    if len(args) == 2 and args[1] == type(None):
        return args[0]
    else:
        raise ValueError(f"{tp} is not Optional")


def get_line_creation_date(repo_path: Path, file_path: Path, line_nb: int):
    """Return the date of creation of the line ``line_nb`` in the file ``file_path`` of the Git repo ``repo_path``."""
    with working_directory(repo_path):
        output, _ = execute(
            f"git log --pretty=short -u -L {line_nb},{line_nb}:{file_path}",
            capture_output=True,
        )
        m = re.match(r"commit (?P<commit>[A-Za-z0-9]+)\n", output)
        assert m is not None
        commit = m["commit"]
        output, _ = execute(f"git show -s --format=%ci {commit}", capture_output=True)
        return datetime.strptime(output.strip(), "%Y-%m-%d %H:%M:%S %z")


def read_url(url: str, num_retries: int = 1) -> str:
    """Read the contents of the URL ``url``. Retry if failed"""
    while True:
        try:
            with urllib.request.urlopen(url) as f:
                return f.read().decode()
        except Exception as ex:
            if num_retries <= 0:
                raise ex
            num_retries -= 1
            logger.debug(f"Request to {url} failed. Retrying...")
            time.sleep(2 - num_retries)


def url_exists(url: str) -> bool:
    """Return True if the URL ``url`` exists."""
    try:
        with urllib.request.urlopen(url) as _:
            return True
    except urllib.error.HTTPError:
        return False


def parse_lean3_version(v: str) -> Tuple[int, int, int]:
    assert v.startswith("v")
    return tuple(int(_) for _ in v[1:].split("."))


def parse_int_list(s: str) -> List[int]:
    assert s.startswith("[") and s.endswith("]")
    return [int(_) for _ in s[1:-1].split(",") if _ != ""]


def parse_str_list(s: str) -> List[str]:
    assert s.startswith("[") and s.endswith("]")
    return [_.strip()[1:-1] for _ in s[1:-1].split(",") if _ != ""]


def get_latest_commit(url: str) -> str:
    """Get the hash of the latest commit of the Git repo at ``url``."""
    url = os.path.join(url, "commits")
    html = read_url(url)
    m = re.search(r"commits/(?P<commit>[a-z0-9]+)/commits_list_item", html)
    assert len(m["commit"]) == 40
    return m["commit"]


def is_git_repo(path: Path) -> bool:
    """Check if ``path`` is a Git repo."""
    with working_directory(path):
        return (
            os.system("git rev-parse --is-inside-work-tree 1>/dev/null 2>/dev/null")
            == 0
        )


def _from_lean_path(root_dir: Path, path: Path, uses_lean4: bool, ext: str) -> Path:
    assert path.suffix == ".lean"
    if path.is_absolute():
        path = path.relative_to(root_dir)

    if not uses_lean4:
        return path.with_suffix(ext)

    if root_dir.name == "lean4":
        return Path("lib") / path.relative_to("src").with_suffix(ext)
    elif path.is_relative_to(LEAN4_DEPS_DIR / "lean4/src"):
        # E.g., "lake-packages/lean4/src/lean/Init.lean"
        p = path.relative_to(LEAN4_DEPS_DIR / "lean4/src").with_suffix(ext)
        return LEAN4_DEPS_DIR / "lean4/lib" / p
    elif path.is_relative_to(LEAN4_DEPS_DIR):
        # E.g., "lake-packages/std/Std.lean"
        p = path.relative_to(LEAN4_DEPS_DIR).with_suffix(ext)
        repo_name = p.parts[0]
        return LEAN4_DEPS_DIR / repo_name / "build/ir" / p.relative_to(repo_name)
    else:
        # E.g., "Mathlib/LinearAlgebra/Basics.lean"
        return Path("build/ir") / path.with_suffix(ext)


def to_xml_path(root_dir: Path, path: Path, uses_lean4: bool) -> Path:
    return _from_lean_path(root_dir, path, uses_lean4, ext=".trace.xml")


def to_dep_path(root_dir: Path, path: Path, uses_lean4: bool) -> Path:
    return _from_lean_path(root_dir, path, uses_lean4, ext=".dep_paths")


def to_json_path(root_dir: Path, path: Path, uses_lean4: bool) -> Path:
    return _from_lean_path(root_dir, path, uses_lean4, ext=".ast.json")


def to_lean_path(root_dir: Path, path: Path, uses_lean4: bool) -> bool:
    if path.is_absolute():
        path = path.relative_to(root_dir)

    if path.suffix in (".xml", ".json"):
        path = path.with_suffix("").with_suffix(".lean")
    else:
        assert path.suffix == ".dep_paths"
        path = path.with_suffix(".lean")

    if not uses_lean4:
        return path

    if root_dir.name == "lean4":
        return Path("src") / path.relative_to("lib")
    elif path.is_relative_to(LEAN4_DEPS_DIR / "lean4/lib"):
        # E.g., "lake-packages/lean4/lib/lean/Init.lean"
        p = path.relative_to(LEAN4_DEPS_DIR / "lean4/lib")
        return LEAN4_DEPS_DIR / "lean4/src" / p
    elif path.is_relative_to(LEAN4_DEPS_DIR):
        # E.g., "lake-packages/std/build/ir/Std.lean"
        p = path.relative_to(LEAN4_DEPS_DIR)
        repo_name = p.parts[0]
        return LEAN4_DEPS_DIR / repo_name / p.relative_to(Path(repo_name) / "build/ir")
    else:
        # E.g., "build/ir/Mathlib/LinearAlgebra/Basics.lean"
        assert path.is_relative_to("build/ir"), path
        return path.relative_to("build/ir")
