"""Read-only dev-host CUDA-owner guard. Shareable by callers; no GPU query or signals.

stdin: newline-separated PIDs from nvidia-smi --query-compute-apps=pid.
Only the two explicitly authorized resident Python modules may coexist.
Never allow by PID, substring, interpreter alone, or a module in other arguments.
"""
import os
from pathlib import Path
import re
import sys

MODULES = frozenset({"service.cotenant_a.cotenant_a", "service.cotenant_b.cotenant_b"})


class Refusal(RuntimeError):
    pass


def allowed_module(argv):
    if (len(argv) >= 3 and re.fullmatch(r"python(?:3(?:\.\d+)?)?", os.path.basename(argv[0]))
            and argv[1] == "-m" and argv[2] in MODULES):
        return argv[2]
    return None


def proc_argv(pid):
    with (Path("/proc") / pid / "cmdline").open("rb") as source:
        raw = source.read(1024 * 1024 + 1)
    if not raw or len(raw) > 1024 * 1024 or not raw.endswith(b"\0"):
        raise Refusal(f"unreadable/malformed command line for CUDA owner pid={pid}")
    return [arg.decode("utf-8", errors="surrogateescape") for arg in raw[:-1].split(b"\0")]


def check_owners(text, reader=proc_argv, emit=print):
    for pid in dict.fromkeys(text.split()):
        if not re.fullmatch(r"[1-9][0-9]*", pid):
            raise Refusal("invalid PID returned by CUDA ownership query")
        try:
            argv = reader(pid)
        except OSError as exc:
            # A vanished process or inaccessible /proc entry is not permission.
            raise Refusal(f"cannot inspect CUDA owner pid={pid}: {type(exc).__name__}") from exc
        module = allowed_module(argv)
        if module is None:
            # Do not disclose arbitrary process arguments (they may contain secrets).
            raise Refusal(f"Refuse concurrent CUDA owner pid={pid}: not an authorized resident module")
        emit(f"ALLOW resident CUDA owner pid={pid} module={module}")


def selftest():
    eyes = ["/opt/venvs/cotenant_a/bin/python", "-m", "service.cotenant_a.cotenant_a"]
    ears = ["/opt/venvs/cotenant_b/bin/python", "-m", "service.cotenant_b.cotenant_b"]
    quiet = lambda _: None
    check_owners("", lambda _: (_ for _ in ()).throw(AssertionError("empty list read /proc")), quiet)
    for a, b in [("11", "22"), ("431", "9831")]:
        processes = {a: eyes, b: ears}
        check_owners(f"{a}\n{b}\n{a}\n", processes.__getitem__, quiet)
    assert allowed_module(["/venv/bin/python3.13", "-m", "service.cotenant_b.cotenant_b"]) is not None
    bad = [
        ["/tmp/attn-parity", "bench"],
        ["python", "-m", "other.module"],
        ["python", "-c", "import service.cotenant_a.cotenant_a"],
        ["python", "bench.py", "-m", "service.cotenant_a.cotenant_a"],
        ["python", "-m", "service.cotenant_a.cotenant_a_bench"],
        ["python", "-m", "service.cotenant_a.cotenant_a.extra"],
        ["not-python", "-m", "service.cotenant_a.cotenant_a"],
        [],
    ]
    for argv in bad:
        try:
            check_owners("11\n33", {"11": eyes, "33": argv}.__getitem__, quiet)
        except Refusal:
            pass
        else:
            raise AssertionError("unknown CUDA owner accepted")
    for bad_pid in ["N/A", "0", "-1", "123,", "../11"]:
        try:
            check_owners(bad_pid, lambda _: eyes, quiet)
        except Refusal:
            pass
        else:
            raise AssertionError("malformed PID accepted")
    for error in [PermissionError(), FileNotFoundError()]:
        def unreadable(_):
            raise error
        try:
            check_owners("11", unreadable, quiet)
        except Refusal:
            pass
        else:
            raise AssertionError("uninspectable CUDA owner accepted")
    print("HOST PASS owner guard: empty/residents/restarted PIDs; 8 other-owner traps; 5 malformed PIDs; 2 inspection failures")


if __name__ == "__main__":
    try:
        if sys.argv[1:] == ["--selftest"]:
            selftest()
        elif len(sys.argv) == 1:
            check_owners(sys.stdin.read())
        else:
            raise Refusal("usage: cuda_owner_guard.py [--selftest]")
    except Refusal as exc:
        print(str(exc), file=sys.stderr)
        sys.exit(2)
