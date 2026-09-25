#!/usr/bin/env python3
"""CPU-only tests of the production Bash guard; no real driver or /proc reads."""
import os
import pathlib
import subprocess
import unittest

GUARD = pathlib.Path(__file__).with_name("gpu_guard.sh")
MOCK = r'''
nvidia-smi() {
  case "$*" in
    *--query-compute-apps=pid*) [[ "$OWNER_ERROR" == 0 ]] || return 9; printf '%s\n' "$OWNERS" ;;
    *--query-gpu=memory.free*) [[ "$MEM_ERROR" == 0 ]] || return 9; printf '%s\n' "$FREE" ;;
    *) return 99 ;;
  esac
}
m26_guard_read_argv() {
  case "$1" in
    101|707) M26_OWNER_ARGV=(/venv/bin/python -m service.cotenant_a.cotenant_a) ;;
    202|808) M26_OWNER_ARGV=(/venv/bin/python3.12 -u -m service.cotenant_b.cotenant_b) ;;
    303) M26_OWNER_ARGV=(/tmp/attn_bench_sm89 decode-128k) ;;
    404) M26_OWNER_ARGV=(python bench.py --note service.cotenant_a.cotenant_a) ;;
    *) return 1 ;;
  esac
}
m26_gpu_guard
'''


class GuardTests(unittest.TestCase):
    def invoke(self, script, args=(), **values):
        env = dict(os.environ, OWNERS="", FREE="18000", OWNER_ERROR="0", MEM_ERROR="0")
        env.pop("CUDA_VISIBLE_DEVICES", None)
        env.update(values)
        return subprocess.run(["bash", "-c", 'set -euo pipefail; source "$1"; shift; ' + script,
                               "guard-test", str(GUARD), *args], env=env,
                              text=True, capture_output=True, timeout=5)

    def service(self, *args):
        return self.invoke('m26_guard_service "$@"', args).returncode == 0

    def guard(self, expected, **values):
        result = self.invoke(MOCK, **values)
        self.assertEqual(result.returncode, expected, result.stdout + result.stderr)
        self.assertIn("GPU_GUARD PASS" if expected == 0 else "RESULT: REFUSE",
                      result.stdout + result.stderr)
        return result

    def test_eyes(self):
        self.assertTrue(self.service("/opt/venvs/cotenant_a/bin/python", "-m", "service.cotenant_a.cotenant_a"))

    def test_ears(self):
        self.assertTrue(self.service("/opt/venvs/cotenant_b/bin/python", "-m", "service.cotenant_b.cotenant_b"))

    def test_python_flags(self):
        self.assertTrue(self.service("python3.12", "-u", "-B", "-m", "service.cotenant_a.cotenant_a", "--port", "1234"))

    def test_module_suffix(self):
        self.assertFalse(self.service("python", "-m", "service.cotenant_a.cotenant_a_bench"))

    def test_submodule(self):
        self.assertFalse(self.service("python", "-m", "service.cotenant_b.cotenant_b.extra"))

    def test_c_payload(self):
        self.assertFalse(self.service("python", "-c", "import service.cotenant_a.cotenant_a"))

    def test_script_argument_spoof(self):
        self.assertFalse(self.service("python", "bench.py", "-m", "service.cotenant_a.cotenant_a"))

    def test_not_python(self):
        self.assertFalse(self.service("not-python", "-m", "service.cotenant_a.cotenant_a"))

    def test_option_terminator(self):
        self.assertFalse(self.service("python", "--", "-m", "service.cotenant_a.cotenant_a"))

    def test_missing_module(self):
        self.assertFalse(self.service("python", "-u", "-m"))

    def test_empty_argv(self):
        self.assertFalse(self.service())

    def test_residents(self):
        self.guard(0, OWNERS="101\n202")

    def test_changed_pids(self):
        self.guard(0, OWNERS="707\n808")

    def test_no_owners(self):
        self.guard(0)

    def test_other_benchmark(self):
        self.guard(3, OWNERS="101\n202\n303")

    def test_spoofed_owner(self):
        self.guard(3, OWNERS="404")

    def test_unreadable_owner(self):
        self.guard(3, OWNERS="909")

    def test_invalid_owner_response(self):
        self.guard(3, OWNERS="N/A")

    def test_driver_failure(self):
        self.guard(3, OWNER_ERROR="1")

    def test_memory_query_failure(self):
        self.guard(3, MEM_ERROR="1")

    def test_memory_floor(self):
        self.guard(0, FREE="4096")

    def test_low_memory(self):
        self.guard(3, FREE="4095")

    def test_invalid_memory(self):
        self.guard(3, FREE="N/A")

    def test_oversized_memory_integer(self):
        self.guard(3, FREE="18446744073709551616")

    def test_multi_gpu_response(self):
        self.guard(3, FREE="18000\n18000")

    def test_remapped_device(self):
        self.guard(3, CUDA_VISIBLE_DEVICES="1")

    def test_explicit_device_zero(self):
        self.guard(0, CUDA_VISIBLE_DEVICES="0")


if __name__ == "__main__":
    unittest.main()
