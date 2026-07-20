import gzip
import io
import logging
import shutil
import struct
import subprocess

from PIL import Image

LOG = logging.getLogger("bot2048.phone")

RAW_GZIP_CMD = "screencap | toybox gzip -1"
FMT_RGBA_8888 = 1


class AdbError(RuntimeError):
    pass


def _find_adb(adb="adb"):
    path = shutil.which(adb)
    if not path:
        raise AdbError(
            "adb not found in PATH. Install it: brew install --cask android-platform-tools"
        )
    return path


def list_devices(adb="adb"):
    out = subprocess.run(
        [_find_adb(adb), "devices"], capture_output=True, text=True, timeout=20
    )
    devices = []
    for line in out.stdout.splitlines()[1:]:
        line = line.strip()
        if line and "\t" in line:
            serial, state = line.split("\t", 1)
            devices.append((serial, state.strip()))
    return devices


class Phone:
    def __init__(self, serial=None, adb="adb"):
        self.adb_bin = _find_adb(adb)
        self.serial = serial
        self._fast_capture = None

    @classmethod
    def connect(cls, serial=None, adb="adb"):
        devices = list_devices(adb)
        if not devices:
            raise AdbError(
                "No device found. Check the cable and that USB debugging is enabled "
                "(Settings -> About phone -> tap 'Build number' 7 times -> "
                "Developer options -> USB debugging)."
            )
        ready = [s for s, st in devices if st == "device"]
        pending = [(s, st) for s, st in devices if st != "device"]
        if pending:
            for s, st in pending:
                LOG.warning("Device %s is in state '%s'", s, st)
            if any(st == "unauthorized" for _, st in pending):
                LOG.warning("Confirm the RSA fingerprint on the phone screen and retry.")
        if serial:
            if serial not in ready:
                raise AdbError("Device %s is unavailable. Ready: %s" % (serial, ready))
        else:
            if not ready:
                raise AdbError("No device is ready: %s" % devices)
            if len(ready) > 1:
                raise AdbError("Multiple devices %s — specify --serial" % ready)
            serial = ready[0]
        LOG.info("Connected to %s", serial)
        return cls(serial=serial, adb=adb)

    def _cmd(self, *args):
        base = [self.adb_bin]
        if self.serial:
            base += ["-s", self.serial]
        return base + list(args)

    def _run(self, *args, **kw):
        binary = kw.pop("binary", False)
        timeout = kw.pop("timeout", 30)
        cmd = self._cmd(*args)
        LOG.debug("adb: %s", " ".join(cmd[1:]))
        proc = subprocess.run(cmd, capture_output=True, timeout=timeout)
        if proc.returncode != 0:
            raise AdbError(
                "adb %s -> code %d: %s"
                % (" ".join(args), proc.returncode, proc.stderr.decode("utf-8", "replace").strip())
            )
        return proc.stdout if binary else proc.stdout.decode("utf-8", "replace")

    @staticmethod
    def _decode_raw(data):
        if len(data) < 16:
            raise ValueError("buffer too short: %d bytes" % len(data))
        w, h, fmt = struct.unpack("<III", data[:12])
        if fmt != FMT_RGBA_8888:
            raise ValueError("unexpected pixel format: %d" % fmt)
        header = len(data) - w * h * 4
        if header not in (12, 16):
            raise ValueError("unrecognized header (%d bytes) or padding present" % header)
        return Image.frombuffer("RGBA", (w, h), data[header:], "raw", "RGBA", 0, 1).convert("RGB")

    def screencap(self):
        if self._fast_capture is not False:
            try:
                blob = self._run("exec-out", RAW_GZIP_CMD, binary=True, timeout=60)
                img = self._decode_raw(gzip.decompress(blob))
                if self._fast_capture is None:
                    LOG.info("Fast capture (raw+gzip) is working.")
                    self._fast_capture = True
                return img
            except (AdbError, ValueError, OSError, struct.error) as e:
                if self._fast_capture is None:
                    LOG.warning("Fast capture unavailable (%s) — falling back to PNG.", e)
                    self._fast_capture = False
                else:
                    raise

        raw = self._run("exec-out", "screencap", "-p", binary=True, timeout=60)
        if not raw:
            raise AdbError("screencap returned empty")
        return Image.open(io.BytesIO(raw)).convert("RGB")

    def screen_size(self):
        out = self._run("shell", "wm", "size")
        for token in out.split():
            if "x" in token and token.replace("x", "").isdigit():
                w, h = token.split("x")
                return int(w), int(h)
        raise AdbError("failed to read screen size: %r" % out)

    def swipe(self, x1, y1, x2, y2, ms=90):
        self._run(
            "shell", "input", "swipe",
            str(int(x1)), str(int(y1)), str(int(x2)), str(int(y2)), str(int(ms)),
        )

    def tap(self, x, y):
        self._run("shell", "input", "tap", str(int(x)), str(int(y)))

    def current_focus(self):
        try:
            out = self._run("shell", "dumpsys window")
        except AdbError:
            return ""
        for line in out.splitlines():
            if "mCurrentFocus" in line:
                return line.strip()
        return ""

    def foreground_activity(self):
        try:
            out = self._run("shell", "dumpsys window")
        except AdbError:
            return ""
        for line in out.splitlines():
            if "mFocusedApp" in line and "/" in line:
                for token in line.split():
                    if "/" in token and "." in token:
                        return token.rstrip("}")
        return ""

    def is_locked(self):
        focus = self.current_focus()
        return any(k in focus for k in ("Keyguard", "NotificationShade", "StatusBar"))

    def wake(self):
        try:
            self._run("shell", "settings", "put", "global", "stay_on_while_plugged_in", "15")
            got = self._run("shell", "settings", "get", "global",
                            "stay_on_while_plugged_in").strip()
            if got != "15":
                LOG.warning("stay_on_while_plugged_in = %s (wanted 15) — screen may turn off", got)
        except AdbError as e:
            LOG.warning("Failed to prevent sleep: %s", e)

        self._run("shell", "input", "keyevent", "KEYCODE_WAKEUP")
        if self.is_locked():
            raise AdbError(
                "Screen is locked. Unlock the phone (fingerprint or code) and "
                "retry — adb cannot enter your PIN."
            )
