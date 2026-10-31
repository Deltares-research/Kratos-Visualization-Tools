"""Local, authenticated handoff to the running plotter's GUI thread."""

from __future__ import annotations

import hashlib
import json
import os
import queue
import secrets
import sys
import tempfile
import threading
import time
from multiprocessing.connection import Client, Listener
from multiprocessing import AuthenticationError
from pathlib import Path


def _endpoint():
    user = hashlib.sha256(str(Path.home()).encode()).hexdigest()[:16]
    directory = Path(tempfile.gettempdir()) / f"kratos-plotter-{user}"
    directory.mkdir(mode=0o700, exist_ok=True)
    key_path = directory / "auth.key"
    try:
        descriptor = os.open(key_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError:
        pass
    else:
        with os.fdopen(descriptor, "wb") as output:
            output.write(secrets.token_bytes(32))
    for _ in range(20):
        key = key_path.read_bytes()
        if len(key) == 32:
            break
        time.sleep(0.05)
    else:
        raise RuntimeError(f"Invalid plotter authentication file: {key_path}")
    if os.name == "nt":
        return rf"\\.\pipe\kratos-plotter-{user}", "AF_PIPE", key
    return str(directory / "session.sock"), "AF_UNIX", key


def _forward(address, family, key, config, output_dir):
    try:
        connection = Client(address, family=family, authkey=key)
    except (FileNotFoundError, ConnectionRefusedError):
        return False
    except AuthenticationError as error:
        raise RuntimeError("Could not authenticate with the open plotter.") from error
    with connection:
        try:
            connection.send_bytes(json.dumps({
                "config": str(config.resolve()),
                "output_dir": str(output_dir.resolve()) if output_dir else None,
            }).encode())
            if not connection.poll(120):
                raise RuntimeError("The open plotter did not respond within 120 seconds.")
            response = json.loads(connection.recv_bytes(65536))
        except EOFError as error:
            raise RuntimeError("The plotter closed before importing the profile.") from error
        if response.get("error"):
            raise ValueError(response["error"])
    return True


def open_session(config: Path, output_dir: Path | None = None):
    """Return a new session, or forward to the existing window and return None."""
    address, family, key = _endpoint()
    if family == "AF_UNIX":
        import fcntl

        # Serialize startup so stale-socket cleanup cannot unlink a new server.
        with Path(address).with_suffix(".lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            return _open_session(address, family, key, config, output_dir)
    return _open_session(address, family, key, config, output_dir)


def _open_session(address, family, key, config, output_dir):
    if _forward(address, family, key, config, output_dir):
        return None
    if family == "AF_UNIX":
        Path(address).unlink(missing_ok=True)
    try:
        listener = Listener(address, family=family, authkey=key)
    except OSError:
        if _forward(address, family, key, config, output_dir):
            return None
        raise
    return PlotterSession(listener, address, family, key)


class PlotterSession:
    def __init__(self, listener, address, family, key):
        self.listener = listener
        self.address, self.family, self.key = address, family, key
        self.requests = queue.Queue()
        self.stopped = threading.Event()
        self.thread = None
        self.timer = None

    def attach(self, viewer):
        self.viewer = viewer
        self.timer = viewer.figure.canvas.new_timer(interval=100)
        self.timer.add_callback(self.process_requests)
        self.timer.start()
        self.thread = threading.Thread(target=self._serve, daemon=True)
        self.thread.start()

    def _serve(self):
        while True:
            try:
                connection = self.listener.accept()
                with connection:
                    if self.stopped.is_set():
                        return
                    if not connection.poll(5):
                        raise TimeoutError("No plotter request received.")
                    request = json.loads(connection.recv_bytes(65536))
                    done, response = threading.Event(), {}
                    self.requests.put((request, done, response))
                    deadline = time.monotonic() + 120
                    while not done.wait(0.1) and not self.stopped.is_set():
                        if time.monotonic() >= deadline:
                            break
                    if not done.is_set():
                        response["error"] = "The plotter GUI did not process the profile."
                        done.set()
                    connection.send_bytes(json.dumps(response).encode())
            except (OSError, EOFError, ValueError, AuthenticationError) as error:
                if not self.stopped.is_set():
                    print(f"Plotter handoff error: {error}", file=sys.stderr)

    def process_requests(self):
        while True:
            try:
                request, done, response = self.requests.get_nowait()
            except queue.Empty:
                return
            if done.is_set():
                continue
            try:
                config = Path(request["config"])
                output = Path(request["output_dir"]) if request.get("output_dir") else None
                self.viewer.add_configuration(config, output)
            except (OSError, KeyError, TypeError, ValueError, RuntimeError) as error:
                response["error"] = str(error)
            finally:
                done.set()

    def close(self):
        if self.stopped.is_set():
            return
        if self.timer:
            self.timer.stop()
        self.stopped.set()
        while not self.requests.empty():
            _, done, response = self.requests.get_nowait()
            response["error"] = "The plotter window was closed."
            done.set()
        if self.thread:
            with Client(self.address, family=self.family, authkey=self.key):
                pass
            self.thread.join(timeout=5)
        self.listener.close()
