import contextlib
import socket
import threading
import time
from collections.abc import Iterator

import uvicorn
from fastapi import FastAPI


@contextlib.contextmanager
def serving(app: FastAPI) -> Iterator[str]:
    """Runs the app with uvicorn on a free local port in a background thread.

    Yields the server's base URL. The app's lifespan runs as it would in production, so the
    engine starts and stops with the server.
    """
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    server = uvicorn.Server(uvicorn.Config(app, log_level="warning"))
    thread = threading.Thread(target=server.run, kwargs={"sockets": [sock]}, daemon=True)
    thread.start()
    deadline = time.monotonic() + 10
    while not server.started:
        if not thread.is_alive() or time.monotonic() > deadline:
            raise RuntimeError("the server did not start")
        time.sleep(0.01)
    try:
        yield f"http://127.0.0.1:{port}"
    finally:
        server.should_exit = True
        thread.join(timeout=10)
        sock.close()
