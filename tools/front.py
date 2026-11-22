#!/usr/bin/env python3
"""One address for servers that share the cards by taking turns (YUZURI_PARK=1 on each).

    front.py --backend http://127.0.0.1:8080 --backend http://127.0.0.1:8081 --port 8000

A request names its model; the front reads the names from each server's /v1/models. When the model asked for is on
the server that waits, the front lets the requests of the server that runs finish, parks it (POST /yuzuri/park),
resumes the other (POST /yuzuri/resume) and passes the request through. Requests are served in the order they came:
one for the waiting model is not overtaken by later ones for the running model. Nothing here knows a model: any
number of servers, whatever they serve.

The front can start the servers as well, one model after the other on the same cards, each parked before the next
starts beside it:

    front.py --port 8000 --serve-arg "--profile auto --tp 2" --model /models/a --model "/models/b --context 131072"

A --model is a model directory and, after it, serve.py arguments of its own; --serve-arg goes to every one. A server
is named after its directory unless its arguments name it, listens on the ports after the front's, and stops with
the front. A model with a launcher of its own (a recipe's serve.sh) is given as that launcher, with its environment
before it as in a shell: --model "MODEL=/models/c /recipes/c/serve.sh --background". The launcher passes the words of
YUZURI_SERVE_ARGS on to serve.py and prints what serve.py prints.

    GET /v1/models      the models of every server
    GET /yuzuri/front   which server runs, the switches so far
"""
import argparse
import http.client
import json
import os
import re
import shlex
import signal
import subprocess
import sys
import threading
import time
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

HOP = {"connection", "keep-alive", "proxy-authenticate", "proxy-authorization", "te", "trailers", "transfer-encoding",
       "upgrade", "host", "content-length"}


class Refused(Exception):
    def __init__(self, status, message):
        super().__init__(message)
        self.status, self.message = status, message


def _say(msg):
    print(f"[front] {time.strftime('%H:%M:%S')} {msg}", file=sys.stderr, flush=True)


class Backend:
    def __init__(self, url):
        u = urllib.parse.urlsplit(url)
        if u.scheme != "http" or not u.hostname or not u.port:
            raise SystemExit(f"--backend {url}: http://host:port")
        self.url, self.host, self.port = url.rstrip("/"), u.hostname, u.port
        self.models = []

    def conn(self, timeout=None):
        return http.client.HTTPConnection(self.host, self.port, timeout=timeout)

    def call(self, method, path):
        """(status, parsed JSON or None); OSError when the server is gone."""
        c = self.conn()
        try:
            c.request(method, path, headers={"Content-Length": "0"} if method == "POST" else {})
            r = c.getresponse()
            raw = r.read()
            try:
                return r.status, json.loads(raw)
            except ValueError:
                return r.status, None
        finally:
            c.close()

    def parked(self):
        status, got = self.call("GET", "/yuzuri/state")
        if status != 200 or not isinstance(got, dict) or "parked" not in got:
            raise Refused(502, f"{self.url} has no /yuzuri/state (HTTP {status}): start it with YUZURI_PARK=1")
        return bool(got["parked"])

    def read_models(self):
        status, got = self.call("GET", "/v1/models")
        if status != 200 or not isinstance(got, dict):
            raise Refused(502, f"{self.url}/v1/models: HTTP {status}")
        self.listing = [m for m in got.get("data", []) if isinstance(m, dict) and m.get("id")]
        self.models = [m["id"] for m in self.listing]
        if not self.models:
            raise Refused(502, f"{self.url}/v1/models names no model")


class Front:
    def __init__(self, backends, pause=0.2):
        self.backends = backends
        self.cv = threading.Condition()
        self.queue = []          # (token, backend index), in arrival order
        self.active = None       # index of the server on the cards
        self.inflight = 0
        self.switching = False
        self.switches = 0
        self.last = None
        self.pause = pause       # between two tries of a park that found the server still busy

    def discover(self):
        owner = {}
        for i, b in enumerate(self.backends):
            b.read_models()
            for name in b.models:
                if name in owner:
                    raise Refused(502, f"model name {name!r} is served by {self.backends[owner[name]].url} and {b.url}: "
                                       "give each server its own --served-model-name")
                owner[name] = i
        self.owner = owner
        up = [i for i, b in enumerate(self.backends) if not b.parked()]
        for i in up[1:]:
            self._park(i)
        self.active = up[0] if up else None
        _say("serving " + ", ".join(f"{n} -> {self.backends[i].url}" for n, i in owner.items())
             + (f"; on the cards: {self.backends[self.active].url}" if up else "; all parked"))

    def _park(self, i):
        b = self.backends[i]
        while True:
            status, got = b.call("POST", "/yuzuri/park")
            if status == 200 and isinstance(got, dict) and got.get("success"):
                return
            msg = (got or {}).get("message") if isinstance(got, dict) else None
            if status == 200 and msg and "running or queued" in msg:
                time.sleep(self.pause)   # a request that was cut off is still leaving the scheduler
                continue
            raise Refused(503, f"{b.url} did not park: HTTP {status} {msg or ''}".strip())

    def _resume(self, i):
        b = self.backends[i]
        status, got = b.call("POST", "/yuzuri/resume")
        if not (status == 200 and isinstance(got, dict) and got.get("success")):
            msg = (got or {}).get("message") if isinstance(got, dict) else None
            raise Refused(503, f"{b.url} did not resume: HTTP {status} {msg or ''}".strip())

    def _switch(self, to):
        t0 = time.perf_counter()
        was = self.active
        try:
            if was is not None:
                self._park(was)
                self.active = None
            t1 = time.perf_counter()
            self._resume(to)
            self.active = to
        except OSError as e:
            raise Refused(503, f"switch to {self.backends[to].url}: {e}") from e
        t2 = time.perf_counter()
        self.switches += 1
        self.last = {"to": self.backends[to].url, "park_s": round(t1 - t0, 3), "resume_s": round(t2 - t1, 3)}
        _say(f"switch {self.switches}: " + (f"{self.backends[was].url} parked in {t1 - t0:.2f} s, " if was is not None else "")
             + f"{self.backends[to].url} resumed in {t2 - t1:.2f} s")

    def enter(self, model):
        """Blocks until the request may go to its server; returns the server. leave() after the answer."""
        if model not in self.owner:
            raise Refused(404, f"model {model!r} is not served here: {', '.join(sorted(self.owner))}")
        to = self.owner[model]
        me = object()
        with self.cv:
            self.queue.append((me, to))
            try:
                while True:
                    at = next(k for k, (t, _) in enumerate(self.queue) if t is me)
                    if not self.switching and all(i == to for _, i in self.queue[:at]):
                        if self.active == to:
                            self.inflight += 1
                            return self.backends[to]
                        if self.inflight == 0:
                            self.switching = True
                            self.cv.release()
                            try:
                                self._switch(to)
                            finally:
                                self.cv.acquire()
                                self.switching = False
                                self.cv.notify_all()
                            continue
                    self.cv.wait()
            finally:
                self.queue = [q for q in self.queue if q[0] is not me]
                self.cv.notify_all()

    def leave(self):
        with self.cv:
            self.inflight -= 1
            self.cv.notify_all()

    def state(self):
        with self.cv:
            return {"active": None if self.active is None else self.backends[self.active].url,
                    "backends": [{"url": b.url, "models": b.models} for b in self.backends],
                    "inflight": self.inflight, "waiting": len(self.queue), "switches": self.switches, "last": self.last}


def handler(front):
    class H(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.0"   # an answer ends when the connection closes: a stream passes through as it comes

        def log_message(self, *a):
            pass

        def _json(self, status, obj):
            raw = json.dumps(obj).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        def _refuse(self, e):
            self._json(e.status, {"error": {"message": e.message, "type": "yuzuri_front", "code": e.status}})

        def do_GET(self):
            path = self.path.split("?")[0]
            if path == "/v1/models":
                return self._json(200, {"object": "list", "data": [m for b in front.backends for m in b.listing]})
            if path == "/yuzuri/front":
                return self._json(200, front.state())
            if path in ("/health", "/"):
                return self._json(200, {"status": "ok"})
            self._refuse(Refused(404, f"GET {path}: the front answers /v1/models, /yuzuri/front, /health; "
                                      "the rest is a server's own, at its own port"))

        def do_POST(self):
            body = self.rfile.read(int(self.headers.get("Content-Length") or 0))
            try:
                model = json.loads(body).get("model")
            except (ValueError, AttributeError):
                model = None
            if not isinstance(model, str):
                return self._refuse(Refused(400, "the request names no model: the front chooses the server by it"))
            try:
                b = front.enter(model)
            except Refused as e:
                return self._refuse(e)
            c = b.conn()
            try:
                head = {k: v for k, v in self.headers.items() if k.lower() not in HOP}
                head["Content-Length"] = str(len(body))
                try:
                    c.request("POST", self.path, body, head)
                    r = c.getresponse()
                except OSError as e:
                    return self._refuse(Refused(502, f"{b.url}: {e}"))
                self.send_response(r.status)
                for k, v in r.getheaders():
                    if k.lower() not in HOP - {"content-length"}:
                        self.send_header(k, v)
                self.end_headers()
                try:
                    while True:
                        piece = r.read1(65536)
                        if not piece:
                            break
                        self.wfile.write(piece)
                        self.wfile.flush()
                except OSError:
                    pass   # the client left: closing the connection below tells the server
            finally:
                c.close()
                front.leave()

    return H


def serve(front, host, port):
    class S(ThreadingHTTPServer):
        daemon_threads = True
        request_queue_size = 128

    return S((host, port), handler(front))


def launch(models, common, port, host, log_dir, serve, say=_say):
    """Start a server for each model with serve.py --background, in order: each is parked once it answers, and the
    next starts beside it; the last stays on the cards. Returns ([Backend], [pid]); stops what it started and
    raises Refused when one does not come up."""
    backends, pids, names = [], [], set()
    try:
        for k, spec in enumerate(models):
            words, envs = shlex.split(spec), {}
            while words and re.match(r"^[A-Za-z_][A-Za-z0-9_]*=", words[0]):
                key, _, value = words.pop(0).partition("=")
                envs[key] = value
            if not words:
                raise Refused(400, "--model: a model directory or a launcher of the model's own, then its arguments")
            path, own = words[0], words[1:]
            # a model served through a launcher of its own (a recipe's serve.sh: the model's switches and files):
            # the front's serve.py arguments reach it in YUZURI_SERVE_ARGS, for it to pass on
            launcher = os.path.isfile(path) and os.access(path, os.X_OK)
            if launcher:
                name = next((os.path.basename(os.path.normpath(v)) for v in envs.values() if os.path.isdir(v)),
                            os.path.basename(os.path.dirname(os.path.abspath(path))))
            else:
                name = os.path.basename(os.path.normpath(path))
            if "--served-model-name" in own and not launcher:
                name = own[own.index("--served-model-name") + 1]
            if name in names:
                raise Refused(400, f"two models named {name!r}: give one a --served-model-name of its own")
            names.add(name)
            ours = (["--port", str(port + 1 + k), "--host", host, "--log-dir", os.path.join(log_dir, name),
                     "--env", "YUZURI_PARK=1"]
                    + ([] if "--served-model-name" in own and not launcher else ["--served-model-name", name])
                    + (["--beside"] if k else []) + common)
            env = dict(os.environ, **envs)
            if launcher:
                cmd = [path] + own
                env["YUZURI_SERVE_ARGS"] = shlex.join(ours)
            else:
                cmd = [sys.executable, serve, "--config-dir", path, "--background"] + ours + own
            say(f"starting {name}: {' '.join(shlex.quote(c) for c in (cmd if launcher else cmd[1:]))}"
                + (f" with YUZURI_SERVE_ARGS={env['YUZURI_SERVE_ARGS']}" if launcher else ""))
            t0 = time.perf_counter()
            # its output to a file: the server it leaves behind may keep the launcher's pipes open
            os.makedirs(os.path.join(log_dir, name), exist_ok=True)
            out = os.path.join(log_dir, name, "launch.out")
            with open(out, "w") as fh:
                code = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, env=env).returncode
            with open(out) as fh:
                said = fh.read()
            m = re.search(r"started pid (\d+)", said)
            if code or not m:
                raise Refused(503, f"{name} did not start: {said.strip()[-600:]}")
            pid = int(m.group(1))
            pids.append(pid)
            b = Backend(f"http://{host}:{port + 1 + k}")
            while True:   # until it answers or its process is gone
                try:
                    if b.call("GET", "/health")[0] == 200:
                        break
                except OSError:
                    pass
                try:
                    os.kill(pid, 0)
                except OSError:
                    raise Refused(503, f"{name} stopped before it answered: see {os.path.join(log_dir, name)}")
                time.sleep(1.0)
            say(f"{name} up on {b.url} after {time.perf_counter() - t0:.0f} s")
            backends.append(b)
            if k + 1 < len(models):
                status, got = b.call("POST", "/yuzuri/park")
                if not (status == 200 and isinstance(got, dict) and got.get("success")):
                    raise Refused(503, f"{name} did not park: HTTP {status} {got}")
    except BaseException:
        stop(pids)
        raise
    return backends, pids


def stop(pids):
    for pid in pids:
        try:
            os.kill(pid, signal.SIGTERM)
        except OSError:
            pass


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--backend", action="append", default=[], metavar="URL", help="a server started with YUZURI_PARK=1 (repeat)")
    ap.add_argument("--model", action="append", default=[], metavar="'DIR [serve.py args]'",
                    help="a model the front starts itself (repeat): its directory, then serve.py arguments of its own")
    ap.add_argument("--serve-arg", action="append", default=[], metavar="ARGS", help="serve.py arguments for every --model")
    ap.add_argument("--serve", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "serve.py"))
    ap.add_argument("--log-dir", default=os.path.join(os.getcwd(), "logs", "front"))
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, required=True)
    a = ap.parse_args()
    if not a.backend and not a.model:
        raise SystemExit("front: --backend URL or --model DIR, one or more")
    pids = []
    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()))
    try:
        backends = [Backend(u) for u in a.backend]
        if a.model:
            started, pids = launch(a.model, [w for x in a.serve_arg for w in shlex.split(x)], a.port, a.host,
                                   a.log_dir, a.serve)
            backends += started
        front = Front(backends)
        front.discover()
        s = serve(front, a.host, a.port)
        _say(f"listening on http://{a.host}:{a.port}")
        s.serve_forever()
    except (Refused, OSError) as e:
        stop(pids)
        raise SystemExit(f"front: {e}")
    except KeyboardInterrupt:
        stop(pids)


if __name__ == "__main__":
    main()
