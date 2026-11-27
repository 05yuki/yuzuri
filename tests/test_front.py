import importlib.util, json, os, threading, time, unittest, urllib.error, urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

_p = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tools", "front.py")
_s = importlib.util.spec_from_file_location("front", _p)
front = importlib.util.module_from_spec(_s); _s.loader.exec_module(front)
front._say = lambda msg: None

EVENTS = []          # (server name, what) in the order it happened, over all fake servers
ON_CARDS = set()     # the fake servers that are not parked: never two


class Fake:
    """A server with /v1/models, /yuzuri/{state,park,resume} and a completion that takes `slow` seconds."""

    def __init__(self, name, parked, slow=0.0, busy_parks=0):
        self.name, self.parked, self.slow, self.busy_parks, self.running = name, parked, slow, busy_parks, 0
        if not parked:
            ON_CARDS.add(name)
        me = self

        class H(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def _json(self, obj, status=200):
                raw = json.dumps(obj).encode()
                self.send_response(status); self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(raw))); self.end_headers(); self.wfile.write(raw)

            def do_GET(self):
                if self.path == "/v1/models":
                    return self._json({"object": "list", "data": [{"id": me.name, "object": "model"}]})
                if self.path == "/yuzuri/state":
                    return self._json({"parked": me.parked})
                self._json({}, 404)

            def do_POST(self):
                body = self.rfile.read(int(self.headers.get("Content-Length") or 0))
                if self.path == "/yuzuri/park":
                    if me.busy_parks:
                        me.busy_parks -= 1
                        return self._json({"success": False, "message": "requests are running or queued: a server parks when it is idle"})
                    assert me.running == 0, "parked under a running request"
                    me.parked = True; ON_CARDS.discard(me.name); EVENTS.append((me.name, "park"))
                    return self._json({"success": True, "message": ""})
                if self.path == "/yuzuri/resume":
                    assert not ON_CARDS, f"resumed beside {ON_CARDS}"
                    me.parked = False; ON_CARDS.add(me.name); EVENTS.append((me.name, "resume"))
                    return self._json({"success": True, "message": ""})
                if me.parked:
                    return self._json({"error": {"message": "parked"}}, 503)
                me.running += 1
                tag = json.loads(body).get("tag")
                EVENTS.append((me.name, f"start {tag}"))
                if json.loads(body).get("stream"):
                    self.send_response(200); self.send_header("Content-Type", "text/event-stream"); self.end_headers()
                    for k in range(3):
                        self.wfile.write(f"data: {me.name} {k}\n\n".encode()); self.wfile.flush(); time.sleep(me.slow / 3)
                else:
                    time.sleep(me.slow)
                    self._json({"served_by": me.name, "tag": tag})
                me.running -= 1

        self.srv = ThreadingHTTPServer(("127.0.0.1", 0), H)
        self.srv.daemon_threads = True
        self.url = f"http://127.0.0.1:{self.srv.server_address[1]}"
        threading.Thread(target=self.srv.serve_forever, daemon=True).start()


def post(url, obj):
    r = urllib.request.Request(url + "/v1/chat/completions", json.dumps(obj).encode(), {"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(r, timeout=30) as f:
            return f.status, f.read()
    except urllib.error.HTTPError as e:
        with e:
            return e.code, e.read()


class FrontTest(unittest.TestCase):
    def up(self, *fakes):
        f = front.Front([front.Backend(x.url) for x in fakes], pause=0.01)
        f.discover()
        s = front.serve(f, "127.0.0.1", 0)
        threading.Thread(target=s.serve_forever, daemon=True).start()
        self.addCleanup(s.server_close)
        self.addCleanup(s.shutdown)
        return f, f"http://127.0.0.1:{s.server_address[1]}"

    def setUp(self):
        EVENTS.clear(); ON_CARDS.clear()

    def test_switches_by_the_model_named(self):
        a, b = Fake("a", parked=False), Fake("b", parked=True)
        f, url = self.up(a, b)
        self.assertEqual(json.loads(post(url, {"model": "a", "tag": 1})[1])["served_by"], "a")
        self.assertEqual(json.loads(post(url, {"model": "b", "tag": 2})[1])["served_by"], "b")
        self.assertEqual(json.loads(post(url, {"model": "b", "tag": 3})[1])["served_by"], "b")
        self.assertEqual(json.loads(post(url, {"model": "a", "tag": 4})[1])["served_by"], "a")
        self.assertEqual(EVENTS, [("a", "start 1"), ("a", "park"), ("b", "resume"), ("b", "start 2"), ("b", "start 3"),
                                  ("b", "park"), ("a", "resume"), ("a", "start 4")])
        self.assertEqual(f.state()["switches"], 2)

    def test_models_of_every_server_and_refusals(self):
        f, url = self.up(Fake("a", parked=False), Fake("b", parked=True))
        with urllib.request.urlopen(url + "/v1/models") as r:
            self.assertEqual([m["id"] for m in json.load(r)["data"]], ["a", "b"])
        self.assertEqual(post(url, {"model": "c"})[0], 404)
        self.assertEqual(post(url, {"messages": []})[0], 400)
        self.assertEqual(EVENTS, [])

    def test_two_on_the_cards_at_start_one_is_parked(self):
        a, b = Fake("a", parked=False), Fake("b", parked=False)
        ON_CARDS.clear(); ON_CARDS.add("a")   # the fake's own check is for resumes
        f, url = self.up(a, b)
        self.assertEqual(f.state()["active"], a.url)
        self.assertTrue(b.parked)

    def test_all_parked_at_start(self):
        a, b = Fake("a", parked=True), Fake("b", parked=True)
        f, url = self.up(a, b)
        self.assertEqual(json.loads(post(url, {"model": "b", "tag": 1})[1])["served_by"], "b")
        self.assertEqual(EVENTS, [("b", "resume"), ("b", "start 1")])

    def test_waits_for_the_running_requests_and_keeps_the_order(self):
        a, b = Fake("a", parked=False, slow=0.4), Fake("b", parked=True, slow=0.1)
        f, url = self.up(a, b)
        got = {}

        def go(tag, model, delay):
            time.sleep(delay); got[tag] = json.loads(post(url, {"model": model, "tag": tag})[1])["served_by"]

        ts = [threading.Thread(target=go, args=x) for x in ((1, "a", 0), (2, "b", 0.1), (3, "a", 0.2))]
        [t.start() for t in ts]; [t.join() for t in ts]
        self.assertEqual(got, {1: "a", 2: "b", 3: "a"})
        # 3 came while a was still running, but after 2: it does not overtake
        self.assertEqual(EVENTS, [("a", "start 1"), ("a", "park"), ("b", "resume"), ("b", "start 2"),
                                  ("b", "park"), ("a", "resume"), ("a", "start 3")])

    def test_requests_for_the_running_model_run_together(self):
        a, b = Fake("a", parked=False, slow=0.3), Fake("b", parked=True)
        f, url = self.up(a, b)
        ts = [threading.Thread(target=post, args=(url, {"model": "a", "tag": k})) for k in range(3)]
        t0 = time.time(); [t.start() for t in ts]; [t.join() for t in ts]
        self.assertLess(time.time() - t0, 0.7)
        self.assertEqual(f.state()["switches"], 0)

    def test_a_park_that_finds_the_server_busy_is_tried_again(self):
        a, b = Fake("a", parked=False, busy_parks=3), Fake("b", parked=True)
        f, url = self.up(a, b)
        self.assertEqual(json.loads(post(url, {"model": "b", "tag": 1})[1])["served_by"], "b")
        self.assertEqual(a.busy_parks, 0)

    def test_a_stream_passes_through(self):
        a = Fake("a", parked=False, slow=0.3)
        f, url = self.up(a, Fake("b", parked=True))
        r = urllib.request.Request(url + "/v1/chat/completions", json.dumps({"model": "a", "stream": True}).encode())
        with urllib.request.urlopen(r, timeout=30) as resp:
            t0 = time.time(); first = resp.readline(); t1 = time.time() - t0; rest = resp.read()
        self.assertEqual(first, b"data: a 0\n")
        self.assertIn(b"data: a 2", rest)
        self.assertEqual(f.state()["inflight"], 0)

    def test_starts_the_servers_one_after_the_other(self):
        import sys, tempfile, textwrap
        d = tempfile.mkdtemp()
        fake = os.path.join(d, "serve.py")   # stands for tools/serve.py --background: a server of its own process
        with open(fake, "w") as fh:
            fh.write(textwrap.dedent('''
                import argparse, json, os, subprocess, sys
                ap = argparse.ArgumentParser()
                for k in ("--config-dir", "--port", "--host", "--log-dir", "--served-model-name", "--env", "--tp"):
                    ap.add_argument(k)
                ap.add_argument("--background", action="store_true"); ap.add_argument("--beside", action="store_true")
                ap.add_argument("--child", action="store_true")
                a = ap.parse_args()
                if not a.child:
                    with open(os.path.join(os.path.dirname(__file__), "calls"), "a") as fh:
                        fh.write(json.dumps([a.served_model_name, a.beside, a.env, a.tp]) + "\\n")
                    p = subprocess.Popen([sys.executable, __file__, "--child"] + sys.argv[1:], stdout=subprocess.DEVNULL,
                                         stderr=subprocess.DEVNULL, stdin=subprocess.DEVNULL, start_new_session=True)
                    print(f"started pid {p.pid}"); sys.exit(0)
                from http.server import BaseHTTPRequestHandler, HTTPServer
                st = {"parked": False}
                class H(BaseHTTPRequestHandler):
                    def log_message(self, *x): pass
                    def out(self, o):
                        raw = json.dumps(o).encode(); self.send_response(200)
                        self.send_header("Content-Length", str(len(raw))); self.end_headers(); self.wfile.write(raw)
                    def do_GET(self):
                        self.out({"data": [{"id": a.served_model_name}]} if self.path == "/v1/models" else st)
                    def do_POST(self):
                        self.rfile.read(int(self.headers.get("Content-Length") or 0))
                        if self.path.startswith("/yuzuri/"):
                            st["parked"] = self.path.endswith("park"); return self.out({"success": True})
                        self.out({"served_by": a.served_model_name})
                HTTPServer((a.host, int(a.port)), H).serve_forever()
            '''))
        import socket
        with socket.socket() as so:
            so.bind(("127.0.0.1", 0)); port = so.getsockname()[1]
        backends, pids = front.launch([os.path.join(d, "alpha"), os.path.join(d, "beta") + " --tp 1"], [], port,
                                      "127.0.0.1", d, fake, say=lambda m: None)
        self.addCleanup(front.stop, pids)
        with open(os.path.join(d, "calls")) as fh:
            calls = [json.loads(x) for x in fh]
        self.assertEqual(calls, [["alpha", False, "YUZURI_PARK=1", None], ["beta", True, "YUZURI_PARK=1", "1"]])
        f = front.Front(backends, pause=0.01); f.discover()
        self.assertEqual(f.state()["active"], backends[1].url)   # the first was parked before the second started
        self.assertTrue(backends[0].parked())
        self.assertEqual(sorted(f.owner), ["alpha", "beta"])

    def test_launch_keeps_backends_on_loopback_when_front_listens_on_all_interfaces(self):
        import socket, tempfile, textwrap
        d = tempfile.mkdtemp()
        fake = os.path.join(d, "serve.py")
        with open(fake, "w") as fh:
            fh.write(textwrap.dedent('''
                import argparse, json, os, subprocess, sys
                ap = argparse.ArgumentParser()
                for k in ("--config-dir", "--port", "--host", "--log-dir", "--served-model-name", "--env"):
                    ap.add_argument(k)
                ap.add_argument("--background", action="store_true"); ap.add_argument("--child", action="store_true")
                a = ap.parse_args()
                if not a.child:
                    with open(os.path.join(os.path.dirname(__file__), "host"), "w") as fh:
                        fh.write(a.host)
                    p = subprocess.Popen([sys.executable, __file__, "--child"] + sys.argv[1:], stdout=subprocess.DEVNULL,
                                         stderr=subprocess.DEVNULL, stdin=subprocess.DEVNULL, start_new_session=True)
                    print(f"started pid {p.pid}"); sys.exit(0)
                from http.server import BaseHTTPRequestHandler, HTTPServer
                class H(BaseHTTPRequestHandler):
                    def log_message(self, *x): pass
                    def do_GET(self):
                        raw = json.dumps({"status": "ok"}).encode(); self.send_response(200)
                        self.send_header("Content-Length", str(len(raw))); self.end_headers(); self.wfile.write(raw)
                HTTPServer((a.host, int(a.port)), H).serve_forever()
            '''))
        with socket.socket() as so:
            so.bind(("127.0.0.1", 0)); port = so.getsockname()[1]
        backends, pids = front.launch([os.path.join(d, "alpha")], [], port, "0.0.0.0", d, fake, say=lambda m: None)
        self.addCleanup(front.stop, pids)
        with open(os.path.join(d, "host")) as fh:
            self.assertEqual(fh.read(), "127.0.0.1")
        self.assertEqual(backends[0].host, "127.0.0.1")

    def test_park_timeout_refuses_when_server_stays_busy(self):
        a, b = Fake("a", parked=False, busy_parks=100), Fake("b", parked=True)
        f = front.Front([front.Backend(a.url), front.Backend(b.url)], pause=0.01, park_timeout=0.05)
        f.discover()
        with self.assertRaises(front.Refused) as ctx:
            f.enter("b")
        self.assertEqual(ctx.exception.status, 503)

    def test_chunked_post_body_is_forwarded(self):
        import http.client, urllib.parse
        a = Fake("a", parked=False)
        f, url = self.up(a, Fake("b", parked=True))
        u = urllib.parse.urlsplit(url)
        payload = json.dumps({"model": "a", "tag": 9}).encode()
        body = f"{len(payload):x}\r\n".encode() + payload + b"\r\n0\r\n\r\n"
        c = http.client.HTTPConnection(u.hostname, u.port, timeout=10)
        try:
            c.putrequest("POST", "/v1/chat/completions")
            c.putheader("Content-Type", "application/json")
            c.putheader("Transfer-Encoding", "chunked")
            c.endheaders()
            c.send(body)
            r = c.getresponse()
            self.assertEqual(r.status, 200)
            self.assertEqual(json.loads(r.read()), {"served_by": "a", "tag": 9})
        finally:
            c.close()

    def test_same_name_on_two_servers_is_refused(self):
        with self.assertRaises(front.Refused):
            front.Front([front.Backend(Fake("a", parked=False).url), front.Backend(Fake("a", parked=True).url)]).discover()


if __name__ == "__main__":
    unittest.main()
