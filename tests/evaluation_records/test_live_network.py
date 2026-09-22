"""Local transport controls; all denied operations stop before network access."""

import importlib.util
import json
import shutil
import socket
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

HERE = Path(__file__).resolve().parents[2] / "examples/integrations/evaluator-live"


def module(name):
    spec = importlib.util.spec_from_file_location(
        "network_test_" + name, HERE / f"{name}.py"
    )
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


NETWORK = module("network")


@pytest.mark.parametrize(
    "event", ["socket.connect", "socket.sendto", "socket.sendmsg", "socket.bind"]
)
def test_socket_rules_allow_only_declared_local_transports(event):
    unix = SimpleNamespace(family=socket.AF_UNIX)
    tcp = SimpleNamespace(
        family=socket.AF_INET, getpeername=lambda: ("127.0.0.1", 1234)
    )
    NETWORK.audit("inspect-ai", event, (unix, "/tmp/local-task"))
    NETWORK.audit("promptfoo", event, (tcp, ("127.0.0.1", 1234)))
    with pytest.raises(RuntimeError, match="local callback"):
        NETWORK.audit("inspect-ai", event, (tcp, ("127.0.0.1", 1234)))
    with pytest.raises(RuntimeError, match="local callback"):
        NETWORK.audit("promptfoo", event, (tcp, ("192.0.2.1", 1234)))
    if event == "socket.sendmsg":
        NETWORK.audit("promptfoo", event, (tcp, None))


@pytest.mark.parametrize(
    "event", ["socket.getaddrinfo", "socket.gethostbyname", "socket.gethostbyaddr"]
)
def test_name_resolution_rejects_external_and_ambiguous_names(event):
    NETWORK.audit("promptfoo", event, ("127.0.0.1",))
    for evaluator, name in [
        ("promptfoo", "localhost"),
        ("promptfoo", "example.invalid"),
        ("garak", "127.0.0.1"),
    ]:
        with pytest.raises(RuntimeError, match="name resolution"):
            NETWORK.audit(evaluator, event, (name,))
    NETWORK.audit("garak", "unrelated.event", ())


def test_configure_sets_optouts_and_installs_real_audit_hook(tmp_path):
    code = """
import importlib.util, json, os, socket, sys
spec=importlib.util.spec_from_file_location('network',sys.argv[1]); m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
m.configure('inspect-ai')
with socket.socket(socket.AF_UNIX) as server:
    server.bind(sys.argv[2]); server.listen(1)
    with socket.socket(socket.AF_UNIX) as client:
        client.connect(sys.argv[2])
        accepted,_=server.accept();accepted.close()
for operation in (lambda: socket.getaddrinfo('example.invalid',80), lambda: socket.socket().connect(('192.0.2.1',80)), lambda: socket.socket(socket.AF_INET,socket.SOCK_DGRAM).sendto(b'x',('192.0.2.1',80))):
    try: operation()
    except RuntimeError: pass
    else: raise AssertionError('outbound operation escaped')
print(json.dumps({k:os.environ[k] for k in m.ENVIRONMENT}))
"""
    # Unix-domain socket paths are short and temporary on macOS.
    import tempfile

    with tempfile.TemporaryDirectory(prefix="nl-", dir="/tmp") as local:
        result = subprocess.run(
            [
                sys.executable,
                "-I",
                "-c",
                code,
                str(HERE / "network.py"),
                str(Path(local) / "s"),
            ],
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        )
    assert json.loads(result.stdout) == NETWORK.ENVIRONMENT


def test_promptfoo_node_guard_rejects_external_operations_and_overrides(tmp_path):
    node = shutil.which("node")
    if not node:
        pytest.skip("requires Node for the local Promptfoo transport guard")
    guard = module("harness").PROMPTFOO_NETWORK_GUARD
    script = tmp_path / "guard.mjs"
    script.write_text(
        guard
        + """
const attempts=[
 ()=>net.connect(443,'192.0.2.1'),
 ()=>http2.connect('http://192.0.2.1'),
 ()=>{const channel=dgram.createSocket('udp4');try{channel.send('x',80,'192.0.2.1');}finally{channel.close();}},
 ()=>http.get('http://127.0.0.1:1234/task',{path:'/other'}),
 ()=>http.get('http://192.0.2.1/task'),
 ()=>http.get('http://127.0.0.1:1234/other'),
 ()=>https.get('https://127.0.0.1:1234/task'),
 ()=>tls.connect({host:'192.0.2.1',port:443}),
 ()=>dns.lookup('example.invalid',()=>{}),
 ()=>dns.promises.resolve('example.invalid'),
 ()=>fetch('https://example.invalid'),
 ()=>fetch('http://127.0.0.1:1235/task'),
 ()=>fetch('http://127.0.0.1:1234/other')];
for(const attempt of attempts){
 let blocked=false;
 try {await attempt();} catch(error) {blocked=error.message.includes('forbids external');}
 if(!blocked) throw Error('outbound operation escaped');
}
console.log(attempts.length);
"""
    )
    result = subprocess.run(
        [node, str(script)],
        env={"LIVE_TASK_URL": "http://127.0.0.1:1234/task"},
        capture_output=True,
        text=True,
        check=True,
        timeout=10,
    )
    assert result.stdout.strip() == "13"


def test_configure_registers_hook_and_overrides_online_settings(monkeypatch):
    environment = dict.fromkeys(NETWORK.ENVIRONMENT, "online")
    hooks = []
    monkeypatch.setattr(NETWORK.os, "environ", environment)
    monkeypatch.setattr(NETWORK.sys, "addaudithook", hooks.append)
    NETWORK.configure("promptfoo")
    assert environment == NETWORK.ENVIRONMENT
    assert len(hooks) == 1
    hooks[0]("socket.getaddrinfo", ("127.0.0.1",))
    with pytest.raises(RuntimeError, match="name resolution"):
        hooks[0]("socket.getaddrinfo", ("example.invalid",))
