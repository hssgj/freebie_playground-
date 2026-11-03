"""Run the assembled PAE through real CLI/HTTP processes in a temporary copy.

Uses only echo and a local HTTP fixture. No live LLM, cloud key or phone state.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from tempfile import TemporaryDirectory
import threading
import time
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]


def run_cli(copy: Path, env: dict, *args: str, user_input: str) -> str:
    result = subprocess.run([sys.executable, 'app.py', *args], input=user_input, cwd=copy/'prototype_v0', env=env, text=True, capture_output=True, timeout=15)
    if result.returncode:
        raise AssertionError('CLI assembly failed: ' + result.stderr[-2000:])
    return result.stdout


def api(port: int, path: str, payload=None):
    data = None if payload is None else json.dumps(payload).encode()
    req = Request(f'http://127.0.0.1:{port}'+path, data=data, headers={'Content-Type':'application/json'})
    with urlopen(req, timeout=5) as response:
        return json.loads(response.read())


@contextmanager
def mobile(copy: Path, env: dict):
    ready = copy/'mobile-ready.json'
    ready.unlink(missing_ok=True)
    script = (
        'import json; from pathlib import Path; import mobile_server as m; '
        "server=m.ThreadingHTTPServer(('127.0.0.1',0),m.Handler); "
        "Path('../mobile-ready.json').write_text(json.dumps({'port':server.server_port})); "
        'server.serve_forever()'
    )
    process = subprocess.Popen([sys.executable,'-u','-c',script], cwd=copy/'prototype_v0', env=env, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
    try:
        deadline=time.monotonic()+10
        while not ready.exists():
            if process.poll() is not None:
                raise AssertionError('Mobile startup failed: '+process.stderr.read()[-2000:])
            if time.monotonic()>deadline:
                raise AssertionError('Mobile startup timed out')
            time.sleep(.05)
        yield json.loads(ready.read_text())['port']
    finally:
        process.terminate()
        try: process.wait(timeout=3)
        except subprocess.TimeoutExpired: process.kill();process.wait(timeout=3)
        process.stderr.close()


def verify() -> dict:
    if not __debug__:
        raise RuntimeError('Run verification without Python optimization; assertions must remain enabled')
    checks=[]
    # Do not inherit API credentials, provider selection, sessions or proxies.
    env={k:os.environ[k] for k in ('PATH','LANG','LC_ALL','SYSTEMROOT','TMPDIR') if k in os.environ}
    env.update({'PYTHONIOENCODING':'utf-8','PYTHONDONTWRITEBYTECODE':'1','PAE_PROFILE':'echo'})
    with TemporaryDirectory(prefix='pae-assembly-') as temp:
        copy=Path(temp)/'PAE';(copy/'prototype_v0').mkdir(parents=True)
        for source in (ROOT/'prototype_v0').glob('*.py'):
            if not source.name.startswith(('test_','install_','apply_')):
                shutil.copy2(source,copy/'prototype_v0'/source.name)
        shutil.copytree(ROOT/'prototype_v0/mobile_web',copy/'prototype_v0/mobile_web')
        shutil.copytree(ROOT/'config',copy/'config')
        source=copy/'sample.md';source.write_text('Assembly fixture: archive door is locked.\n')
        listed=json.loads(run_cli(copy,env,'--list-profiles',user_input=''))
        assert {x['profile'] for x in listed}=={'echo','local-qwen','groq','gemini'}
        assert not (copy/'prototype_v0/data').exists()
        checks.append('catalog_lists_without_credentials_or_session')
        output=run_cli(copy,env,'--profile','echo','--session','assembly-cli','--load',str(source),user_input='assembly-ping\n/quit\n')
        assert '[echo] assembly-ping' in output
        session_path=copy/'prototype_v0/data/sessions/assembly-cli.json'
        session=json.loads(session_path.read_text())
        assert session['messages']==[{'role':'user','content':'assembly-ping'},{'role':'assistant','content':'[echo] assembly-ping'}]
        assert session['sources'][0]['content']==source.read_text()
        checks.append('cli_input_provider_agent_source_and_session')
        output=run_cli(copy,env,'--profile','echo','--session','assembly-cli',user_input='/history\n/quit\n')
        assert 'user> assembly-ping' in output
        assert json.loads(session_path.read_text())['messages']==session['messages']
        checks.append('cli_restart_preserves_history_and_source')

        requests=[]
        class Fixture(BaseHTTPRequestHandler):
            def log_message(self,*args): pass
            def do_POST(self):
                payload=json.loads(self.rfile.read(int(self.headers['Content-Length'])))
                requests.append({'payload':payload,'auth':self.headers.get('Authorization'),'agent':self.headers.get('User-Agent')})
                last=next(x['content'] for x in reversed(payload['messages']) if x['role']=='user')
                data=json.dumps({'choices':[{'message':{'content':'[fixture] '+last},'finish_reason':'stop'}],'model':'fixture-server'}).encode()
                self.send_response(200);self.send_header('Content-Type','application/json');self.send_header('Content-Length',str(len(data)));self.end_headers();self.wfile.write(data)
        fixture=ThreadingHTTPServer(('127.0.0.1',0),Fixture)
        thread=threading.Thread(target=fixture.serve_forever,daemon=True);thread.start()
        try:
            profiles=copy/'fixture-profiles.json'
            # context_tokens must be able to host the runtime's own preamble (tool
            # protocol + session facts + history rules), which the context guard
            # counts like any other message. 1024 could not, so the transport check
            # below would be refused before the request. max_output_tokens stays 77,
            # which is what this check actually asserts.
            profiles.write_text(json.dumps({'schema_version':1,'profiles':{'fixture':{'provider':'openai-compatible','url':f'http://127.0.0.1:{fixture.server_port}/v1/chat/completions','model':'fixture-only','context_tokens':8192,'max_output_tokens':77,'disable_thinking':True}}}))
            output=run_cli(copy,env,'--profile','fixture','--profiles-file',str(profiles),'--session','assembly-cli',user_input='transport-ping\n/quit\n')
            assert '[fixture] transport-ping' in output
            assert len(requests)==1 and requests[0]['auth'] is None and requests[0]['agent']=='PAE/0.1'
            assert requests[0]['payload']['model']=='fixture-only' and requests[0]['payload']['max_tokens']==77
            session=json.loads(session_path.read_text())
            assert session['messages'][0]['content']=='assembly-ping' and len(session['messages'])==4
            checks.append('profile_selection_existing_http_provider_same_session')
        finally:
            fixture.shutdown();fixture.server_close();thread.join(timeout=2)

        with mobile(copy,env) as port:
            assert api(port,'/api/health')['status']=='ready'
            info=api(port,'/api/runtime')
            assert info['profile']=='echo'
            assert {'github_read','github_list','github_search','gmail_search','gmail_read'} <= set(info['tools'])
            assert api(port,'/api/chat',{'message':'mobile-ping'})['reply']=='[echo] mobile-ping'
            assert len(api(port,'/api/history')['messages'])==2
        checks.append('mobile_http_to_shared_agent_and_tool_registry')
        with mobile(copy,env) as port:
            history=api(port,'/api/history')['messages']
            assert history==[{'role':'user','content':'mobile-ping'},{'role':'assistant','content':'[echo] mobile-ping'}]
        checks.append('mobile_process_restart_preserves_history')
    inputs=list((ROOT/'prototype_v0').glob('*.py'))+[ROOT/'config/model_profiles.json',Path(__file__).resolve()]
    hashes={p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(inputs)}
    return {'schema':'pae-assembly-proof-v1','checked_utc':datetime.now(timezone.utc).isoformat(),'passed':True,'checks':checks,'source_sha256':hashes,'network_scope':'loopback fixture only','real_llm_inference':False,'phone_installation_verified':False,'credentials_used':False}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,help='Optionally save the evidence JSON')
    args=parser.parse_args()
    result=verify();text=json.dumps(result,ensure_ascii=False,indent=2)+'\n'
    if args.output:
        args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(text)
    print(text,end='')


if __name__=='__main__': main()
