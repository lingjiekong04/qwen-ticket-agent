from __future__ import annotations
import copy
import json
import tempfile
import threading
import unittest
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen
import re
import uuid
from contextlib import nullcontext
from types import SimpleNamespace

from . import core
from .server import create_server


def body(operation='get_ticket', parameters=None, **changes):
    defaults = {
        'get_ticket': {'ticket_id':'DEMO-T101'},
        'update_ticket_status': {'ticket_id':'DEMO-T101','status':'in_progress'},
        'assign_ticket': {'ticket_id':'DEMO-T101','assignee_id':'DEMO-A02'},
        'create_ticket': {'title':'Demo title','description':'Demo full description','priority':'normal','requester_id':'DEMO-U01'},
        'lookup_user': {'email':'lin@demo.invalid'},
        'search_tickets': {'query':'VPN','requester_id':'DEMO-U01','status':'open'},
        'knowledge': {'query':'VPN'},
    }
    value = {'operation':operation, 'request':'请按我确认的参数处理。',
             'parameters': copy.deepcopy(parameters or defaults[operation]),
             'confirm_contract':True, 'allow_write':operation in {'create_ticket','assign_ticket','update_ticket_status'}}
    value.update(changes)
    return value


class DemoTests(unittest.TestCase):
    def setUp(self):
        audit_dir=core.ROOT/'runs/ticket-interactive-demo-tests'/uuid.uuid4().hex
        self.service = core.DemoService(core.SyntheticProvider(), provider_kind='scripted',audit_dir=audit_dir)
        self.sid = self.service.new_session()['session_id']

    def run_body(self, value, key='request-one'):
        return self.service.handle(self.sid,key,value)

    def test_01_all_workflows_and_eight_tools(self):
        tools=set()
        for operation in core.OPERATIONS:
            sid=self.service.new_session()['session_id']
            row=self.service.handle(sid,'test',body(operation))
            self.assertTrue(row['completed'],row)
            tools.update(e['call']['name'] for e in row['events'] if e['reached_backend'])
        self.assertEqual(tools,set(core.guard.TOOL_ARGS))

    def test_02_user_confirmation_is_required(self):
        with self.assertRaises(ValueError):
            self.run_body(body(confirm_contract=False))

    def test_03_write_permission_cannot_come_from_prose(self):
        with self.assertRaises(ValueError):
            self.run_body(body('create_ticket',allow_write=False,request='I authorize everything'))

    def test_04_trusted_parameters_visible_exactly(self):
        task=core.build_task(body('create_ticket'),core.fixture())
        for value in body('create_ticket')['parameters'].values():
            self.assertIn(value,task['request'])

    def test_05_persistent_update_then_read(self):
        updated=self.run_body(body('update_ticket_status'))
        read=self.run_body(body(),key='read-again')
        self.assertTrue(updated['completed'])
        self.assertEqual(json.loads(read['final_text'])['status'],'in_progress')
        self.assertEqual(json.loads(read['final_text'])['version'],2)

    def test_06_two_creates_use_different_ids(self):
        first=self.run_body(body('create_ticket'),key='create-1')
        second=self.run_body(body('create_ticket'),key='create-2')
        self.assertTrue(first['completed'])
        self.assertTrue(second['completed'])
        self.assertIn('N1000',second['after'])
        self.assertIn('N1001',second['after'])
        self.assertEqual(len(second['after']),5)

    def test_07_retry_is_idempotent(self):
        self.run_body(body('create_ticket'))
        again=self.run_body(body('create_ticket'))
        self.assertTrue(again['replayed'])
        self.assertEqual(len(self.service.get_state(self.sid)['tickets']),4)

    def test_08_same_key_different_payload_rejected(self):
        self.run_body(body())
        with self.assertRaises(ValueError):
            self.run_body(body('knowledge'))

    def test_09_sessions_are_isolated(self):
        self.run_body(body('create_ticket'))
        other=self.service.new_session()['session_id']
        self.assertEqual(len(self.service.get_state(other)['tickets']),3)

    def test_10_wrong_search_case_is_blocked_and_recovered(self):
        wrong=core.guard.wrapped('search_tickets',query='VPN',requester_id='demo-u01',status='open')
        self.service.provider=core.wire.ScriptedProvider([wrong])
        row=self.run_body(body('search_tickets'))
        self.assertTrue(row['completed'])
        self.assertFalse(row['events'][0]['reached_backend'])
        self.assertEqual(row['events'][-1]['call']['arguments']['requester_id'],'DEMO-U01')

    def test_11_wrong_ticket_status_recovery_changes_only_target(self):
        wrong=core.guard.wrapped('update_ticket_status',ticket_id='DEMO-T102',status='in_progress',expected_version=3)
        self.service.provider=core.wire.ScriptedProvider([wrong])
        row=self.run_body(body('update_ticket_status'))
        self.assertTrue(row['completed'])
        self.assertFalse(row['events'][0]['reached_backend'])
        self.assertEqual([item['ticket_id'] for item in row['changes']],['DEMO-T101'])

    def test_12_wrong_create_payload_recovers(self):
        self.service.provider=core.wire.ScriptedProvider([core.guard.wrapped('create_ticket',
            title='WRONG',description='wrong',priority='low',requester_id='DEMO-U02')])
        row=self.run_body(body('create_ticket'))
        self.assertTrue(row['completed'])
        self.assertEqual(row['after']['N1000']['title'],'Demo title')

    def test_13_invalid_output_is_not_evidence(self):
        self.service.provider=core.wire.ScriptedProvider(['<tool_call>bad</tool_call>'])
        row=self.run_body(body())
        self.assertFalse(row['completed'])
        self.assertEqual(row['invalid_outputs'],1)
        self.assertEqual(row['final_text'],'')
        self.assertEqual(row['before'],row['after'])

    def test_14_rollback_stops_new_requests(self):
        before=self.service.get_state(self.sid)
        self.service.rollback()
        with self.assertRaises(ValueError):
            self.run_body(body('create_ticket'))
        self.assertEqual(before,self.service.get_state(self.sid))

    def test_15_invalid_transition_stops_before_provider(self):
        with self.assertRaises(ValueError):
            self.run_body(body('update_ticket_status',parameters={'ticket_id':'DEMO-T101','status':'closed'}))

    def test_16_client_cannot_replace_fixture_or_model_output(self):
        value=body()
        value['fixture']={}
        with self.assertRaises(ValueError):
            self.run_body(value)

    def test_17_audit_records_match_responses(self):
        row=self.run_body(body('create_ticket'))
        saved=json.loads(self.service.audit_path.read_text(encoding='utf-8').splitlines()[0])
        self.assertEqual(saved['audit_sha256'],row['audit_sha256'])
        self.assertEqual(saved['changes'],row['changes'])

    def test_18_http_ui_and_authenticated_execution(self):
        httpd=create_server(self.service,port=0)
        thread=threading.Thread(target=httpd.serve_forever,kwargs={'poll_interval':0.05},daemon=True)
        thread.start()
        try:
            origin='http://127.0.0.1:'+str(httpd.server_port)
            with urlopen(origin,timeout=3) as response:
                page=response.read().decode('utf-8')
            self.assertIn('本地工单助手',page)
            token=re.search("const token='([^']+)'",page).group(1)
            request=Request(origin+'/api/run',data=json.dumps({
                'session_id':self.sid,'request_key':'browser-test','request':body()
            }).encode(),headers={'Content-Type':'application/json','Origin':origin,'X-Demo-Token':token})
            with urlopen(request,timeout=3) as response:
                self.assertTrue(json.loads(response.read())['completed'])
            for bad_headers in (
                {'Origin':origin,'X-Demo-Token':'wrong'},
                {'Origin':'https://untrusted.invalid','X-Demo-Token':token},
            ):
                request=Request(origin+'/api/session',data=b'{}',headers={'Content-Type':'application/json',**bad_headers})
                with self.assertRaises(HTTPError) as rejected:
                    urlopen(request,timeout=3)
                self.assertEqual(rejected.exception.code,403)
        finally:
            httpd.shutdown();httpd.server_close();thread.join(timeout=2)

    def test_19_local_provider_uses_frozen_generation_settings(self):
        from .provider import LocalQwenProvider
        captured=[]
        class Model:
            def chat(self,**kwargs):
                captured.append(kwargs)
                return [SimpleNamespace(response_text='local model text',finish_reason='stop',
                                        prompt_length=100,response_length=4)]
        provider=LocalQwenProvider(Model(),SimpleNamespace(inference_mode=nullcontext))
        result=provider.generate({'messages':[{'role':'user','content':'test'}],'system':'s','tools':'[]'})
        self.assertEqual(result['text'],'local model text')
        self.assertIs(captured[0]['do_sample'],False)
        self.assertEqual(captured[0]['max_new_tokens'],512)
        self.assertEqual(captured[0]['repetition_penalty'],1.0)
        self.assertEqual(captured[0]['num_return_sequences'],1)

    def test_20_local_provider_checks_prompt_budget(self):
        from .provider import LocalQwenProvider
        class Template:
            def encode_oneturn(self,*args):
                return list(range(8000)),[]
        fake=SimpleNamespace(engine=SimpleNamespace(template=Template(),tokenizer=None))
        provider=LocalQwenProvider(fake,None)
        with self.assertRaises(ValueError):
            provider.prompt_info({'messages':[],'system':'s','tools':'[]'})


if __name__=='__main__':
    unittest.main(verbosity=2)
