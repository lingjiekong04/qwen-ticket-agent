"""Independent public-history replay, provenance, completeness and claim checks."""
import copy
from collections import Counter
import json
import re
from . import core as c

CHAINS = {
    1: ['search_kb','get_kb_article'], 2: ['search_kb','get_kb_article','get_kb_article'],
    3: ['search_kb'], 4: ['search_kb'], 5: ['search_kb','get_kb_article','get_kb_article'],
    6: [], 7: [], 8: [], 9: ['lookup_user','create_ticket'], 10: [], 11: [], 12: [], 13: [],
    14: ['lookup_user','create_ticket'], 15: ['lookup_user','create_ticket'],
    16: ['lookup_user','create_ticket','assign_ticket'],
    17: ['get_ticket','update_ticket_status','update_ticket_status','update_ticket_status'],
    18: ['get_ticket','assign_ticket']*3, 19: ['get_ticket','get_ticket'], 20: ['lookup_user']}
ERRORS = {5:['article_not_found'], 18:['version_conflict']*3, 20:['user_not_found']}


def normalized(text):
    text = re.sub(r'(?:U|T|KB)[A-Z0-9]{6,}|[A-Za-z0-9._+-]+@example\.test', ' identity ', text)
    return ''.join(ch.lower() for ch in text if ch.isalnum())


def forbidden():
    # These read only frozen DEV, development diagnostics and previous training.
    # No old final-test path is opened or enumerated.
    from extensions.ticket_policy_repair_build_v1.validate import forbidden as old_forbidden
    blocked = old_forbidden()
    path = c.ROOT / 'data/ticket_policy_repair_candidate_v1/episodes.json'
    old = c.read(path)
    for e in old:
        blocked['requests'].add(e['row']['conversations'][0]['value'])
        for table in ('users','tickets','articles'):
            for row in e['initial_state'][table]:
                for key in ('user_id','ticket_id','article_id','email'):
                    if key in row:
                        blocked['ids'].add(row[key])
    blocked['previous_repair_episodes_sha256'] = c.sha(path)
    blocked['normalized_requests'] = {normalized(x) for x in blocked['requests']}
    return blocked


def observe(name, response, evidence):
    if not response['ok']:
        return
    rows = response['result'] if isinstance(response['result'],list) else [response['result']]
    for row in rows:
        if not isinstance(row,dict):
            continue
        if name == 'lookup_user':
            evidence['users'][row['user_id']] = copy.deepcopy(row)
        if 'ticket_id' in row:
            evidence['tickets'][row['ticket_id']] = copy.deepcopy(row)
        if 'article_id' in row:
            evidence['article_ids'].add(row['article_id'])
            if name == 'get_kb_article':
                evidence['bodies'][row['article_id']] = copy.deepcopy(row)


def claim_check(claim, e, evidence, human, history):
    kind, n = claim['kind'], e['spec']['index']
    if kind == 'denied':
        c.require(not e['spec']['allow_write'] and not history and n in (6,7,8), 'Incorrect or late refusal')
    elif kind == 'missing':
        field = claim['field']
        c.require(e['spec']['allow_write'] and not history and n in (14,15)
                  and field == ('email' if n==14 else 'title') and e['spec']['fields'][field] not in human, 'Incorrect clarification')
    elif kind == 'article':
        a = claim['article']
        c.require(evidence['bodies'].get(a['article_id']) == a, 'Unsupported article body')
    elif kind == 'titles':
        c.require(n==3 and len(history)==1 and claim['articles']==history[0]['response']['result'], 'Unsupported titles claim')
    elif kind == 'empty':
        c.require(n==4 and len(history)==1 and history[0]['response']['ok'] and history[0]['response']['result']==[], 'Missing empty-search evidence')
        c.require(claim['query']==history[0]['call']['arguments']['query'], 'Empty query changed')
    elif kind == 'article_missing':
        c.require(claim['article_id'] in evidence['article_ids'] and claim['article_id'] not in evidence['bodies'], 'Missing body claim inconsistent')
        c.require(any(x['call']=={'name':'get_kb_article','arguments':{'article_id':claim['article_id']}}
                      and x['response'].get('error',{}).get('code')=='article_not_found' for x in history), 'No article failure evidence')
    elif kind in ('ticket','conflict'):
        t = claim['ticket']
        c.require(evidence['tickets'].get(t['ticket_id'])==t, 'Unsupported ticket state')
        if kind == 'conflict':
            c.require(n==18 and [x['response'].get('error',{}).get('code') for x in history if x['call']['name']=='assign_ticket']==['version_conflict']*3, 'Retry exhaustion not evidenced')
        else:
            c.require(n in (9,14,15,16,17,19), 'Unexpected ticket success')
    elif kind == 'user_missing':
        c.require(n==20 and len(history)==1 and history[0]['call']=={'name':'lookup_user','arguments':{'email':claim['email']}}
                  and history[0]['response'].get('error',{}).get('code')=='user_not_found', 'Missing requester error evidence')
    else:
        c.require(kind=='priority' and not history and claim['variant']==e['spec']['variant']
                  and claim['order']==['high','normal','low'] and claim['sla'] is None, 'Invalid priority claim')


def validate_episode(e, blocked=None):
    spec, row, locale = e['spec'], e['row'], e['locale']
    c.require(locale in ('zh','en') and spec==c.specs()[spec['index']-1] and e['id']==spec['pair_id']+'-'+locale, 'Episode/spec mismatch')
    c.require(set(row)=={'system','tools','conversations'} and row['system']==c.system(spec,locale)
              and json.loads(row['tools'])==c.TOOLS, 'Policy/schema changed or hidden metadata inserted')
    messages, n = row['conversations'], spec['index']
    c.require(messages[0]=={'from':'human','value':c.request(spec,locale)} and messages[-1]['from']=='gpt', 'Request/final changed')
    c.require(len(messages)%2==0 and len({x['index'] for x in e['claims']})==len(e['claims']), 'Invalid roles/claim indices')
    if blocked is not None:
        public = c.canonical(row)
        ids = set(re.findall(r'(?:U|T|KB)[A-Z0-9]{6,}|[A-Za-z0-9._+-]+@example\.test',public))
        c.require(not ids & blocked['ids'], 'Previous identity copied')
        c.require(normalized(messages[0]['value']) not in blocked['normalized_requests'], 'Previous request copied even after identity replacement')
    evidence = {'users':{},'tickets':{},'article_ids':set(),'bodies':{}}
    human, history, cursor = '', [], 0
    claims = {x['index']:x['claims'] for x in e['claims']}
    with c.Environment(spec) as env:
        c.require(env.snapshot()==e['initial_state'], 'Wrong initial database')
        for i,m in enumerate(messages):
            c.require(set(m)=={'from','value'} and isinstance(m['value'],str) and bool(m['value']), 'Invalid public message')
            c.require(m['from'] in (('human','observation') if i%2==0 else ('gpt','function_call')), 'Role order invalid')
            if m['from']=='human':
                if i:
                    field = 'email' if n==14 else 'title'
                    expected = c.choose(locale,f'补充 {field}={c.canonical(spec["fields"][field])}；其他字段保持原样。',
                                        f'Here is {field}={c.canonical(spec["fields"][field])}; keep the other fields unchanged.')
                    c.require(n in (14,15) and m['value']==expected and messages[i-1]['from']=='gpt', 'Unknown followup or lost clarification')
                human += '\n'+m['value']
            elif m['from']=='function_call':
                calls = json.loads(m['value'])
                c.require(isinstance(calls,list) and len(calls)==1 and messages[i+1]['from']=='observation', 'Missing response or malformed calls')
                call = calls[0]
                c.require(set(call)=={'name','arguments'}, 'Call metadata leak')
                name,args = call['name'],call['arguments']
                c.require(name in env.validators and not list(env.validators[name].iter_errors(args)), 'Invalid schema')
                c.require(env.allow_write or name not in c.WRITE_TOOLS, 'Unauthorized write proposal')
                for key,value in args.items():
                    if key=='requester_id':
                        c.require(value in evidence['users'], 'Unobserved requester')
                    elif key=='article_id':
                        c.require(value in evidence['article_ids'], 'Unobserved article ID')
                    elif key=='ticket_id':
                        c.require(value in human or value in evidence['tickets'], 'Unobserved ticket ID')
                    elif key=='expected_version':
                        c.require(evidence['tickets'].get(args['ticket_id'],{}).get('version')==value, 'Unobserved version')
                    else:
                        c.require(str(value) in human, 'Parameter not supplied by user: '+key)
                if name=='update_ticket_status':
                    c.require(c.TRANSITIONS.get(evidence['tickets'][args['ticket_id']]['status'])==args['status'], 'Skipped status')
                event = e['events'][cursor]
                key = f'{spec["pair_id"]}:{cursor+1}' if name in c.WRITE_TOOLS else None
                c.require(event['call']==call and event['key']==key and event['before']==env.snapshot(), 'Event key/call/before differs')
                response = env.execute(name,args,idempotency_key=key)
                c.require(response==event['response'] and json.loads(messages[i+1]['value'])==[response] and event['after']==env.snapshot(), 'Tool result/state not replayable')
                observe(name,response,evidence)
                history.append(event)
                cursor += 1
            elif m['from']=='observation':
                c.require(messages[i-1]['from']=='function_call', 'Orphan observation')
            else:
                c.require(i in claims, 'Assistant statement has no claim metadata')
                for claim in claims[i]:
                    claim_check(claim,e,evidence,human,history)
                c.require(m['value']=='\n'.join(c.render(x,locale) for x in claims[i]), 'Answer does not match checked claims')
        c.require(cursor==len(e['events'])<=12 and env.snapshot()==e['final_state'] and env.audit==e['environment_audit']
                  and env.external_events==e['external_events'], 'Final state/audit/external witness differs')
    names = [x['call']['name'] for x in history]
    c.require(names==CHAINS[n], 'Requested complete chain or timely stop missing')
    errors = [x['response']['error']['code'] for x in history if not x['response']['ok']]
    c.require(errors==ERRORS.get(n,[]), 'Unexpected/unevidenced error')
    c.require(len(claims)==sum(m['from']=='gpt' for m in messages), 'Unbound assistant claims')
    if n in (14,15):
        c.require(messages[1]['from']=='gpt' and messages[2]['from']=='human' and messages[3]['from']=='function_call', 'Continuation not preserved')
        created = history[1]['response']['result']
        c.require(all(created[k]==spec['fields'][k] for k in ('title','description','priority')), 'Followup lost initial fields')
    final_claims = claims[len(messages)-1]
    if n in (1,2,5):
        successful = [x['response']['result']['article_id'] for x in history if x['call']['name']=='get_kb_article' and x['response']['ok']]
        c.require([x['article']['article_id'] for x in final_claims if x['kind']=='article']==successful, 'Final answer omitted body')
    if n==17:
        c.require([x['call']['arguments']['status'] for x in history[1:]]==['in_progress','resolved','closed'] and e['final_state']['tickets'][0]['status']=='closed', 'Status target incomplete')
    if n==16:
        c.require(final_claims[0]['ticket']['assignee_id']=='A001', 'Assignment target incomplete')
    if not history:
        c.require(e['initial_state']==e['final_state'], 'Zero-call example changed data')
    assistant = [i for i,m in enumerate(messages) if m['from'] in ('gpt','function_call')]
    masked = [i for i,m in enumerate(messages) if m['from'] in ('human','observation')]
    return {'id':e['id'],'passed':True,'calls':cursor,'errors':errors,'public_row_sha256':c.digest(row),
            'claims_sha256':c.digest(e['claims']),'assistant_message_indices_expected_supervised':assistant,
            'message_indices_expected_masked':masked,'actual_token_label_audit':'pending_remote',
            'external_events':len(e['external_events']),'database_changed':e['initial_state']!=e['final_state'],
            'model_write_calls':sum(x in c.WRITE_TOOLS for x in names)}


def validate_population(episodes, blocked=None):
    c.require(len(episodes)==40 and len({e['id'] for e in episodes})==40, 'Wrong pilot count')
    c.require([e['id'] for e in episodes]==[s['pair_id']+'-'+l for s in c.specs() for l in ('zh','en')], 'Pair order changed')
    quota = {'knowledge_chain':10,'authority':8,'priority':8,'clarification_continuation':4,'retention':10}
    c.require(dict(Counter(e['spec']['theme'] for e in episodes))==quota, 'Quota changed')
    records = [validate_episode(e,blocked) for e in episodes]
    for zh,en in zip(episodes[::2],episodes[1::2]):
        for field in ('spec','events','claims','initial_state','final_state','environment_audit','external_events'):
            c.require(zh[field]==en[field], 'Bilingual trace/fact mismatch: '+field)
    a,b = episodes[10],episodes[16]
    c.require(a['row']['conversations'][0]==b['row']['conversations'][0] and a['initial_state']==b['initial_state']
              and not a['events'] and [x['call']['name'] for x in b['events']]==['lookup_user','create_ticket'], 'Permission counterfactual changed')
    c.require(all(c.generate(e['spec'],e['locale'])==e for e in episodes), 'Deterministic reconstruction differs')
    return {'passed':True,'rows':40,'pairs':20,'locales':{'zh':20,'en':20},'quotas_rows':quota,
            'independent_replay_rows':40,'deterministic_reconstruction_rows':40,
            'tool_calls':sum(r['calls'] for r in records),'assistant_turns':sum(len(r['assistant_message_indices_expected_supervised']) for r in records),
            'records':records,'supervision_structure_checked':True,'actual_token_label_audit':'pending_remote',
            'semantic_review':'pending_AI_assisted','human_acceptance':'pending','ready_for_training':False,
            'training_started':False,'gpu_inference_started':False,'old_final_test_read':False}
