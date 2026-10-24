"""Replay proposed public calls, audit observed provenance and explicit claims."""
import json
import re
from collections import Counter
from . import core as c
from extensions.ticket_policy_retention_pilot_v1 import validate as old

CHAINS={1:[],2:['lookup_user','create_ticket'],3:[],4:['get_ticket','assign_ticket'],5:[],
        6:['get_ticket','update_ticket_status'],7:['search_kb','get_kb_article'],
        8:['get_ticket','assign_ticket']*3,9:['get_ticket']*2,
        10:['lookup_user','create_ticket'],11:['lookup_user','create_ticket'],
        12:['get_ticket','assign_ticket']*2}
ERRORS={8:['version_conflict']*2,12:['ticket_not_found']}


def assignment_target_gate(evidence,args):
    c.require(evidence['tickets'].get(args['ticket_id'],{}).get('status')!='closed','Known closed ticket must not receive an assignment proposal')


def blocked_sources():
    blocked=old.forbidden()
    episodes=c.read(c.ROOT/'data/ticket_policy_retention_pilot_v1/episodes.json')
    for e in episodes:
        blocked['requests'].add(e['row']['conversations'][0]['value'])
        for table in ('users','tickets','articles'):
            for row in e['initial_state'][table]:
                for key in ('user_id','ticket_id','article_id','email'):
                    if key in row: blocked['ids'].add(row[key])
    blocked['normalized_requests']={old.normalized(x) for x in blocked['requests']}
    return blocked


def check_claim(claim,e,evidence,history,human):
    n,kind=e['spec']['index'],claim['kind']
    if kind=='denied':
        c.require(n in (1,3,5) and not e['spec']['allow_write'] and not history,'Permission refusal late or incorrect')
        c.require(claim['action']=={1:'create',3:'assign',5:'close'}[n],'Incorrect refused action')
    elif kind=='missing':
        field='description' if n==10 else 'priority'
        c.require(n in (10,11) and e['spec']['allow_write'] and not history and claim['field']==field
                  and e['spec']['fields'][field] not in human,'Missing field guessed or already supplied')
    elif kind=='ticket':
        t=claim['ticket']
        c.require(evidence['tickets'].get(t['ticket_id'])==t,'Unsupported ticket state')
    elif kind=='summary':
        a=claim['article']
        c.require(n==7 and evidence['bodies'].get(a['article_id'])==a,'Unobserved body')
        c.require(claim['facts']==c.FACTS and ''.join(x['source'] for x in claim['facts'])==a['content'], 'Summary source clauses not aligned')
    else:
        c.require(kind=='failed_assignment' and n==12 and claim['code']=='ticket_not_found','Unknown/unsupported failure')
        tid=claim['ticket_id']
        c.require(evidence['tickets'][tid]['status']=='resolved' and any(x['call']['name']=='assign_ticket'
                  and x['call']['arguments']['ticket_id']==tid and x['response'].get('error',{}).get('code')=='ticket_not_found' for x in history),'No read-then-missing assignment error evidence')


def validate_episode(e,blocked=None):
    spec,row,locale=e['spec'],e['row'],e['locale']
    c.require(locale in ('zh','en') and spec==c.specs()[spec['index']-1] and e['id']==spec['pair_id']+'-'+locale,'Spec/ID mismatch')
    c.require(set(row)=={'system','tools','conversations'} and row['system']==c.system(spec,locale)
              and json.loads(row['tools'])==c.TOOLS,'Policy/schema/metadata changed')
    messages=row['conversations']
    c.require(messages[0]=={'from':'human','value':c.request(spec,locale)} and messages[-1]['from']=='gpt','Request or final missing')
    c.require(len(messages)%2==0,'Alternating roles missing')
    if blocked is not None:
        public=c.canonical(row)
        ids=set(re.findall(r'(?:U|T|KB)[A-Z0-9]{6,}|[A-Za-z0-9._+-]+@example\.test',public))
        c.require(not ids&blocked['ids'] and old.normalized(messages[0]['value']) not in blocked['normalized_requests'],'Previous identities/request copied')
    claims={x['index']:x['claims'] for x in e['claims']}
    c.require(len(claims)==len(e['claims']),'Duplicate claim index')
    evidence={'users':{},'tickets':{},'article_ids':set(),'bodies':{}}
    history,human=[],''
    with c.Environment(spec) as env:
        c.require(env.snapshot()==e['initial_state'],'Initial state differs')
        for i,m in enumerate(messages):
            c.require(set(m)=={'from','value'} and isinstance(m['value'],str) and m['value'],'Invalid public message')
            c.require(m['from'] in (('human','observation') if i%2==0 else ('gpt','function_call')),'Invalid role order')
            if m['from']=='human':
                if i: c.require(spec['index'] in (10,11) and i==2 and m['value']==c.supplement(spec,locale) and messages[i-1]['from']=='gpt','Lost or changed supplement')
                human+='\n'+m['value']
            elif m['from']=='function_call':
                calls=json.loads(m['value'])
                c.require(isinstance(calls,list) and len(calls)==1 and messages[i+1]['from']=='observation','Malformed call/response')
                call=calls[0]
                c.require(set(call)=={'name','arguments'},'Call hidden metadata leaked')
                name,args=call['name'],call['arguments']
                c.require(name in env.validators and not list(env.validators[name].iter_errors(args)),'Invalid schema')
                c.require(env.allow_write or name not in c.WRITE_TOOLS,'Unauthorized write proposal, even if blocked')
                for key,value in args.items():
                    if key=='requester_id': c.require(value in evidence['users'],'Unobserved requester')
                    elif key=='article_id': c.require(value in evidence['article_ids'],'Unobserved article ID')
                    elif key=='ticket_id': c.require(value in human or value in evidence['tickets'],'Unobserved ticket ID')
                    elif key=='expected_version': c.require(evidence['tickets'].get(args['ticket_id'],{}).get('version')==value,'Unobserved version')
                    else: c.require(str(value) in human,'Unsupplied parameter '+key)
                if name=='update_ticket_status': c.require(c.TRANSITIONS.get(evidence['tickets'][args['ticket_id']]['status'])==args['status'],'Skipped status')
                if name=='assign_ticket': assignment_target_gate(evidence,args)
                if name=='assign_ticket' and history and history[-1]['response'].get('error',{}).get('code')=='version_conflict':
                    c.require(False,'Conflict retry without re-read')
                cursor=len(history)
                c.require(cursor<len(e['events']),'Unrecorded proposal')
                event=e['events'][cursor]
                key=f'{spec["pair_id"]}:{cursor+1}' if name in c.WRITE_TOOLS else None
                c.require(event['call']==call and event['key']==key and event['before']==env.snapshot(),'Call/key/before changed')
                result=env.execute(name,args,idempotency_key=key)
                c.require(result==event['response'] and json.loads(messages[i+1]['value'])==[result] and event['after']==env.snapshot(),'Fabricated return/state')
                old.observe(name,result,evidence)
                history.append(event)
            elif m['from']=='observation': c.require(messages[i-1]['from']=='function_call','Orphan response')
            else:
                c.require(i in claims,'Unbound assistant statement')
                for claim in claims[i]: check_claim(claim,e,evidence,history,human)
                c.require(m['value']=='\n'.join(c.render(x,locale) for x in claims[i]),'Final wording/claims differ')
        c.require(len(history)==len(e['events'])<=12 and env.snapshot()==e['final_state'] and env.audit==e['environment_audit']
                  and env.external_events==e['external_events'],'Final audit/external witness differs')
    n=spec['index']
    c.require([x['call']['name'] for x in history]==CHAINS[n],'Complete chain/timely stopping missing')
    c.require([x['response']['error']['code'] for x in history if not x['response']['ok']]==ERRORS.get(n,[]),'Wrong error/recovery count')
    c.require(len(claims)==sum(m['from']=='gpt' for m in messages),'Assistant claims missing')
    finals=claims[len(messages)-1]
    if n in (2,10,11):
        t=history[-1]['response']['result']
        c.require(all(t[k]==spec['fields'][k] for k in ('title','description','priority')) and t['assignee_id'] is None,'Create lost field or invented assignment')
        c.require(finals==[{'kind':'ticket','ticket':t}],'Created ticket omitted')
    if n in (4,8): c.require(finals[0]['ticket']['assignee_id']=='A002','Assignment target missing')
    if n==6: c.require(finals[0]['ticket']['status']=='closed','Closure incomplete')
    if n==9:
        c.require([x['ticket'] for x in finals]==[x['response']['result'] for x in history]
                  and finals[0]['ticket']['priority']!=finals[1]['ticket']['priority'],'Dual result conflated')
    if n==12:
        c.require([x['kind'] for x in finals]==['ticket','failed_assignment'] and finals[0]['ticket']['assignee_id']=='A002','Partial success omitted/conflated')
        target=spec['fixture']['tickets'][1]['ticket_id']
        witnesses=e['external_events']
        c.require(len(witnesses)==1 and witnesses[0]['kind']=='external_ticket_delete' and witnesses[0]['after_call']==3
                  and witnesses[0]['ticket_id']==target and history[2]['response']['result']['status']=='resolved'
                  and any(t['ticket_id']==target for t in witnesses[0]['before']['tickets'])
                  and not any(t['ticket_id']==target for t in witnesses[0]['after']['tickets']),'Partial failure has no actual post-read deletion witness')
    if not history: c.require(e['initial_state']==e['final_state'],'Zero calls changed DB')
    return {'id':e['id'],'passed':True,'calls':len(history),'errors':ERRORS.get(n,[]),'public_row_sha256':c.digest(row),
            'assistant_message_indices_expected_supervised':[i for i,m in enumerate(messages) if m['from'] in ('gpt','function_call')],
            'message_indices_expected_masked':[i for i,m in enumerate(messages) if m['from'] in ('human','observation')],
            'actual_token_label_audit':'pending_remote','external_events':len(e['external_events'])}


def validate_population(episodes,blocked=None):
    c.require(len(episodes)==24 and [e['id'] for e in episodes]==[s['pair_id']+'-'+l for s in c.specs() for l in ('zh','en')],'Wrong count/order')
    records=[validate_episode(e,blocked) for e in episodes]
    for zh,en in zip(episodes[::2],episodes[1::2]):
        for field in ('spec','events','claims','initial_state','final_state','environment_audit','external_events'):
            c.require(zh[field]==en[field],'Bilingual facts/actions changed: '+field)
    for denied,allowed in ((episodes[0],episodes[2]),(episodes[4],episodes[6]),(episodes[8],episodes[10])):
        c.require(denied['initial_state']==allowed['initial_state'] and denied['row']['conversations'][0]==allowed['row']['conversations'][0]
                  and denied['spec']['fields']==allowed['spec']['fields'] and not denied['events'] and allowed['events'],'Counterfactual treatment changed')
    c.require(all(c.generate(e['spec'],e['locale'])==e for e in episodes),'Deterministic reconstruction changed')
    return {'passed':True,'rows':24,'pairs':12,'locales':{'zh':12,'en':12},'quotas_rows':dict(Counter(e['spec']['theme'] for e in episodes)),
            'independent_replay_rows':24,'deterministic_reconstruction_rows':24,'tool_calls':sum(x['calls'] for x in records),
            'assistant_turns':sum(len(x['assistant_message_indices_expected_supervised']) for x in records),'records':records,
            'actual_token_label_audit':'pending_remote','AI_semantic_review':'pending','human_acceptance':'pending',
            'ready_for_training':False,'bulk_generated':False,'training_started':False,'gpu_inference_started':False,'old_final_test_read':False}
