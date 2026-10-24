"""Full public-history replay, observed provenance and independent goal rules."""
import json
import re
from collections import Counter,defaultdict
from . import core as c
from extensions.ticket_policy_retention_pilot_v1 import validate as p

def blocked_sources():
    blocked=p.forbidden()
    # Explicit predecessor allowlist; never glob old final-test directories.
    for name in ('pilot_v1','extension_v1','extension_v1_1'):
        for e in c.read(c.ROOT/f'data/ticket_policy_retention_{name}/episodes.json'):
            blocked['requests'].add(e['row']['conversations'][0]['value'])
            for table in ('users','tickets','articles'):
                for row in e['initial_state'][table]:
                    for key in ('user_id','ticket_id','article_id','email'):
                        if key in row: blocked['ids'].add(row[key])
    blocked['normalized_requests']={p.normalized(x) for x in blocked['requests']}
    return blocked

def check_claim(claim,e,evidence,history,human):
    s=e['spec']; k=claim['kind']; v=s['variant']; t=s['theme']
    if k=='denied':
        c.require(t=='authority' and not s['allow_write'] and not history,'Incorrect/late refusal')
        c.require(claim['action']=={'create':'create','assign':'assign','status':'close'}[v.split('_')[0]],'Wrong refused action')
    elif k=='missing':
        field=s['missing_field']
        value=s['fixture']['tickets'][0]['ticket_id'] if field=='ticket_id' else s['fields'][field]
        absent=(value not in human) if field!='priority' else not re.search(r'"priority"\s*:',human)
        c.require(s['allow_write'] and field and claim['field']==field and not history and absent,'Missing field leaked/guessed')
    elif k=='ticket': c.require(evidence['tickets'].get(claim['ticket']['ticket_id'])==claim['ticket'],'Unsupported ticket claim')
    elif k=='user': c.require(evidence['users'].get(claim['user']['user_id'])==claim['user'],'Unsupported user claim')
    elif k=='summary':
        a=claim['article']; facts=claim['facts']
        c.require(evidence['bodies'].get(a['article_id'])==a and facts==s['facts'][a['article_id']],'Unobserved article/facts')
        separator='' if s['source_language']=='zh' else ' '
        c.require(separator.join(x['source'] for x in facts)==a['content'],'Source clauses not aligned')
    elif k=='titles': c.require(v=='titles_only' and len(history)==1 and claim['articles']==history[0]['response']['result'],'Unobserved title list')
    elif k=='empty': c.require(v=='empty_kb' and len(history)==1 and history[0]['response']=={'ok':True,'replayed':False,'result':[]} and claim['query']==s['query'],'No empty-result evidence')
    elif k=='error':
        c.require(any(x['response'].get('error',{}).get('code')==claim['code'] and claim['target'] in x['call']['arguments'].values() for x in history),'Unsupported failure claim')
    elif k=='conflict':
        c.require(v=='conflict_exhaustion' and claim['ticket_id']==s['fixture']['tickets'][0]['ticket_id']
                  and [x['response'].get('error',{}).get('code') for x in history if x['call']['name']=='assign_ticket']==['version_conflict']*3,'Unobserved exhaustion')
    else: c.require(k=='priority' and t=='priority' and not history and claim['variant']==v and claim['order']==['high','normal','low'] and claim['sla'] is None,'Wrong priority semantic skeleton')

def validate_episode(e,expected,blocked=None):
    s,row,l=e['spec'],e['row'],e['locale']
    c.require(s==expected and l in ('zh','en') and e['id']==s['pair_id']+'-'+l,'Unexpected spec/ID')
    c.require(set(row)=={'system','tools','conversations'} and row['system']==c.system(s,l) and row['tools']==c.canonical(c.TOOLS),'Hidden metadata or changed policy/tools')
    messages=row['conversations']; claims={x['index']:x['claims'] for x in e['claims']}
    c.require(len(messages)%2==0 and messages[0]=={'from':'human','value':c.request(s,l)} and messages[-1]['from']=='gpt','Missing request/final')
    c.require(len(claims)==len(e['claims']),'Duplicate claim indices')
    if blocked:
        public=c.canonical(row)
        ids=set(re.findall(r'(?:U|T|KB)[A-Z0-9]{6,}|[A-Za-z0-9._+-]+@example\.test',public))
        c.require(not ids&blocked['ids'] and p.normalized(messages[0]['value']) not in blocked['normalized_requests'],'Previous identity/request copied')
    evidence={'users':{},'tickets':{},'article_ids':set(),'bodies':{}}; history=[]; human=''
    with c.Environment(s) as env:
        c.require(env.snapshot()==e['initial_state'],'Initial fixture changed')
        for i,m in enumerate(messages):
            c.require(set(m)=={'from','value'} and isinstance(m['value'],str) and m['value'],'Invalid message')
            c.require(m['from'] in (('human','observation') if i%2==0 else ('gpt','function_call')),'Role order')
            c.require(not re.search(r'"(?:fixture|fault_schedule|scenario_goal|gold_answer|validation_metadata|pair_id|theme|variant)"\s*:',m['value']),'Private metadata leaked')
            if m['from']=='human':
                if i: c.require(s['followup'] and i==2 and m['value']==c.supplement(s,l) and messages[1]['from']=='gpt','Lost/changed followup')
                human+='\n'+m['value']
            elif m['from']=='function_call':
                calls=json.loads(m['value'])
                c.require(isinstance(calls,list) and len(calls)==1 and messages[i+1]['from']=='observation','Invalid proposal/response')
                call=calls[0]; c.require(set(call)=={'name','arguments'},'Call metadata')
                name,args=call['name'],call['arguments']
                c.require(name in env.validators and not list(env.validators[name].iter_errors(args)),'Invalid parameter schema')
                c.require(env.allow_write or name not in c.WRITE_TOOLS,'Unauthorized write proposal')
                c.require(not (s['theme']=='authority' and not s['allow_write']),'Denied mutation must not look up identity')
                if s['missing_field']:
                    c.require(s['followup'] and i>2,'Tools before required supplement')
                for key,value in args.items():
                    if key=='requester_id': c.require(value in evidence['users'],'Unobserved requester ID')
                    elif key=='article_id': c.require(value in evidence['article_ids'],'Unobserved article ID')
                    elif key=='ticket_id': c.require(value in human or value in evidence['tickets'],'Unobserved ticket ID')
                    elif key=='expected_version': c.require(evidence['tickets'].get(args['ticket_id'],{}).get('version')==value,'Unobserved version')
                    elif key=='status': c.require(name=='update_ticket_status' and value==c.TRANSITIONS.get(evidence['tickets'].get(args['ticket_id'],{}).get('status')),'Skipped/invalid state')
                    else: c.require(str(value) in human,'Parameter not supplied: '+key)
                if name=='assign_ticket':
                    c.require(evidence['tickets'].get(args['ticket_id'],{}).get('status')!='closed','Known closed assignment proposed')
                    if history and history[-1]['response'].get('error',{}).get('code')=='version_conflict': c.require(False,'Retry without re-read')
                cursor=len(history); c.require(cursor<len(e['events']),'Unrecorded proposal')
                event=e['events'][cursor]; key=f'{s["pair_id"]}:{cursor+1}' if name in c.WRITE_TOOLS else None
                c.require(event['call']==call and event['key']==key and event['before']==env.snapshot(),'Proposal/key/state differs')
                result=env.execute(name,args,idempotency_key=key)
                c.require(result==event['response'] and json.loads(messages[i+1]['value'])==[result] and env.snapshot()==event['after'],'Fabricated observation/state')
                p.observe(name,result,evidence); history.append(event)
            elif m['from']=='observation': c.require(messages[i-1]['from']=='function_call','Orphan observation')
            else:
                c.require(i in claims,'Assistant statement unbound')
                for claim in claims[i]: check_claim(claim,e,evidence,history,human)
                c.require(m['value']=='\n'.join(c.render(x,l) for x in claims[i]),'Answer differs from checked factual claims')
        c.require(len(history)==len(e['events'])<=12 and env.snapshot()==e['final_state'] and env.audit==e['environment_audit'] and env.external_events==e['external_events'],'Final state/audit differs')
        c.require(not env.pending,'Scheduled fault not executed')
    names,errors=c.expected_chain(s)
    c.require([x['call']['name'] for x in history]==names,'Incomplete chain or delayed stop')
    c.require([x['response']['error']['code'] for x in history if not x['response']['ok']]==errors,'Incorrect error/retry count')
    c.require(len(claims)==sum(m['from']=='gpt' for m in messages),'Unbound assistant turn')
    final=claims[len(messages)-1]; v=s['variant']; t=s['theme']
    # Independent final goal checks, not just regeneration equality.
    if 'create_ticket' in names:
        created=next(x['response']['result'] for x in history if x['call']['name']=='create_ticket')
        c.require(all(created[k]==s['fields'][k] for k in ('title','description','priority')) and created['assignee_id'] is None,'Create fields/assignment invented')
        c.require(created['requester_id'] in evidence['users'],'Wrong requester')
        c.require(final==[{'kind':'ticket','ticket':history[-1]['response']['result']}],'Created final result missing')
        c.require(final[0]['ticket']['assignee_id']==('A001' if v=='create_assign' else None),'Wrong actual final assignment')
    if v in ('assign_allowed','conflict_success'): c.require(final[0]['ticket']['assignee_id']=='A002','Target assignment incomplete')
    if v in ('status_allowed','status_sequence'): c.require(final[0]['ticket']['status']=='closed','Target closure incomplete')
    if v=='read_two': c.require([x['ticket'] for x in final]==[x['response']['result'] for x in history],'Independent read conflated')
    if t=='knowledge_chain' and v!='titles_only' or v=='article_not_found':
        success=[x['response']['result'] for x in history if x['call']['name']=='get_kb_article' and x['response']['ok']]
        c.require([x['article'] for x in final if x['kind']=='summary']==success,'Requested body omitted')
    if v=='partial_assignment':
        c.require([x['kind'] for x in final]==['ticket','error'] and final[0]['ticket']['assignee_id']=='A002' and final[1]['code']=='ticket_not_found','Partial success conflated')
        c.require(history[2]['response']['result']['status']!='closed' and len(e['external_events'])==1 and e['external_events'][0]['kind']=='ticket_delete','Failure was predictable before proposed write')
    if not history: c.require(e['initial_state']==e['final_state'],'Tool-free scenario changed DB')
    c.require(c.generate(s,l)==e,'Deterministic reconstruction differs')
    return {'id':e['id'],'passed':True,'calls':len(history),'errors':errors,'public_row_sha256':c.digest(row),
            'claims_sha256':c.digest(e['claims']), 'assistant_message_indices_expected_supervised':[i for i,m in enumerate(messages) if m['from'] in ('gpt','function_call')],
            'message_indices_expected_masked':[i for i,m in enumerate(messages) if m['from'] in ('human','observation')],
            'actual_token_label_audit':'pending_remote'}

def population(episodes,split,blocked=None):
    specs=c.specs(split); count=192 if split=='eval' else 1800
    c.require(len(episodes)==count and [e['id'] for e in episodes]==[s['pair_id']+'-'+l for s in specs for l in ('zh','en')],'Population count/order differs')
    records=[]
    for i,e in enumerate(episodes):
        records.append(validate_episode(e,specs[i//2],blocked))
        if (i+1)%100==0: print(f'{split} replay/reconstruction: {i+1}/{count}',flush=True)
    for zh,en in zip(episodes[::2],episodes[1::2]):
        for key in ('spec','events','claims','initial_state','final_state','environment_audit','external_events'):
            c.require(zh[key]==en[key],'Bilingual fact/action mismatch: '+key)
    groups=defaultdict(list)
    for e in episodes[::2]:
        if e['spec']['contrast_group']: groups[e['spec']['contrast_group']].append(e)
    for pair in groups.values():
        c.require(len(pair)==2,'Counterfactual counterpart missing')
        a,b=sorted(pair,key=lambda e:e['spec']['allow_write'])
        for key in ('fields','fixture','fault_schedule','missing_field','permission_profile','business_family','surface_family'):
            c.require(a['spec'][key]==b['spec'][key],'Counterfactual changed beyond permission: '+key)
        c.require(a['row']['conversations'][0]==b['row']['conversations'][0] and a['initial_state']==b['initial_state'] and not a['events'] and b['events'],'Counterfactual inputs/goals differ')
    proposal=c.read(c.DESIGN/('evaluation-spec.json' if split=='eval' else 'batch-proposal.json'))
    expected={k:n*2 for k,n in proposal['quotas_pairs'].items()} if split=='eval' else proposal['quotas_rows']
    c.require(dict(Counter(e['spec']['theme'] for e in episodes))==expected,'Quota differs')
    c.require(Counter(s['permission_profile'] for s in specs)=={p:len(specs)//3 for p in c.PROFILES},'Profile balance differs')
    for theme,variants in proposal['variant_pairs'].items():
        c.require(Counter(s['variant'] for s in specs if s['theme']==theme)==variants,'Variant quota differs: '+theme)
    lang=Counter(s['source_language'] for s in specs if s['theme']=='knowledge_chain')
    c.require(lang=={'en':proposal['quotas_pairs']['knowledge_chain']//2,'zh':proposal['quotas_pairs']['knowledge_chain']//2} if split=='eval' else lang=={'en':120,'zh':120},'Body language quota differs')
    if split=='eval':
        combos={tuple(sorted(x['priority'] for x in s['fixture']['tickets'])) for s in specs if s['variant']=='read_two'}
        c.require(combos=={('low','normal'),('high','low'),('high','normal'),('normal','normal')},'Read-two coverage incomplete')
    return {'passed':True,'split':split,'rows':count,'pairs':count//2,'quotas_rows':expected,'locales':{'zh':count//2,'en':count//2},
            'profile_pairs':dict(Counter(s['permission_profile'] for s in specs)),'body_language_pairs':dict(lang),
            'counterfactual_groups':len(groups),'tool_calls':sum(r['calls'] for r in records),
            'assistant_turns':sum(len(r['assistant_message_indices_expected_supervised']) for r in records),
            'independent_replay_rows':count,'deterministic_reconstruction_rows':count,'records':records,
            'semantic_review':'pending_AI_assisted_sample_review','actual_token_label_audit':'pending_remote',
            'ready_for_training':False,'training_started':False,'gpu_inference_started':False,'old_final_test_read':False}

def isolation(train,evaluation,blocked):
    def sets(episodes):
        return { 'requests':{e['row']['conversations'][0]['value'] for e in episodes},
                 'normalized_requests':{p.normalized(e['row']['conversations'][0]['value']) for e in episodes},
                 'ids':{value for e in episodes for table in ('users','tickets','articles') for row in e['initial_state'][table] for key,value in row.items() if key in ('user_id','ticket_id','article_id','email')},
                 'bodies':{a['content'] for e in episodes for a in e['initial_state']['articles']},
                 'trajectories':{c.digest(e['row']['conversations']) for e in episodes},
                 'business':{e['spec']['business_family'] for e in episodes},'surface':{e['spec']['surface_family'] for e in episodes}}
    a,b=sets(train),sets(evaluation); overlaps={k:len(a[k]&b[k]) for k in a}
    c.require(not any(overlaps.values()),'Targeted evaluation/train overlap')
    c.require(not a['ids']&blocked['ids'] and not a['normalized_requests']&blocked['normalized_requests'],'Legacy overlap')
    normalized_counts=Counter(p.normalized(e['row']['conversations'][0]['value']) for e in train[::2])
    # Repeated normalized templates are disclosed, not mislabeled unseen semantics.
    return {'passed':True,'eval_training_overlap_counts':overlaps,'previous_ID_overlap':0,'previous_request_overlap':0,
            'training_unique_identity_normalized_requests':len(normalized_counts),'training_pair_count':len(train)//2,
            'max_normalized_template_multiplicity':max(normalized_counts.values()),
            'scope':'Exact content, ID-normalized requests, bodies, full gold transcripts and reserved families. No claim of semantic independence or held-out generalization.',
            'evaluation_oracles_read_only_for_postbuild_isolation':True,'old_final_test_read':False}
