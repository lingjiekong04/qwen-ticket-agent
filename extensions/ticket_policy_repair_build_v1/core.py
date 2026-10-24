"""Independent repair candidate builder. No model, GPU or training entrypoint."""
import copy
from functools import lru_cache
from pathlib import Path
import random
from extensions.ticket_policy_repair_pilot_v1 import core as base
from . import wording as w
s, audit, sim, TOOLS = base.s, base.audit, base.sim, base.TOOLS
ROOT = s.ROOT
HERE = Path(__file__).resolve().parent
OUT = ROOT / 'data/ticket_policy_repair_candidate_v1'
REPORT = ROOT / 'reports/ticket-policy-repair-build-v1'
REVISION, SEED = 'ticket-policy-repair-build-v1', 2026100708
choose = base.choose
system = base.system
ticket_claim = base.ticket_claim
QUOTAS = {'authority': 400, 'clarify': 300, 'faithful': 300, 'priority': 200}
VARIANTS = {'clarify': {'email_stop':40,'email_followup':40,'title_followup':40,'permission_before_missing':30},
 'faithful': {'unassigned':30,'partial':30,'requester_not_found':30,'ticket_not_found':30,'empty_kb':30},
 'priority': {'order':40,'examples':40,'no_sla':20}}


def profile_list(n):
    return ['json']*(n//3+n%3) + ['sentence']*(n//3) + ['explicit']*(n//3)


@lru_cache(maxsize=1)
def specs():
    out=[]
    for category in QUOTAS:
        definitions = [('create',40),('override',35),('assign',25)] if category=='authority' else list(VARIANTS[category].items())
        total = 100 if category=='authority' else sum(n for _,n in definitions)
        profiles=profile_list(total)
        random.Random(f'{SEED}:{category}:profiles').shuffle(profiles)
        ordinal=0
        for variant,count in definitions:
            for slot in range(count):
                identity=f'{category}:{ordinal}'
                rng=random.Random(f'{SEED}:{identity}')
                prefix='RB'+s.digest([REVISION,SEED,identity])[:14].upper()
                asset,symptoms=w.ASSETS[rng.randrange(len(w.ASSETS))]
                symptom=symptoms[rng.randrange(3)]
                location=rng.choice(w.LOCATIONS)
                timing=rng.choice(w.TIMINGS)
                title=f'{asset}: {symptom}'
                description=f'{asset} at {location}: {symptom}; observed {timing}.'
                email=f'contact.{prefix.lower()}@example.test'
                priority=rng.choice(['low','normal','high'])
                users=[{'user_id':'U'+prefix+'1','email':email,'name':'Repair Requester'}]
                tickets=[{'ticket_id':'T'+prefix+str(i),'title':title if i==1 else f'{asset}: archived service record',
                    'description':description if i==1 else 'Previous repair completed; retain historical assignment.',
                    'requester_id':users[0]['user_id'],'priority':priority if i==1 else rng.choice(['low','normal','high']),
                    'status':rng.choice(['open','in_progress','resolved']) if i==1 else 'closed',
                    'assignee_id':rng.choice([None,'A001','A002']) if i==1 else 'A001',
                    'version':rng.randrange(1,150)} for i in (1,2)]
                fixture={'users':users,'tickets':tickets,'articles':[{'article_id':'KB'+prefix+'1',
                    'title':f'{asset} inspection checklist','content':'Record the symptom and equipment label before requesting an approved inspection.'}]}
                query=f'Unlisted service note for {asset} at {location}'
                permissions=[False,True] if category=='authority' else [variant not in {'permission_before_missing','ticket_not_found','empty_kb'} and (category!='priority' or ordinal%2==0)]
                for allowed in permissions:
                    effective=(variant+('_allowed' if allowed else '_denied')) if category=='authority' else variant
                    spec={'pair_id':f'rb1-{len(out)+1:04d}','index':len(out)+1,'category':category,'variant':effective,
                        'profile':profiles[ordinal],'contrast_group':identity if category=='authority' else None,
                        'seed':SEED,'prefix':prefix,'fault_reads':0,'policy':{'allow_write':allowed,'max_calls':12},
                        'fields':{'title':title,'description':description,'priority':priority,'email':email},
                        'fixture':copy.deepcopy(fixture),'missing_email':'absent.'+prefix.lower()+'@example.test',
                        'unknown_ticket':'T'+prefix+'99','empty_query':query,'request_style':ordinal%8,
                        'answer_style':ordinal%8,'audience_style':(ordinal//8)%4,'level_style':(ordinal//32)%3,
                        'business':{'asset':asset,'location':location,'timing':timing},'ordinal':ordinal}
                    out.append(spec)
                ordinal+=1
    s.require(len(out)==600,'Specification quota differs')
    return out


def request(spec,locale):
    v,f=spec['variant'],spec['fields']
    idx=spec['request_style']
    pick=lambda rows: choose(locale,*rows[idx])
    if spec['category']=='priority':
        levels=['low, normal, high','high, low, normal','normal, high, low'][spec['level_style']]
        audience=choose(locale,*w.AUDIENCES[spec['audience_style']])
        prompt=pick(w.CONCEPT_REQUESTS).format(levels=levels,audience=audience)
        if v!='order': prompt+=' '+choose(locale,'三个等级各给一个通用例子，具体分级依组织规则。','Give a generic example for each level; actual classification follows organizational rules.')
        if v=='no_sla': prompt+=' '+choose(locale,'没有 SLA 信息，请不要给出小时或天数承诺。','No SLA is supplied; do not promise a number of hours or days.')
        return prompt
    fields='; '.join(f'{k}={audit.canonical(f[k])}' for k in ('title','description','priority') if v!='title_followup' or k!='title')
    if v.startswith('assign_') or v=='partial':
        targets=spec['fixture']['tickets'][:2 if v=='partial' else 1]
        ids=', '.join(t['ticket_id'] for t in targets)
        prompt=pick(w.ASSIGN_REQUESTS).format(ids=ids)
        if v=='partial': prompt+=' '+choose(locale,'逐张读取；已关闭的保留原分配，不写入，并分别报告结果。','Read each ticket; preserve a closed ticket without writing, and report each outcome separately.')
        return prompt
    if v=='ticket_not_found': return pick(w.READ_REQUESTS).format(id=spec['unknown_ticket'])
    if v=='empty_kb': return pick(w.SEARCH_REQUESTS).format(query=spec['empty_query'])
    email=spec['missing_email'] if v=='requester_not_found' else f['email']
    prompt=pick(w.CREATE_REQUESTS).format(fields=fields)
    if v not in {'email_stop','email_followup','permission_before_missing'}:
        prompt+=' '+choose(locale,f'提交人 email={audit.canonical(email)}。',f'Requester email={audit.canonical(email)}.')
    if v.startswith('override_'): prompt+=' '+choose(locale,*w.OVERRIDES[idx])
    if v in {'email_stop','email_followup'}: prompt+=' '+choose(locale,*w.MISSING_EMAIL[idx])
    elif v=='permission_before_missing': prompt+=' '+choose(locale,'尚缺提交人邮箱，也请先建后补。','The requester email is missing; create first and fill it in later.')
    elif v=='title_followup': prompt+=' '+choose(locale,'标题尚未提供，请先告诉我缺少什么。','The title has not been provided; tell me what is missing first.')
    elif v=='requester_not_found': prompt+=' '+choose(locale,'若找不到用户就停止；不要创建账户或使用其他身份。','Stop if the requester is unknown; do not create an account or use another identity.')
    else: prompt+=' '+choose(locale,'只创建，不分配；说明实际结果，不承诺后续分配。','Create only, without assignment; report the actual outcome without promising future assignment.')
    return prompt


def followup(spec,locale,field):
    value=f'{field}={audit.canonical(spec["fields"][field])}'
    return choose(locale,*w.FOLLOWUPS[spec['request_style']]).format(value=value)


def render(claim,locale,style):
    k=claim['kind']
    pick=lambda rows: choose(locale,*rows[style])
    if k=='denied': return pick(w.REFUSALS)
    if k=='missing':
        field=choose(locale,'提交人邮箱' if claim['field']=='email' else '工单标题','requester email' if claim['field']=='email' else 'ticket title')
        return pick(w.CLARIFICATIONS).format(field=field)
    if k=='priority':
        answer=pick(w.PRIORITIES)
        if claim['examples']: answer+=' '+pick(w.EXAMPLES)
        return answer+' '+pick(w.NO_SLA)
    # All concrete status statements remain grounded in actual observed returns.
    answer=base.render(claim,locale,0)
    if k=='ticket':
        t=claim['ticket']
        if t['assignee_id'] is None: return pick(w.CREATED).format(**t)
        return pick(w.ASSIGNED).format(**t)
    return answer


class Recorder(base.Recorder):
    def __init__(self,spec,locale,env):
        self.spec,self.locale,self.env=spec,locale,env
        self.messages=[{'from':'human','value':request(spec,locale)}]
        self.events,self.claims=[],[]

    def say(self,claims,style=0):
        style=self.spec['answer_style']
        self.claims.append({'index':len(self.messages),'style':style,'claims':copy.deepcopy(claims)})
        self.messages.append({'from':'gpt','value':'\n'.join(render(x,self.locale,style) for x in claims)})


def source_hashes():
    return {p.relative_to(ROOT).as_posix():s.sha(p) for p in sorted(HERE.glob('*.py'))}

def generate(spec, locale):
    with sim.Environment(spec) as env:
        before = env.snapshot()
        r = Recorder(spec, locale, env)
        v, f = spec["variant"], spec["fields"]
        if not spec["policy"]["allow_write"] and spec["category"] != "priority" and v not in {"ticket_not_found", "empty_kb"}:
            style = {"create_denied": 0, "override_denied": 1, "assign_denied": 2, "permission_before_missing": 3}[v]
            r.say([{"kind": "denied"}], style)
        elif spec["category"] == "priority":
            r.say([{"kind": "priority", "order": ["high", "normal", "low"], "examples": v != "order", "sla": None}])
        elif v == "partial" or v.startswith("assign_"):
            claims = []
            targets = spec["fixture"]["tickets"] if v == "partial" else spec["fixture"]["tickets"][:1]
            for original in targets:
                t = r.call("get_ticket", ticket_id=original["ticket_id"])["result"]
                if t["status"] == "closed":
                    claims.append({"kind": "closed", "id": t["ticket_id"], "assignee_id": t["assignee_id"]})
                else:
                    claims.append(ticket_claim(r.call("assign_ticket", ticket_id=t["ticket_id"], assignee_id="A002", expected_version=t["version"])))
            r.say(claims)
        elif v == "ticket_not_found":
            r.call("get_ticket", ticket_id=spec["unknown_ticket"])
            r.say([{"kind": "not_found", "id": spec["unknown_ticket"]}])
        elif v == "empty_kb":
            r.call("search_kb", query=spec["empty_query"])
            r.say([{"kind": "empty", "query": spec["empty_query"]}])
        else:
            if v in {"email_stop", "email_followup", "title_followup"}:
                field = "title" if v == "title_followup" else "email"
                r.say([{"kind": "missing", "field": field}], spec["index"] % 2)
                if v != "email_stop":
                    r.messages.append({"from": "human", "value": followup(spec, locale, field)})
            if v != "email_stop":
                email = spec["missing_email"] if v == "requester_not_found" else f["email"]
                user = r.call("lookup_user", email=email)
                if not user["ok"]:
                    r.say([{"kind": "no_requester", "email": email}])
                else:
                    result = r.call("create_ticket", requester_id=user["result"]["user_id"], **{k: f[k] for k in ("title", "description", "priority")})
                    r.say([ticket_claim(result)])
        row = {"system": system(spec, locale), "tools": audit.canonical(TOOLS), "conversations": copy.deepcopy(r.messages)}
        return {"id": spec["pair_id"] + "-" + locale, "locale": locale, "spec": copy.deepcopy(spec),
                "row": row, "events": r.events, "claims": r.claims, "initial_state": before,
                "final_state": env.snapshot(), "environment_audit": env.audit, "external_events": env.external_events}
