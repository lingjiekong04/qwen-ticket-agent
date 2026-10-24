"""Freeze specification metadata only, not prompts, fixtures, episodes or models."""
from collections import Counter, defaultdict
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'reports/ticket-policy-retention-design-v1'
sys.path.insert(0,str(ROOT))
PROFILES = ['sentence','json','explicit_actions']
EVAL_QUOTAS = {'authority':18,'knowledge_chain':18,'clarification':12,'priority':12,'faithful':12,'retention':24}
BATCH_QUOTAS = {'knowledge_chain':480,'authority':360,'clarification':240,'faithful':240,'priority':240,'retention':240}
VARIANTS = {
    'authority': {'create_denied','create_allowed','assign_denied','assign_allowed','status_denied','status_allowed'},
    'knowledge_chain': {'single_body','multiple_bodies','titles_only'},
    'clarification': {'email_stop','email_followup','title_followup','description_followup','priority_followup'},
    'priority': {'impact_examples','handling_order','no_sla','priority_vs_status'},
    'faithful': {'empty_kb','article_not_found','requester_not_found','ticket_not_found','partial_assignment','created_unassigned'},
    'retention': {'create_assign','status_sequence','conflict_success','conflict_exhaustion','read_two','lookup_only'},
}
PILOT_MANIFEST = '9440e26fb98bd2543add20b858a5e9b5819f6f22b9db27eb8da4abc08b5011da'
PILOT_EPISODES = 'b539bf2e63a20ac5f5ddb14e5faae06b60482a0f0cdf3ed0c2de314e705c0635'
AI_REVIEW = 'ff08e72d1078be774326aac67878762f23c12b6c9585def3474a5674d62017db'
AI_REPORT = 'e55e15e4e35a85f559521f8284025d07137c01e3fd5073ae75652fc191360780'
SOURCE_FILES = ['user-pilot-review-acceptance.json','evaluation-spec.json','batch-proposal.json','extension-spec.json','design.md']


def require(condition,message):
    if not condition:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_new(path,value):
    with path.open('x',encoding='utf-8') as stream:
        stream.write(json.dumps(value,ensure_ascii=False,indent=2)+'\n')


def validate(acceptance,evaluation,batch,extension):
    require((evaluation['seed'],batch['seed'],extension['seed'])==(2026100802,2026100803,2026100804), 'Design seed changed')
    require(acceptance['user_message']=='接受' and acceptance['sample_review_accepted'] is True
            and acceptance['acceptance_kind']=='user_acceptance_of_ai_assisted_review'
            and acceptance['per_row_human_inspection_attested'] is False, 'Acceptance provenance changed')
    require((acceptance['pilot_manifest_sha256'],acceptance['pilot_episodes_sha256'],acceptance['ai_review_sha256'],acceptance['ai_review_report_sha256'])
            ==(PILOT_MANIFEST,PILOT_EPISODES,AI_REVIEW,AI_REPORT),'Acceptance source changed')
    require(all(acceptance[k] is False for k in ('bulk_generation_authorized','training_authorized','gpu_inference_authorized','old_artifacts_rewritten')), 'Acceptance authority expanded')
    require(evaluation['version']=='ticket-policy-retention-evaluation-spec-v1' and evaluation['stage']=='specification_only_not_instantiated_tasks', 'Evaluation stage/version changed')
    require((evaluation['pairs'],evaluation['episodes'],evaluation['locales'])==(96,192,['zh','en']), 'Evaluation size changed')
    require(evaluation['quotas_pairs']==EVAL_QUOTAS and sum(EVAL_QUOTAS.values())==96, 'Evaluation quota changed')
    require(set(evaluation['variant_pairs'])==set(EVAL_QUOTAS), 'Missing evaluation theme')
    for theme,variants in evaluation['variant_pairs'].items():
        require(set(variants)==VARIANTS[theme], 'Evaluation variant coverage changed')
        require(all(type(n) is int and n>0 for n in variants.values()) and sum(variants.values())==EVAL_QUOTAS[theme], 'Invalid evaluation variant quota')
    require(evaluation['variant_pairs']['authority']=={'create_denied':3,'create_allowed':3,'assign_denied':3,'assign_allowed':3,'status_denied':3,'status_allowed':3}, 'Counterfactual factorial missing')
    require(evaluation['permission_profiles']==PROFILES and evaluation['coverage_requirements']['permission_profile_counts']=={p+'_pairs':32 for p in PROFILES}, 'Profile quota changed')
    require(evaluation['authorization']=={'task_instantiation':False,'model_inference':False,'training':False}, 'Evaluation executable authority expanded')
    require(evaluation['immutability']=={'freeze_before_bulk':True,'post_result_edits':False,
            'future_instantiation':'Record complete request, system, tools, fixture, user supplements, faults, oracle and all hashes before training begins. Until then only specifications are frozen.',
            'new_version_on_spec_change':True,'legacy_dev_rows':60,'legacy_diagnostic_rows':72,'legacy_scores_changed':False,'old_final_test_read':False}, 'Evaluation boundary changed')
    families = evaluation['family_reservations']
    require(families['evaluation_namespace']!=families['training_namespace'], 'Namespace collision')
    for kind in ('business','surface'):
        require(len(families['evaluation_'+kind+'_families'])==3 and len(families['training_'+kind+'_families'])==3
                and not set(families['evaluation_'+kind+'_families'])&set(families['training_'+kind+'_families']), 'Reserved families overlap')
    runtime = evaluation['runtime_proposal']
    require((runtime['seed'],runtime['do_sample'],runtime['max_new_tokens'],runtime['context_tokens'],runtime['max_generations_per_complete_episode'],runtime['max_tool_proposals_per_complete_episode'],runtime['conflict_retries_after_initial_attempt'])
            ==(42,False,512,8192,8,12,2),'Inference or retry limits changed')
    require(set(evaluation['visible_inputs']['hidden_fields_excluded'])=={'scenario_goal','fixture','expected_calls','gold_answer','category','variant','validation_metadata','review_annotation'}, 'Hidden input exclusion missing')
    require(batch['status']=='proposed_pending_user_confirmation' and (batch['episodes'],batch['pairs'])==(1800,900)
            and batch['locales']=={'zh':900,'en':900} and batch['quotas_rows']==BATCH_QUOTAS, 'Batch size/status changed')
    require(set(batch['variant_pairs'])==set(BATCH_QUOTAS) and sum(BATCH_QUOTAS.values())==1800, 'Batch population changed')
    for theme,variants in batch['variant_pairs'].items():
        require(set(variants)==VARIANTS[theme], 'Batch variant coverage changed')
        require(all(type(n) is int and n>0 for n in variants.values()) and sum(variants.values())*2==BATCH_QUOTAS[theme], 'Batch variant quota changed')
    require(batch['permission_profiles']=={p+'_pairs':300 for p in PROFILES} and batch['counterfactual_groups']==90, 'Batch profile/counterfactual count changed')
    for action in ('create','assign','status'):
        counts=batch['variant_pairs']['authority']
        require(counts[action+'_allowed']==counts[action+'_denied'] and counts[action+'_allowed']%3==0, 'Batch counterfactual profile balance changed')
    require(batch['source_language']=={'knowledge_chain_english_body_pairs':120,'knowledge_chain_chinese_body_pairs':120,'same_body_language_within_bilingual_pair':True}, 'Batch body language changed')
    require(batch['authorization']=={'extension_generation':False,'bulk_generation':False,'training':False,'gpu_inference':False} and batch['ready_for_training'] is False, 'Batch authority expanded')
    policy = batch['data_policy']
    require(all(policy[k] is False for k in ('old_data_mixed','accepted_pilot_copied_into_batch','dev_in_training','old_diagnostic_copied_into_training','old_final_test_read','manual_success_observations','failed_rows_dropped','model_visible_hidden_metadata','output_overwrite')), 'Unsafe data policy')
    require(policy['real_simulated_returns'] is True and policy['independent_replay_all_rows'] is True, 'Real execution/replay gate missing')
    require(batch['supervision_proposal']=={'template':'qwen','cutoff_len':4096,'train_on_prompt':False,'mask_history':False,'packing':False,'all_assistant_turns_supervised':True,'user_and_tool_returns_masked':True,'real_token_label_audit':'required_remote_before_release'}, 'Supervision changed')
    sample = batch['review_proposal']
    require((sample['new_variant_extension_pairs'],sample['post_cpu_audit_sample_pairs'],sample['post_cpu_audit_sample_rows'])==(12,60,120)
            and sample['sample_allocation_pairs']=={'knowledge_chain':16,'authority':12,'clarification':8,'faithful':8,'priority':8,'retention':8}, 'Review strata changed')
    experiment = batch['experiment_proposal']
    require(experiment['initializer_selection_confirmed'] is False and experiment['training_configuration_confirmed'] is False
            and experiment['learning_rate'] is None and experiment['epochs'] is None and experiment['extra_ablation_training_run'] is False, 'Training recommendation turned into execution')
    require(experiment['future_evaluation_rows_per_variant']=={'smoke':12,'dev':60,'existing_diagnostic':72,'new_targeted':192}
            and 2*sum(experiment['future_evaluation_rows_per_variant'].values())==experiment['future_evaluation_episode_count']==672, 'Evaluation budget arithmetic changed')
    require((extension['pairs'],extension['episodes'],extension['status'])==(12,24,'proposed_specifications_not_generated')
            and extension['training_or_evaluation_rows'] is False and len(extension['cases'])==12, 'Extension size/status changed')
    require({x['id'] for x in extension['cases']}=={f'nx1-p{i:02}' for i in range(1,13)} and all(x['profile'] in PROFILES for x in extension['cases']), 'Extension IDs/profiles changed')
    require(extension['authorization']=={'generation':False,'bulk_generation':False,'training':False,'gpu_inference':False}, 'Extension authority expanded')
    groups=defaultdict(list)
    for case in extension['cases']:
        if case.get('contrast_group'): groups[case['contrast_group']].append(case)
    require(set(groups)=={'nx1-c1','nx1-c2','nx1-c3'}, 'Extension contrasts missing')
    for group,action,profile in zip(('nx1-c1','nx1-c2','nx1-c3'),('create','assign','status'),PROFILES):
        rows=groups[group]
        require(len(rows)==2 and {x['variant'] for x in rows}=={action+'_denied',action+'_allowed'}
                and all(x['theme']=='authority' and x['profile']==profile for x in rows), 'Extension counterpart changed')


def compile_cases(spec):
    """No user prompt, tool result, fixture or gold answer is constructed here."""
    cases=[]
    for theme,variants in spec['variant_pairs'].items():
        counter=0
        for variant,count in variants.items():
            for ordinal in range(count):
                profile=PROFILES[ordinal%3] if theme=='authority' else PROFILES[counter%3]
                action=variant.rsplit('_',1)[0] if theme=='authority' else None
                group=f'ne1-{action}-{profile}' if action else None
                allowed = variant.endswith('_allowed') if action else theme in ('clarification','retention')
                if theme=='priority': allowed=counter%2==0
                if theme=='faithful': allowed=variant in ('requester_not_found','partial_assignment','created_unassigned')
                if variant in ('read_two','lookup_only'): allowed=False
                identity=group or f'ne1-{theme}-{counter+1:03}'
                fixture_seed=int(hashlib.sha256(f'{spec["seed"]}:{identity}'.encode()).hexdigest()[:8],16)
                family_index=(PROFILES.index(profile)+['create','assign','status'].index(action))%3 if action else counter%3
                row={'pair_id':f'ne1-{theme}-{counter+1:03}','theme':theme,'variant':variant,'profile':profile,
                     'allow_write':allowed,'contrast_group':group,'fixture_seed':fixture_seed,
                     'business_family':spec['family_reservations']['evaluation_business_families'][family_index],
                     'surface_family':spec['family_reservations']['evaluation_surface_families'][family_index],
                     'source_language':('en' if counter%2==0 else 'zh') if theme=='knowledge_chain' else None,
                     'locales':['zh','en'],'status':'metadata_only_not_instantiated',
                     'request':None,'fixture':None,'gold_answer':None,'fault_schedule':None}
                cases.append(row)
                counter+=1
    return cases


def validate_cases(cases,spec):
    require(cases==compile_cases(spec) and len({x['pair_id'] for x in cases})==96, 'Blueprint changed or duplicated')
    require(Counter(x['profile'] for x in cases)=={p:32 for p in PROFILES}, 'Blueprint profile balance invalid')
    require(Counter(x['source_language'] for x in cases if x['theme']=='knowledge_chain')=={'en':9,'zh':9}, 'Source language balance invalid')
    groups=defaultdict(list)
    for row in cases:
        if row['contrast_group']: groups[row['contrast_group']].append(row)
        require(row['request'] is None and row['fixture'] is None and row['gold_answer'] is None and row['fault_schedule'] is None, 'Actual task payload added during design freeze')
    require(len(groups)==9,'Missing counterfactual group')
    for group,rows in groups.items():
        require(len(rows)==2 and {x['allow_write'] for x in rows}=={False,True}, 'Missing allowed/denied counterpart')
        for key in ('fixture_seed','profile','business_family','surface_family'):
            require(rows[0][key]==rows[1][key], 'Counterfactual non-policy dimensions changed')


def gate():
    from extensions.ticket_policy_retention_pilot_v1.pipeline import verify
    pilot=verify()
    require(pilot['manifest_sha256']==PILOT_MANIFEST,'Accepted pilot differs')
    require(sha(ROOT/'data/ticket_policy_retention_pilot_v1/episodes.json')==PILOT_EPISODES
            and sha(ROOT/'reports/ticket-policy-retention-pilot-v1/ai-review.json')==AI_REVIEW
            and sha(ROOT/'reports/ticket-policy-retention-pilot-v1/ai-review.md')==AI_REPORT,'Accepted review differs')
    values=[read(OUT/name) for name in SOURCE_FILES[:4]]
    validate(*values)
    return values


def freeze():
    values=gate()
    names=['evaluation-cases.json','freeze.json','tests.log','verification.json']
    require(not any((OUT/name).exists() for name in names),'Existing freeze/output; refuse overwrite')
    cases=compile_cases(values[1])
    validate_cases(cases,values[1])
    write_new(OUT/'evaluation-cases.json',cases)
    sources=SOURCE_FILES+['evaluation-cases.json']
    source_hashes={name:sha(OUT/name) for name in sources}
    code_paths=[Path(__file__).resolve(),ROOT/'tests/test_ticket_policy_retention_design_v1.py']
    receipt={'version':'ticket-policy-retention-design-freeze-v1','scope':'Specification metadata frozen; concrete 192 evaluation tasks not yet instantiated',
             'evaluation_pairs':96,'evaluation_episodes_proposed':192,'concrete_evaluation_episodes_generated':0,
             'batch_episodes_proposed':1800,'batch_episodes_generated':0,'pilot_review_acceptance':'user_acceptance_of_ai_assisted_review',
             'per_row_human_inspection_attested':False,'actual_token_label_audit':'pending_remote',
             'files_sha256':source_hashes,'sources_sha256':{p.relative_to(ROOT).as_posix():sha(p) for p in code_paths},
             'bulk_generation_authorized':False,'training_authorized':False,'gpu_inference_authorized':False,
             'training_started':False,'gpu_inference_started':False,'old_final_test_read':False}
    write_new(OUT/'freeze.json',receipt)
    command=[sys.executable,'-B','-m','unittest','tests.test_ticket_policy_retention_design_v1','-v']
    run=subprocess.run(command,cwd=ROOT,capture_output=True,text=True,encoding='utf-8')
    with (OUT/'tests.log').open('x',encoding='utf-8') as stream: stream.write(run.stdout+run.stderr)
    require(run.returncode==0,'Tests failed; preserve freeze/diagnostics without a pass receipt')
    result=verify()
    result.update({'test_log_sha256':sha(OUT/'tests.log'),'local_python':sys.version.split()[0]})
    write_new(OUT/'verification.json',result)
    print(json.dumps(result,ensure_ascii=False,indent=2))


def verify():
    values=gate()
    receipt=read(OUT/'freeze.json')
    require(receipt['version']=='ticket-policy-retention-design-freeze-v1' and receipt['concrete_evaluation_episodes_generated']==0
            and receipt['batch_episodes_generated']==0 and receipt['actual_token_label_audit']=='pending_remote','Frozen status overstated')
    require(all(receipt[k] is False for k in ('bulk_generation_authorized','training_authorized','gpu_inference_authorized','training_started','gpu_inference_started','old_final_test_read','per_row_human_inspection_attested')), 'Frozen authority expanded')
    require(set(receipt['files_sha256'])==set(SOURCE_FILES+['evaluation-cases.json']), 'Missing freeze file binding')
    require(set(receipt['sources_sha256'])=={'scripts/freeze_ticket_policy_retention_design_v1.py','tests/test_ticket_policy_retention_design_v1.py'}, 'Missing verifier source binding')
    for name,expected in receipt['files_sha256'].items():
        require(Path(name).name==name and sha(OUT/name)==expected,'Frozen specification changed: '+name)
    for name,expected in receipt['sources_sha256'].items():
        path=ROOT/name
        require(path.resolve().is_relative_to(ROOT.resolve()) and sha(path)==expected,'Frozen verifier source changed')
    validate_cases(read(OUT/'evaluation-cases.json'),values[1])
    return {'passed':True,'freeze_sha256':sha(OUT/'freeze.json'),'evaluation_spec_sha256':sha(OUT/'evaluation-spec.json'),
            'evaluation_blueprint_pairs':96,'evaluation_tasks_instantiated':False,'batch_rows_proposed':1800,
            'pilot_user_acceptance':'accepted_AI_assisted_review_not_per_row_human_inspection',
            'bulk_generated':False,'training_started':False,'gpu_inference_started':False,
            'actual_token_label_audit':'pending_remote','legacy_bound_files_checked':35,'old_final_test_read':False}


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('stage',choices=['freeze','verify'])
    args=parser.parse_args()
    if args.stage=='freeze': freeze()
    else: print(json.dumps(verify(),ensure_ascii=False,indent=2))
