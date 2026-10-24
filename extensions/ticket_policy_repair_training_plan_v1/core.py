"""CPU-only proposed continuation plan. No model/optimizer/inference launcher."""
from collections import Counter, defaultdict
from pathlib import Path
import random
from extensions.ticket_policy_repair_release_v1 import core as release
s = release.s
ROOT = release.ROOT
HERE = Path(__file__).resolve().parent
OUT = ROOT / 'reports/ticket-policy-repair-training-plan-v1'
START = '/konglingjie/sft_lora/lora_sft_toolcall_day1/runs/ticket-sft-v3-hermes-r8-v1.1/train/adapter'
START_SHA = '789144f12d35807f193974a8b753ada3e18327da64557e5551f96815553cb802'
START_CONFIG_SHA = '55b566d72adb5d90795db64ed76c33e5bff318ec032a735b593d4c25e6455bc2'
NEW_RUN = 'runs/ticket-policy-repair-sft-v1'
DIAGNOSTIC_RUN = ROOT / 'reports/ticket-policy-audit-v1/local/inputs/run-ticket-sft-v3.json'
DIAGNOSTIC_RUN_SHA = '9617c1883f46a1b345d5059b8bbef6908d602d7aae81f5635589b8bcaf759b7e'


def recipe():
    return {
        'status': 'proposed_pending_full_plan_confirmation', 'date': '2026-10-07',
        'confirmed_context': {'initializer': 'ticket-sft-v3', 'gpu': '1', 'training_rows': 1200, 'old_data_mixed': False},
        'data': {'path': 'data/ticket_policy_repair_train_release_v1', 'dataset': release.DATASET,
                 'rows': 1200, 'pairs': 600, 'quotas': release.c.QUOTAS, 'shuffle_seed': 42,
                 'release_manifest_sha256': '83e056ca7b6186e28cdecde0b3afaac8c991347f5dfea91f7b4db22570d21aa7',
                 'resplit': False, 'old_data_mixed': False, 'dev_train_rows': 0},
        'initializer': {'base': s.MODEL, 'adapter': START, 'adapter_sha256': START_SHA,
                        'adapter_config_sha256': START_CONFIG_SHA, 'create_new_adapter': False,
                        'adapter_path_input_type': 'string', 'resume_from_checkpoint': None,
                        'optimizer_fresh': True, 'scheduler_fresh': True, 'preflight_weights_reused': False,
                        'runtime_weight_hash_check': 'pending_remote_before_load'},
        'lora': {'rank': 8, 'alpha': 16, 'dropout': 0.05, 'target_modules': sorted(s.TARGETS),
                 'quantization': False, 'dora': False, 'rslora': False, 'extra_adapters': False},
        'training': {'learning_rate': 1e-5, 'epochs': 2, 'batch_size': 1, 'gradient_accumulation': 8,
                     'optimizer': 'adamw_torch', 'adam_beta1': 0.9, 'adam_beta2': 0.999,
                     'adam_epsilon': 1e-8, 'weight_decay': 0.0, 'max_grad_norm': 1.0,
                     'scheduler': 'cosine', 'warmup_ratio': 0.05, 'expected_steps': 300,
                     'steps_per_epoch': 150, 'seed': 42, 'data_seed': 42, 'bf16': True,
                     'attention': 'sdpa', 'gradient_checkpointing': True, 'drop_last': False,
                     'save_strategy': 'epoch', 'save_total_limit': 2, 'final_adapter_only_for_eval': True,
                     'dev_loss': False, 'early_stopping': False, 'best_checkpoint_selection': False},
        'supervision': release.POLICY,
        'preflight': {'rows': 80, 'pairs': 40, 'pairs_per_category': 10, 'authority_complete_groups': 5,
                      'max_steps': 10, 'seed': 42, 'include_longest_each_category': True,
                      'cover_variants_and_profiles': True, 'training_only': True,
                      'formal_training_reloads_original_initializer': True},
        'runtime': {'gpu': '1', 'processes': 1, 'python': '3.11', 'versions': s.STACK,
                    'llamafactory_commit': 'ca75f1edf3cb50343ed1c98605141c3e22075b5f',
                    'offline': True, 'sanitize_distributed_environment': True,
                    'fabricate_rank_environment': False, 'dependency_upgrades': False,
                    'busy_gpu_action': 'stop_without_killing_other_processes', 'run_root': NEW_RUN,
                    'existing_output_action': 'stop_no_overwrite_no_resume'},
        'evaluation': {'variants': ['ticket-sft-v3', 'ticket-policy-repair-v1'],
                       'smoke_rows_per_variant': 12, 'frozen_dev_rows_per_variant': 60,
                       'diagnostic_rows_per_variant': 72, 'total_generated_episodes': 288,
                       'positive_dev_rows': 46, 'negative_dev_rows': 14,
                       'diagnostic_profiles': ['sentence', 'json_equivalent', 'explicit_actions'],
                       'policy': {'seed': 42, 'context_tokens': 8192, 'max_new_tokens': 512,
                                  'max_generations': 8, 'max_calls': 12, 'do_sample': False, 'repetition_penalty': 1.0},
                       'old_then_new_sequential': True, 'original_prompts_tools_scores_preserved': True,
                       'independent_trajectory_replay': True, 'semantic_review': 'pending_after_generation',
                       'ai_review_is_human_approval': False, 'low_score_cases_kept_in_denominator': True,
                       'smoke_quality_threshold': None, 'old_final_test_read': False,
                       'is_generalization_evidence': False,
                       'coverage_limits': ['Existing DEV/diagnostics do not completely measure clarification followup, empty KB and partial-success variants.',
                                           'Existing 72 probes were used to diagnose weaknesses, not an independent final test.']},
        'authorization': {'full_recipe_accepted': False, 'launcher_implementation_authorized': False,
                          'training_authorized': False, 'training_started': False, 'gpu_inference_started': False},
    }


def validate(plan):
    s.require(plan == recipe(), 'Plan differs from proposed recipe or overstates authorization')
    t = plan['training']
    denominator = t['batch_size'] * t['gradient_accumulation'] * plan['runtime']['processes']
    s.require(plan['data']['rows'] % denominator == 0 and plan['data']['rows'] // denominator == t['steps_per_epoch']
              and t['steps_per_epoch'] * t['epochs'] == t['expected_steps'] == 300, 'Optimizer step arithmetic differs')
    s.require(plan['preflight']['rows'] == plan['preflight']['pairs'] * 2
              == plan['preflight']['pairs_per_category'] * len(release.c.QUOTAS) * 2 == 80, 'Preflight quota differs')
    ev = plan['evaluation']
    s.require((ev['smoke_rows_per_variant'] + ev['frozen_dev_rows_per_variant'] + ev['diagnostic_rows_per_variant'])
              * len(ev['variants']) == ev['total_generated_episodes'] == 288, 'Evaluation denominator differs')


def select_preflight(episodes, records):
    s.require(len(episodes) == len(records) == 1200 and [e['id'] for e in episodes] == [r['id'] for r in records], 'Preflight sources differ')
    lengths = {r['id']: r['total_tokens'] for r in records}
    pairs = defaultdict(list)
    for e in episodes: pairs[e['spec']['pair_id']].append(e)
    s.require(len(pairs) == 600 and all(len(v) == 2 and {e['locale'] for e in v} == {'zh', 'en'} for v in pairs.values()), 'Bilingual pairs incomplete')
    score = lambda pid: max(lengths[e['id']] for e in pairs[pid])
    selected, longest = set(), {}
    for category in release.c.QUOTAS:
        ids = [pid for pid, rows in pairs.items() if rows[0]['spec']['category'] == category]
        longest[category] = sorted(ids, key=lambda p: (-score(p), p))[0]
        if category == 'authority':
            groups = defaultdict(list)
            for pid in ids: groups[pairs[pid][0]['spec']['contrast_group']].append(pid)
            s.require(len(groups) == 100 and all(len(v) == 2 for v in groups.values()), 'Broken permission contrast groups')
            ordered = sorted(groups, key=lambda g: (-max(score(p) for p in groups[g]), g))
            chosen = [next(g for g in ordered if longest[category] in groups[g])]
            representative = lambda g: pairs[groups[g][0]][0]['spec']
            for stem in ('create', 'override', 'assign'):
                if not any(representative(g)['variant'].startswith(stem + '_') for g in chosen):
                    chosen.append(next(g for g in ordered if representative(g)['variant'].startswith(stem + '_')))
            for profile in ('json', 'sentence', 'explicit'):
                if not any(representative(g)['profile'] == profile for g in chosen):
                    chosen.append(next(g for g in ordered if representative(g)['profile'] == profile))
            s.require(len(chosen) <= 5, 'Authority preflight coverage exceeds quota')
            rest = sorted(set(groups) - set(chosen))
            random.Random('42:preflight:authority').shuffle(rest)
            chosen += rest[:5-len(chosen)]
            selected.update(pid for g in chosen for pid in groups[g])
        else:
            ordered = sorted(ids, key=lambda p: (-score(p), p))
            chosen = [longest[category]]
            spec = lambda pid: pairs[pid][0]['spec']
            for variant in sorted(release.c.VARIANTS[category]):
                if not any(spec(p)['variant'] == variant for p in chosen):
                    chosen.append(next(p for p in ordered if spec(p)['variant'] == variant))
            for profile in ('json', 'sentence', 'explicit'):
                if not any(spec(p)['profile'] == profile for p in chosen):
                    chosen.append(next(p for p in ordered if spec(p)['profile'] == profile))
            s.require(len(chosen) <= 10, 'Preflight variant coverage exceeds quota')
            rest = sorted(set(ids) - set(chosen))
            random.Random('42:preflight:' + category).shuffle(rest)
            selected.update(chosen + rest[:10-len(chosen)])
    indices = [i for i, e in enumerate(episodes) if e['spec']['pair_id'] in selected]
    s.require(len(indices) == 80 and len(selected) == 40
              and Counter(episodes[i]['spec']['category'] for i in indices) == {k: 20 for k in release.c.QUOTAS}, 'Preflight population differs')
    return indices, longest
