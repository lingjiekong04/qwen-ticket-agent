"""Record explicit AI judgments after actual reading, never human approval.

NOTES are authored judgments for the fixed uploaded sample, not conclusions
inferred from template claims or from mechanical validation alone.
"""
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from extensions.ticket_policy_retention_build_v1 import core as c, pipeline as p
from scripts import review_ticket_policy_retention_upload_v1 as upload

OUT = upload.OUT
NOTES = {
    'knowledge_chain': {
        '041': '单篇英文正文，两条约束均读取后翻译；未绕过输入配置。',
        '102': '两篇中文脚踏正文；分别记录输入结果、同段比较及跳帧保留均忠实。',
        '105': '两篇英文导出正文；列名、暗场保留、标题比较和授权维护均准确。',
        '108': '两篇中文脚踏正文；逐篇归属正确，建议未声称已经执行。',
        '112': 'sentence 只读权限不阻止正文读取；脚踏两篇依据完整。',
        '117': '英文脚踏正文转中英回答一致；没有以标题代替正文依据。',
        '126': '该层最长样例；两篇中文脚踏正文与双语摘要逐条一致。',
        '132': '脚踏两篇正文均取得；保留跳帧号及批准配置限制。',
        '138': '中文暗场导出正文；不覆盖原件及授权维护建议正确保留。',
        '144': '中文导出两篇；预览/导出标题比较未被误称为完成修复。',
        '163': '英文正文、sentence 权限；脚踏摘要没有新增未读取建议。',
        '165': '英文导出正文；逐篇翻译、条件和动作边界一致。',
        '167': '英文导出正文、JSON 权限；读出正文后才给证据摘要。',
        '174': '中文脚踏正文；分别记录及相同录像段比较没有相互混淆。',
        '180': '中文导出正文；两篇内容、文章 ID 和标题对应正确。',
        '226': '仅请求标题/ID；search 后及时停止，未读正文或编造建议。',
    },
    'authority': {
        '029': 'JSON 拒写即使缺邮箱且用户要求覆盖；零调用，无需试探写工具。',
        '065': '029 的允许写入反事实；先询问邮箱，补齐后查用户并只创建。',
        '081': '动作权限禁止分配，缺 ID 不改变拒写优先级；零调用。',
        '111': '081 的允许反事实；补 ID 后读取 219，再分配 A002 至 220。',
        '133': 'sentence 禁止写，缺 ID 且要求关闭；直接拒写，无隐式关闭。',
        '139': '同类拒写反事实组；不受用户覆盖要求影响，不调用工具。',
        '145': '脚踏业务关闭请求；缺 ID 但系统只读，优先拒绝写操作。',
        '151': '帧计数关闭请求；零工具、零数据变更，中英拒绝一致。',
        '157': '133 的允许反事实；补 ID 后 596→599 合法依次关闭，保持 high/A001。',
        '163': '139 的允许反事实；394→397 合法关闭，最终 low/A001 如实返回。',
        '169': '145 的允许反事实；159→162 逐步推进，最终 closed/high/A001 正确。',
        '175': '151 的允许反事实；725→728 逐步推进，最终 closed/low/A001 正确。',
    },
    'clarification': {
        '006': '缺邮箱立即询问并停止；没有借用数据库身份或调用工具。',
        '047': '补邮箱后续接，原 title/description/high 保留；创建未分配。',
        '048': '动作权限允许；补邮箱前无工具，之后只查用户与创建，high 保留。',
        '050': 'JSON 权限续接；新邮箱仅来自用户补充，low 未被默认值覆盖。',
        '053': '缺 title，补齐后保留邮箱/描述/normal；返回真实新 ID。',
        '063': '缺 title 续接；补充 CSV continuous 标题准确进入 create，保持 high。',
        '084': '缺 description，初轮未根据标题猜描述；补齐后原字段保持。',
        '115': 'priority 缺失，描述中的 highlight 不充当 high 参数；补充后才创建。',
    },
    'faithful': {
        '009': '空 search 结果；明确未取得正文，不编造 KB 建议，也不多调用。',
        '027': '两篇搜索命中但第二正文不存在；只概括第一真实英文正文并明确失败。',
        '058': '未知邮箱查不到用户即停止；未创建账户、工单或替换身份。',
        '072': '未知工单仅 get 后如实报错；不创建替代。先前成功项套话无实际前项，应视为非阻断表达冗余。',
        '089': '第一分配成功保留；第二先读到 resolved，随后真实外部删除导致失败，无重建或承诺。',
        '091': '第一 low/in_progress/A002 成功；第二读取合法但删除后失败，中英部分成功边界准确。',
        '098': '第一原未分配变 A002；第二消失未分配，未把整批描述为全成功。',
        '118': '只创建未分配，open/normal/新 ID 来自工具，明确 unassigned。',
    },
    'priority': {
        '006': '影响/紧迫性三档例子合理，以组织规则为准，不承诺 SLA，零调用。',
        '009': 'high→normal→low 顺序与通用例子一致，不把当前业务症状直接定级。',
        '021': '可比条件限定及三个例子完整；未根据 high 字符串发起业务调用。',
        '024': '概念解释与中英翻译一致，未给具体小时数。',
        '030': 'high 关键中断、normal 有替代办法、low 样式问题；均是可能分类而非绝对 SLA。',
        '044': '输入顺序 low/high/normal 不影响正确 high→normal→low 处理顺序。',
        '072': '明确无组织 SLA 不能换算完成小时数，未调用工具。',
        '103': '区分优先级与生命周期；给出本模拟工具合法路径而非泛称所有平台。',
    },
    'retention': {
        '003': 'lookup→create→assign A001，ID/版本来自创建结果，最终 low/open/A001。',
        '034': '548→551 三步合法关闭；保持 low/未分配，无越过版本来源。',
        '051': '一次冲突，重读 508 后成功至 509；最终 low/in_progress/A002 忠实。',
        '060': '两次冲突，重读 280/281 后成功至 282；resolved 状态允许分配，不偷改状态。',
        '071': '三次冲突，只有两次重试，各自先重读；用尽即停止，不虚报分配。',
        '085': '两张工单各自查询；open/low 与 in_progress/normal 独立报告，无修改。',
        '096': '相同 normal 不导致混淆不同 resolved/open，分别报告未分配。',
        '114': '仅邮箱查询；用户 ID/邮箱/姓名来自返回，未创建工单。',
    },
}


def gate(out):
    result = c.read(out / 'evidence-review.json')
    status = c.read(out / 'evidence-status.json')
    c.require(status['status'] == 'complete' and status['evidence_review_sha256'] == c.sha(out / 'evidence-review.json'), 'Missing/unbound evidence completion')
    c.require(result['passed'] is True and result['all_1800_records_verified'] is True and result['all_1800_public_trajectories_replayed'] is True and result['frozen_192_targeted_oracles_replayed'] is True and result['fixed_60_pair_sample_recomputed'] is True, 'Incomplete evidence checks')
    c.require(result['human_acceptance'] == 'pending' and result['ready_for_training'] is False and result['training_started'] is False and result['gpu_inference_started'] is False and result['old_final_test_read'] is False, 'Wrong stage authority')
    c.require(result['new_review_source_sha256'] == c.sha(Path(upload.__file__)), 'Evidence verifier source changed')
    c.require(result['candidate_manifest_sha256'] == c.sha(c.OUT / 'manifest.json') and result['targeted_eval_manifest_sha256'] == c.sha(c.EVAL / 'manifest.json'), 'Candidate/evaluation provenance changed')
    c.require(set(result['member_sha256']) == upload.EXPECTED, 'Incomplete evidence members')
    files = {}
    for name, digest in result['member_sha256'].items():
        path = out / 'evidence' / name
        c.require(c.sha(path) == digest, 'Uploaded evidence changed: ' + name)
        files[name] = path.read_bytes()
    upload.stage_logs(files)
    c.require(result['read_view_sha256'] == c.sha(out / 'semantic-review-view.md'), 'Review view changed')
    sample = upload.parse(files['remote/review-episodes.json'])
    expected = [f'tr1-{theme}-{serial}' for theme, entries in NOTES.items() for serial in entries]
    c.require(result['sample_pair_ids'] == expected == [e['spec']['pair_id'] for e in sample[::2]], 'Review notes do not cover exact fixed sample')
    c.require(len(sample) == 120 and len(expected) == 60, 'Wrong sample size')
    records = [upload.parse(line) for line in files['remote/records.jsonl'].splitlines()]
    c.require(len(records) == 1800 and len({r['id'] for r in records}) == 1800, 'Missing/duplicate token records')
    indexed = {r['id']: r for r in records}
    for zh, en in zip(sample[::2], sample[1::2]):
        c.require(zh['locale'] == 'zh' and en['locale'] == 'en' and zh['spec'] == en['spec'] and zh['events'] == en['events'], 'Bilingual pair trace differs')
        for e in (zh, en):
            c.require(indexed[e['id']]['row_sha256'] == c.digest(e['row']) and indexed[e['id']]['passed'] is True, 'Sample/token record mismatch')
    return result, sample, indexed


def record(out=OUT):
    # Preflight all exclusive-create outputs before writing any review.
    outputs = ['ai-review-notes.json', 'ai-semantic-review.json', 'report.md', 'status.json']
    for name in outputs:
        p.fresh(out / name)
    evidence, sample, indexed = gate(out)
    notes = []
    for zh, en in zip(sample[::2], sample[1::2]):
        spec = zh['spec']
        pair_id = spec['pair_id']
        notes.append({
            'pair_id': pair_id, 'theme': spec['theme'], 'variant': spec['variant'],
            'permission_profile': spec['permission_profile'], 'allow_write': spec['allow_write'],
            'decision': 'acceptable_with_nonblocking_notes' if pair_id == 'tr1-faithful-072' else 'acceptable',
            'note_zh': NOTES[spec['theme']][pair_id.rsplit('-', 1)[1]],
            'bilingual_request_and_answer_read': True,
            'review_type': 'AI_assisted_semantic_review', 'human_approved': False,
            'row_evidence': [{
                'id': e['id'], 'episode_sha256': c.digest(e), 'row_sha256': c.digest(e['row']),
                'events_sha256': c.digest(e['events']), 'final_state_sha256': c.digest(e['final_state']),
                'input_ids_sha256': indexed[e['id']]['input_ids_sha256'],
                'labels_sha256': indexed[e['id']]['labels_sha256'],
            } for e in (zh, en)],
        })
    p.write(out / 'ai-review-notes.json', notes)
    result = {
        'review_type': 'AI_assisted_semantic_sample_review', 'reviewed_pairs': 60, 'reviewed_rows': 120,
        'allocation': dict(Counter(n['theme'] for n in notes)), 'blocking_issues': [],
        'recommendation': 'accept_sample_review_then_prepare_release_only',
        'nonblocking_notes': [
            '固定分层抽样包含最长样例与变体覆盖，但偏向长文本：知识链 14/16 为多正文，权限 8/12 为状态推进，优先级 5/8 为举例；不是独立均匀随机抽样，也未穷尽变体交叉组合。',
            '英文知识库请求出现句点后紧接 Read 的空格瑕疵，不改变请求含义；本审核不修改冻结数据。',
            'tr1-faithful-072 的单项失败回答含“先前成功项保留”套话，当前没有前项成功；未指名虚构成功，但表达冗余。未知邮箱 058 可更友好地询问修正邮箱，现有回答已如实停止。',
            '这些是模板驱动模拟数据；900 对不等于 900 个独立语义模板，AI 样本审核不证明模型能力提升。',
        ],
        'evidence_review_sha256': c.sha(out / 'evidence-review.json'),
        'sample_episodes_sha256': evidence['member_sha256']['remote/review-episodes.json'],
        'sample_selection_sha256': evidence['member_sha256']['remote/review-pairs.json'],
        'notes_sha256': c.sha(out / 'ai-review-notes.json'),
        'review_recorder_sha256': c.sha(Path(__file__)),
        'candidate_manifest_sha256': evidence['candidate_manifest_sha256'],
        'targeted_eval_manifest_sha256': evidence['targeted_eval_manifest_sha256'],
        'human_acceptance': 'pending', 'per_row_human_inspection_attested': False,
        'ready_for_training': False, 'training_authorized': False, 'training_started': False,
        'gpu_inference_started': False, 'old_final_test_read': False,
    }
    p.write(out / 'ai-semantic-review.json', result)
    lines = [
        '# 保持能力候选数据：上传证据与 AI 辅助样本审核', '',
        '结论：证据核验通过，固定 60 对（120 条）AI 辅助语义抽检未发现阻断问题；建议接受后准备发布，不启动训练。', '',
        '## 已核验的证据', '',
        '- 1800 条／900 对候选训练数据及冻结 192 条定向评测 gold 的独立重放、确定性重建通过。',
        '- 上传的固定 Python 3.11／LLaMA-Factory CPU 审计覆盖 1800 条，最长 2188 tokens；5202 个助手轮次监督证据完整。用户、工具和系统标签屏蔽符合预期，权限文本仍在输入中。',
        '- 本地核验了上传记录的哈希、来源、固定抽样、实际日志和 compare 重算；没有在本机独立再次运行真实 tokenizer，也未加载模型权重。',
        '- 固定 DEV、既有数据、适配器和旧评分未改动；没有读取旧最终测试。', '',
        '## 实际语义抽检范围', '',
        '知识链 16 对；权限 12 对；澄清、忠实返回、优先级及能力保持各 8 对。全部中英请求、补充轮次、最终回答已阅读，并与真实工具结果核对；各对判断与输入／标签来源哈希在 ai-review-notes.json。', '',
        '重点确认：正文需要实际读取；只读优先拒写且不试探；允许写时缺参先询问；部分成功不虚报；冲突先重读、最多两次重试；无 SLA 不承诺时限；多工单信息分别报告。', '',
        '## 非阻断备注及证据局限', '',
    ] + ['- ' + x for x in result['nonblocking_notes']] + [
        '', '## 下一道门禁', '',
        '当前用户接受状态仍为 pending，ready_for_training=false。AI 辅助审核不记为人工逐条批准。需要用户明确接受此 60 对抽检结论后，才生成独立训练发布包；后续训练与 GPU 配对评测仍另行授权。', '',
        '原始远端证据、原候选数据和旧报告保持原样；本目录只新增审核证据。数量和 tokenizer 正确性不等于模型泛化或策略学习成功。', '',
        '## 来源', '',
        '候选 manifest：`' + result['candidate_manifest_sha256'] + '`', '',
        '定向评测 manifest：`' + result['targeted_eval_manifest_sha256'] + '`', '',
        '上传 archive：`' + evidence['archive_sha256'] + '`', '',
        '原 evidence-review.json 的 pending_actual_reading 是审核开始前的不可变阶段快照；本次 ai-semantic-review.json 为完成阅读后的追加记录。', '',
    ]
    p.text(out / 'report.md', '\n'.join(lines))
    p.write(out / 'status.json', {
        'status': 'AI_review_complete_waiting_user_acceptance',
        'ai_review_sha256': c.sha(out / 'ai-semantic-review.json'),
        'report_sha256': c.sha(out / 'report.md'), 'notes_sha256': result['notes_sha256'],
        'human_acceptance': 'pending', 'ready_for_training': False,
        'training_started': False, 'gpu_inference_started': False,
    })
    print(json.dumps({k: result[k] for k in ('reviewed_pairs', 'reviewed_rows', 'blocking_issues', 'human_acceptance', 'ready_for_training', 'training_started')}, ensure_ascii=False, indent=2))
    return result


if __name__ == '__main__':
    record()
