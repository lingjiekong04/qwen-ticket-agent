from __future__ import annotations

import copy
import unittest

from extensions.ticket_runtime_contract_guard_v1_1 import core as guard
from extensions.ticket_runtime_evidence_completion_v1_5 import core as c
from extensions.ticket_runtime_general_recovery_activation_dev_v2 import core as source


class EvidenceCompletionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.knowledge = next(task for task in source.generate()[0]
                             if task["primary_tool"] == "search_kb")

    def episode(self):
        return c.EvidenceCompletionEpisode(copy.deepcopy(self.knowledge))

    def test_01_extra_argument_is_blocked_then_projected(self):
        episode = self.episode()
        bad = guard.call("search_kb", query=episode.workflow.kb_query,
                         toolbench_rapid_response="false")
        result, _ = episode.propose(bad, "model_proposal")
        self.assertTrue(result["allowed"])
        self.assertFalse(episode.events[0]["reached_backend"])
        self.assertFalse(episode.events[0]["schema_valid"])
        self.assertEqual(episode.events[1]["call"],
                         guard.call("search_kb", query=episode.workflow.kb_query))

    def test_02_projection_never_fills_missing_required_value(self):
        self.assertIsNone(c._schema_projection(guard.call("search_kb", extra="x")))

    def test_03_projection_never_coerces_value(self):
        self.assertIsNone(c._schema_projection(
            guard.call("search_kb", query=7, extra="x")))

    def test_04_exact_search_reads_only_returned_articles(self):
        episode = self.episode()
        call = guard.call("search_kb", query=episode.workflow.kb_query)
        result, _ = episode.propose(call, "model_proposal")
        self.assertTrue(result["allowed"])
        returned = {row["article_id"] for row in episode.guard.kb_search_results}
        read = {event["call"]["arguments"]["article_id"] for event in episode.events
                if event["call"]["name"] == "get_kb_article"}
        self.assertTrue(read)
        self.assertLessEqual(read, returned)
        self.assertLessEqual(len(read), episode.workflow.max_articles)

    def test_05_knowledge_completion_releases_trusted_answer(self):
        episode = self.episode()
        episode.propose(guard.call("search_kb", query=episode.workflow.kb_query),
                        "model_proposal")
        rendered = episode.render_if_complete(model_draft="unsupported")
        self.assertTrue(rendered["released"])
        self.assertEqual(episode.answers[-1]["provenance"], "trusted_evidence_renderer")

    def test_06_wrong_query_never_reaches_backend(self):
        episode = self.episode()
        bad = guard.call("search_kb", query="wrong", extra="x")
        episode.propose(bad, "model_proposal")
        self.assertFalse(episode.events[0]["reached_backend"])
        self.assertFalse(any(event["reached_backend"] and
                             event["call"].get("arguments", {}).get("query") == "wrong"
                             for event in episode.events))

    def test_07_frozen_v1_4_source_is_not_modified(self):
        from pathlib import Path
        import hashlib
        path = Path(__file__).resolve().parents[2] / "extensions/ticket_runtime_blocked_action_recovery_v1_4/core.py"
        self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(),
                         "ceb56ac97e1a39694addcaa71cb5a6009f8698bf2e7272502bac207d8fbfeda3")


if __name__ == "__main__":
    unittest.main()
