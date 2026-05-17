"""
DeFiScope — Unit Test Suite
============================
Author: ZHU Ruiqi
Module: Testing & Validation

Comprehensive unit tests covering:
  - Blockchain audit module (SHA-256 integrity)
  - Data validation (mock_data.json schema)
  - Risk score computation
  - Agent output schema verification
  - Configuration loading

Run:
    python tests/test_agents.py
    python -m pytest tests/ -v
"""

import hashlib
import json
import os
import sys
import unittest
from pathlib import Path

# Ensure project root is on sys.path so we can import sibling modules
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# =========================================================================
# 1. Blockchain Audit Tests
# =========================================================================
class TestBlockchainAudit(unittest.TestCase):
    """Tests for SHA-256 hashing and recommendation integrity verification."""

    def setUp(self):
        self.sample_recommendation = {
            "query": "Allocate $10k aggressive growth",
            "allocation": {"Aave": 35, "Lido": 25, "Uniswap": 20,
                           "MakerDAO": 15, "Compound": 5},
            "explanation": "Diversified across blue-chip protocols.",
            "timestamp": "2026-04-15T14:32:00Z",
        }

    def test_sha256_hash_length(self):
        """SHA-256 hash should be exactly 64 hex characters."""
        payload = json.dumps(self.sample_recommendation, sort_keys=True)
        h = hashlib.sha256(payload.encode("utf-8")).hexdigest()
        self.assertEqual(len(h), 64)
        self.assertTrue(all(c in "0123456789abcdef" for c in h))

    def test_hash_determinism(self):
        """Same input must produce identical hash across runs."""
        payload = json.dumps(self.sample_recommendation, sort_keys=True)
        h1 = hashlib.sha256(payload.encode()).hexdigest()
        h2 = hashlib.sha256(payload.encode()).hexdigest()
        self.assertEqual(h1, h2)

    def test_hash_uniqueness(self):
        """Different inputs must produce different hashes."""
        rec1 = json.dumps(self.sample_recommendation, sort_keys=True)
        modified = dict(self.sample_recommendation)
        modified["allocation"] = {"Aave": 50, "Lido": 50}
        rec2 = json.dumps(modified, sort_keys=True)
        h1 = hashlib.sha256(rec1.encode()).hexdigest()
        h2 = hashlib.sha256(rec2.encode()).hexdigest()
        self.assertNotEqual(h1, h2)

    def test_tampering_detection(self):
        """Modified recommendation should produce different hash (tamper-evident)."""
        original = json.dumps(self.sample_recommendation, sort_keys=True)
        original_hash = hashlib.sha256(original.encode()).hexdigest()

        tampered = dict(self.sample_recommendation)
        tampered["allocation"]["Aave"] = 99
        tampered_str = json.dumps(tampered, sort_keys=True)
        tampered_hash = hashlib.sha256(tampered_str.encode()).hexdigest()
        self.assertNotEqual(original_hash, tampered_hash)

    def test_full_verify_workflow(self):
        """Generate → recompute → verify match."""
        payload = json.dumps(self.sample_recommendation, sort_keys=True)
        generated_hash = hashlib.sha256(payload.encode()).hexdigest()
        recomputed_hash = hashlib.sha256(payload.encode()).hexdigest()
        self.assertEqual(generated_hash, recomputed_hash)

    def test_empty_recommendation(self):
        """Empty dict should still produce a valid hash."""
        h = hashlib.sha256(json.dumps({}).encode()).hexdigest()
        self.assertEqual(len(h), 64)

    def test_unicode_handling(self):
        """Hash should handle non-ASCII characters correctly."""
        rec = {"note": "中文测试 — DeFi投资建议"}
        payload = json.dumps(rec, sort_keys=True, ensure_ascii=False)
        h = hashlib.sha256(payload.encode("utf-8")).hexdigest()
        self.assertEqual(len(h), 64)

    def test_key_order_invariance(self):
        """Hash should be invariant to key insertion order when sorted."""
        a = {"x": 1, "y": 2, "z": 3}
        b = {"z": 3, "x": 1, "y": 2}
        h_a = hashlib.sha256(json.dumps(a, sort_keys=True).encode()).hexdigest()
        h_b = hashlib.sha256(json.dumps(b, sort_keys=True).encode()).hexdigest()
        self.assertEqual(h_a, h_b)

    def test_audit_module_ignores_stored_hash_field(self):
        """Stored hash metadata should not invalidate later verification."""
        from blockchain_audit import BlockchainAuditModule

        audit = BlockchainAuditModule()
        rec = dict(self.sample_recommendation)
        rec["sha256_hash"] = ""

        generated_hash = audit.record_hash(rec)["sha256_hash"]
        rec["sha256_hash"] = generated_hash

        self.assertTrue(audit.verify_hash(rec, generated_hash))


# =========================================================================
# 2. Data Validation Tests
# =========================================================================
class TestDataValidation(unittest.TestCase):
    """Tests for mock_data.json structure and integrity."""

    @classmethod
    def setUpClass(cls):
        cls.data_file = PROJECT_ROOT / "mock_data.json"
        if not cls.data_file.exists():
            cls.data = None
        else:
            with open(cls.data_file, encoding="utf-8") as f:
                cls.data = json.load(f)

    def setUp(self):
        if self.data is None:
            self.skipTest("mock_data.json not found — run from project root")

    def test_data_file_loads(self):
        self.assertIsInstance(self.data, dict)

    def test_protocols_section_exists(self):
        self.assertIn("protocols", self.data)
        self.assertIsInstance(self.data["protocols"], list)
        self.assertGreater(len(self.data["protocols"]), 0)

    def test_protocol_required_fields(self):
        """Every protocol must have name, chain, category, tvl, audit_status."""
        required = {"name", "chain", "category", "tvl", "audit_status"}
        for p in self.data["protocols"]:
            missing = required - set(p.keys())
            self.assertEqual(missing, set(),
                             f"Protocol {p.get('name')} missing: {missing}")

    def test_protocol_tvl_non_negative(self):
        for p in self.data["protocols"]:
            self.assertGreaterEqual(p["tvl"], 0)

    def test_protocol_audit_status_valid(self):
        valid_statuses = {"AUDITED", "PARTIALLY_AUDITED", "UNAUDITED"}
        for p in self.data["protocols"]:
            self.assertIn(p["audit_status"], valid_statuses)

    def test_protocol_composite_risk_in_range(self):
        """Composite risk score must be in [0, 100]."""
        for p in self.data["protocols"]:
            score = p.get("composite_risk_score")
            self.assertIsNotNone(score, f"{p.get('name')} missing composite_risk_score")
            self.assertGreaterEqual(score, 0)
            self.assertLessEqual(score, 100)

    def test_news_section_exists(self):
        self.assertIn("news_articles", self.data)
        self.assertIsInstance(self.data["news_articles"], list)

    def test_news_required_fields(self):
        required = {"title", "source", "date", "sentiment", "protocol_name"}
        for n in self.data["news_articles"]:
            missing = required - set(n.keys())
            self.assertEqual(missing, set(),
                             f"News article missing: {missing}")

    def test_news_sentiment_valid(self):
        """Sentiment label must be one of the recognized categories."""
        valid_sentiments = {"POSITIVE", "NEGATIVE", "NEUTRAL",
                            "VERY_NEGATIVE", "VERY_POSITIVE"}
        for n in self.data["news_articles"]:
            self.assertIn(n["sentiment"], valid_sentiments)


# =========================================================================
# 3. Risk Score Tests
# =========================================================================
class TestRiskScore(unittest.TestCase):
    """Tests for the composite risk score formula."""

    def setUp(self):
        try:
            from config import compute_risk_score, classify_risk
            self.compute = compute_risk_score
            self.classify = classify_risk
        except ImportError:
            self.skipTest("config module not available")

    def test_score_in_valid_range(self):
        score = self.compute(10.0, "audited", 1.5)
        self.assertGreaterEqual(score, 0)
        self.assertLessEqual(score, 100)

    def test_audited_lower_risk_than_unaudited(self):
        s_audited = self.compute(5.0, "audited", 2.0)
        s_unaudited = self.compute(5.0, "unaudited", 2.0)
        self.assertLess(s_audited, s_unaudited)

    def test_higher_tvl_lower_risk(self):
        s_high = self.compute(15.0, "audited", 1.0)
        s_low = self.compute(0.5, "audited", 1.0)
        self.assertLess(s_high, s_low)

    def test_higher_volatility_higher_risk(self):
        s_calm = self.compute(5.0, "audited", 0.5)
        s_volatile = self.compute(5.0, "audited", 15.0)
        self.assertLess(s_calm, s_volatile)

    def test_classification_boundaries(self):
        self.assertEqual(self.classify(15), "Low")
        self.assertEqual(self.classify(45), "Moderate")
        self.assertEqual(self.classify(85), "High")

    def test_extreme_values(self):
        s_min = self.compute(100.0, "audited", 0.0)
        s_max = self.compute(0.0, "unaudited", 100.0)
        self.assertLess(s_min, s_max)


# =========================================================================
# 4. Agent Output Schema Tests
# =========================================================================
class TestAgentOutputSchemas(unittest.TestCase):
    """Validate expected output structures from each agent."""

    def test_orchestrator_decompose_schema(self):
        sample = {
            "subgoals": [
                {"agent": "ChainAgent", "task": "Score protocols by on-chain risk"},
                {"agent": "SentimentAgent", "task": "Classify recent news"},
                {"agent": "RiskProfileAgent", "task": "Match user profile to allocation"},
            ]
        }
        self.assertIn("subgoals", sample)
        for s in sample["subgoals"]:
            self.assertIn("agent", s)
            self.assertIn("task", s)

    def test_chain_agent_schema(self):
        sample = {
            "scored_protocols": [
                {"name": "Aave", "risk_score": 18, "rationale": "audited, high TVL"}
            ]
        }
        for p in sample["scored_protocols"]:
            self.assertIn("name", p)
            self.assertIn("risk_score", p)
            self.assertGreaterEqual(p["risk_score"], 0)
            self.assertLessEqual(p["risk_score"], 100)

    def test_sentiment_agent_schema(self):
        sample = {
            "sentiment_score": 0.62,
            "classifications": [
                {"title": "Aave reaches new high", "sentiment": "POSITIVE"}
            ]
        }
        self.assertGreaterEqual(sample["sentiment_score"], -1)
        self.assertLessEqual(sample["sentiment_score"], 1)

    def test_recommendation_allocation_sums_to_100(self):
        allocation = {"Aave": 35, "Lido": 25, "Uniswap": 20,
                      "MakerDAO": 15, "Compound": 5}
        self.assertEqual(sum(allocation.values()), 100)


# =========================================================================
# 5. Configuration Tests
# =========================================================================
class TestConfig(unittest.TestCase):
    """Tests for config module constants and helpers."""

    def setUp(self):
        try:
            import config
            self.config = config
        except ImportError:
            self.skipTest("config module not available")

    def test_risk_weights_sum_to_one(self):
        total = sum(self.config.RISK_WEIGHTS.values())
        self.assertAlmostEqual(total, 1.0, places=5)

    def test_llm_config_has_required_keys(self):
        required = {"backend", "base_url", "model", "temperature"}
        self.assertTrue(required.issubset(set(self.config.LLM_CONFIG.keys())))

    def test_app_config_metadata(self):
        self.assertEqual(self.config.APP_CONFIG["course_code"], "SC6105")
        self.assertEqual(self.config.APP_CONFIG["team"], "Group 9")


# =========================================================================
# Test Runner
# =========================================================================
def run_all_tests() -> int:
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    for cls in [TestBlockchainAudit, TestDataValidation, TestRiskScore,
                TestAgentOutputSchemas, TestConfig]:
        suite.addTests(loader.loadTestsFromTestCase(cls))

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    print("\n" + "=" * 60)
    print(f"  Total tests : {result.testsRun}")
    print(f"  Passed      : {result.testsRun - len(result.failures) - len(result.errors) - len(result.skipped)}")
    print(f"  Failed      : {len(result.failures)}")
    print(f"  Errors      : {len(result.errors)}")
    print(f"  Skipped     : {len(result.skipped)}")
    print("=" * 60)
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    sys.exit(run_all_tests())
