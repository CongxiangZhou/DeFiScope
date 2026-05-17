"""
DeFiScope — Blockchain Audit Module
SHA-256 hash computation and verification for recommendation integrity.
"""

import hashlib
import json
from datetime import datetime


class BlockchainAuditModule:
    """
    Computes SHA-256 hashes of recommendation payloads.
    MVP version: local hash computation and verification only.
    Phase 2: integrate with Sepolia smart contract via Web3.py.
    """

    def __init__(self):
        self.hash_records: list[dict] = []

    def compute_sha256(self, payload: str) -> str:
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def _canonical_payload(self, recommendation_dict: dict) -> str:
        """Serialize recommendation content while excluding generated hash metadata."""
        payload_dict = dict(recommendation_dict)
        payload_dict.pop("sha256_hash", None)
        return json.dumps(payload_dict, sort_keys=True, ensure_ascii=False)

    def record_hash(self, recommendation_dict: dict) -> dict:
        """Compute and store hash for a recommendation."""
        payload = self._canonical_payload(recommendation_dict)
        sha256_hash = self.compute_sha256(payload)
        timestamp = datetime.now().isoformat()

        record = {
            "sha256_hash": sha256_hash,
            "timestamp": timestamp,
            "payload_length": len(payload),
        }
        self.hash_records.append(record)
        return record

    def verify_hash(self, recommendation_dict: dict, expected_hash: str) -> bool:
        """Verify that a recommendation matches its recorded hash."""
        payload = self._canonical_payload(recommendation_dict)
        computed = self.compute_sha256(payload)
        return computed == expected_hash

    def get_all_records(self) -> list[dict]:
        return self.hash_records
