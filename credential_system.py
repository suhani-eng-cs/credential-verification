from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sqlite3
from typing import Any

import numpy as np
import qrcode
from sklearn.ensemble import IsolationForest


DEFAULT_DB_PATH = Path("credential_verification.db")


def canonical_payload(record: dict[str, Any]) -> str:
    """Serialize credential fields deterministically for hashing."""
    normalized = {
        "credential_id": str(record["credential_id"]).strip(),
        "holder_name": str(record["holder_name"]).strip(),
        "credential_type": str(record["credential_type"]).strip(),
        "issue_date": str(record["issue_date"]).strip(),
    }
    return json.dumps(normalized, sort_keys=True, separators=(",", ":"))


def generate_credential_hash(record: dict[str, Any]) -> str:
    """Return a SHA-256 hex digest for a credential record."""
    payload = canonical_payload(record).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def generate_qr_code(credential_id: str, payload_hash: str, output_path: Path) -> Path:
    """Create a QR code PNG containing the credential ID and SHA-256 hash."""
    qr_data = json.dumps(
        {"credential_id": credential_id, "hash": payload_hash},
        sort_keys=True,
        separators=(",", ":"),
    )
    qr = qrcode.QRCode(version=1, box_size=8, border=3)
    qr.add_data(qr_data)
    qr.make(fit=True)
    image = qr.make_image(fill_color="black", back_color="white")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)
    return output_path


class FraudDetector:
    """Anomaly detector over [requests_per_min, hash_mismatches, hour_of_day]."""

    def __init__(self, contamination: float = 0.05, threshold: float = 0.65) -> None:
        self.model = IsolationForest(contamination=contamination, random_state=42)
        self.threshold = threshold
        self.is_fitted = False

    def fit(self, baseline_samples: np.ndarray) -> None:
        arr = np.asarray(baseline_samples, dtype=float)
        if arr.ndim != 2 or arr.shape[1] != 3 or len(arr) < 2:
            raise ValueError("baseline_samples must have shape (n, 3) with at least 2 rows.")
        self.model.fit(arr)
        self.is_fitted = True

    def predict_risk(
        self, requests_per_min: float, hash_mismatches: float, hour_of_day: int
    ) -> tuple[float, bool]:
        if not self.is_fitted:
            return 0.0, False
        features = np.array([[float(requests_per_min), float(hash_mismatches), float(hour_of_day)]])
        raw_decision = float(self.model.decision_function(features)[0])
        # Lower/negative decision_function values indicate stronger anomalies.
        risk_score = 1.0 / (1.0 + float(np.exp(raw_decision * 5.0)))
        risk_score = round(min(max(risk_score, 0.0), 1.0), 4)
        return risk_score, risk_score >= self.threshold


class CredentialVerificationSystem:
    """End-to-end issuer, verifier, fraud scorer, and audit logger."""

    def __init__(self, db_path: Path = DEFAULT_DB_PATH, fraud_threshold: float = 0.65) -> None:
        self.db_path = Path(db_path)
        self.fraud_detector = FraudDetector(threshold=fraud_threshold)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON;")
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS credentials (
                    credential_id TEXT PRIMARY KEY,
                    holder_name TEXT NOT NULL,
                    credential_type TEXT NOT NULL,
                    issue_date TEXT NOT NULL,
                    payload_hash TEXT NOT NULL UNIQUE,
                    status TEXT NOT NULL CHECK (status IN ('ACTIVE', 'REVOKED')),
                    created_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS verification_logs (
                    log_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    credential_id TEXT NOT NULL,
                    submitted_hash TEXT NOT NULL,
                    result_status TEXT NOT NULL CHECK (
                        result_status IN ('VALID', 'TAMPERED', 'REVOKED', 'INVALID')
                    ),
                    fraud_score REAL NOT NULL,
                    flagged INTEGER NOT NULL,
                    verified_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS fraud_assessments (
                    assessment_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    log_id INTEGER NOT NULL,
                    requests_per_min REAL NOT NULL,
                    hash_mismatches REAL NOT NULL,
                    hour_of_day INTEGER NOT NULL,
                    fraud_score REAL NOT NULL,
                    is_flagged INTEGER NOT NULL,
                    assessed_at TEXT NOT NULL,
                    FOREIGN KEY (log_id) REFERENCES verification_logs (log_id)
                );
                """
            )

    def train_fraud_model(self, baseline_samples: np.ndarray) -> None:
        self.fraud_detector.fit(baseline_samples)

    def issue_credential(
        self,
        credential_id: str,
        holder_name: str,
        credential_type: str,
        issue_date: str,
        qr_output_path: Path | None = None,
    ) -> dict[str, Any]:
        record = {
            "credential_id": credential_id,
            "holder_name": holder_name,
            "credential_type": credential_type,
            "issue_date": issue_date,
        }
        digest = generate_credential_hash(record)
        now_iso = datetime.now(timezone.utc).isoformat()

        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO credentials (
                    credential_id, holder_name, credential_type, issue_date,
                    payload_hash, status, created_at
                ) VALUES (?, ?, ?, ?, ?, 'ACTIVE', ?)
                """,
                (
                    record["credential_id"],
                    record["holder_name"],
                    record["credential_type"],
                    record["issue_date"],
                    digest,
                    now_iso,
                ),
            )

        qr_path: str | None = None
        if qr_output_path is not None:
            qr_path = str(generate_qr_code(record["credential_id"], digest, qr_output_path))

        return {
            "credential_id": record["credential_id"],
            "payload_hash": digest,
            "status": "ACTIVE",
            "qr_code_path": qr_path,
        }

    def revoke_credential(self, credential_id: str) -> bool:
        with self._connect() as conn:
            cursor = conn.execute(
                "UPDATE credentials SET status = 'REVOKED' WHERE credential_id = ?",
                (credential_id.strip(),),
            )
            return cursor.rowcount > 0

    def verify_credential(
        self,
        credential_id: str,
        submitted_hash: str,
        requests_per_min: float = 1.0,
        hash_mismatches: float = 0.0,
        hour_of_day: int | None = None,
    ) -> dict[str, Any]:
        clean_id = credential_id.strip()
        clean_hash = submitted_hash.strip().lower()
        now_dt = datetime.now(timezone.utc)
        now_iso = now_dt.isoformat()
        hour = now_dt.hour if hour_of_day is None else int(hour_of_day)

        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT credential_id, holder_name, credential_type, issue_date,
                       payload_hash, status
                FROM credentials
                WHERE credential_id = ?
                """,
                (clean_id,),
            ).fetchone()

            if row is None:
                status = "INVALID"
                effective_mismatches = max(hash_mismatches, 1.0)
            elif row["status"] == "REVOKED":
                status = "REVOKED"
                effective_mismatches = hash_mismatches
            else:
                recomputed = generate_credential_hash(dict(row))
                hashes_match = (clean_hash == recomputed) and (row["payload_hash"] == recomputed)
                if hashes_match:
                    status = "VALID"
                    effective_mismatches = hash_mismatches
                else:
                    status = "TAMPERED"
                    effective_mismatches = max(hash_mismatches, 1.0)

            fraud_score, flagged = self.fraud_detector.predict_risk(
                requests_per_min=requests_per_min,
                hash_mismatches=effective_mismatches,
                hour_of_day=hour,
            )

            cursor = conn.execute(
                """
                INSERT INTO verification_logs (
                    credential_id, submitted_hash, result_status,
                    fraud_score, flagged, verified_at
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (clean_id, clean_hash, status, fraud_score, int(flagged), now_iso),
            )
            log_id = cursor.lastrowid

            conn.execute(
                """
                INSERT INTO fraud_assessments (
                    log_id, requests_per_min, hash_mismatches, hour_of_day,
                    fraud_score, is_flagged, assessed_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    log_id,
                    float(requests_per_min),
                    float(effective_mismatches),
                    hour,
                    fraud_score,
                    int(flagged),
                    now_iso,
                ),
            )

        return {
            "log_id": log_id,
            "credential_id": clean_id,
            "status": status,
            "fraud_score": fraud_score,
            "flagged": flagged,
        }


def run_demo() -> None:
    db_file = Path("demo_credentials.db")
    if db_file.exists():
        db_file.unlink()

    system = CredentialVerificationSystem(db_path=db_file, fraud_threshold=0.65)

    # Fit baseline on typical daytime verification traffic.
    baseline = np.array(
        [
            [1.0, 0.0, 10],
            [2.0, 0.0, 11],
            [1.5, 0.0, 14],
            [2.5, 0.0, 15],
            [1.0, 0.0, 16],
            [3.0, 0.0, 12],
            [2.0, 0.0, 13],
            [1.0, 0.0, 17],
        ]
    )
    system.train_fraud_model(baseline)

    issued = system.issue_credential(
        credential_id="CRED-2026-104",
        holder_name="Riya Verma",
        credential_type="B.Tech Computer Science",
        issue_date="2026-11-22",
        qr_output_path=Path("output/riya_qr.png"),
    )
    print("Issued:", issued)

    valid_check = system.verify_credential(
        credential_id="CRED-2026-104",
        submitted_hash=issued["payload_hash"],
        requests_per_min=2.0,
        hash_mismatches=0.0,
        hour_of_day=14,
    )
    print("Valid check:", valid_check)

    tampered_check = system.verify_credential(
        credential_id="CRED-2026-104",
        submitted_hash="0" * 64,
        requests_per_min=50.0,
        hash_mismatches=8.0,
        hour_of_day=3,
    )
    print("Tampered + anomalous check:", tampered_check)


if __name__ == "__main__":
    run_demo()