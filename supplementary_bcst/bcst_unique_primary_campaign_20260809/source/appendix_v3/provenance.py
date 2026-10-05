"""Frozen provenance bindings for the reviewer appendix campaign.

This module deliberately contains expected digests rather than discovering a
mutable source tree at run time.  A scientific runner must call
``assert_frozen_provenance`` before constructing an optimizer cell.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping


PLAN_SHA256 = "20476043adf7233c5cd721e620ec4cd620ca58c5becfd1a192bd0495c25baf08"
APPENDIX_PLAN_SHA256 = PLAN_SHA256
CAMPAIGN_ID = "lp_qaoa_reviewer_appendix_v3_20260731"
PLAN_RELATIVE_PATH = "reviewer_appendix_campaign_20260731/EXPERIMENT_PLAN_V3.json"
BASE_PLAN_RELATIVE_PATH = "reviewer_appendix_campaign_20260731/EXPERIMENT_PLAN.json"
BASE_PLAN_SHA256 = "f08094101eed3ea48f7a14d28948fceaaf1c240a7a7a592584974951aabe780c"
AMENDMENT_RELATIVE_PATH = (
    "reviewer_appendix_campaign_20260731/"
    "POSTLAUNCH_OPTIMIZER_GATE_AMENDMENT_20260731.json"
)
AMENDMENT_SHA256 = "20c357669f343f22ad10f927faf633634a683fc76ec848d72dc4cba7315f92be"
EXCLUSION_RELATIVE_PATH = (
    "reviewer_appendix_campaign_20260731/V2_EXCLUSION_MANIFEST_20260731.json"
)
EXCLUSION_SHA256 = "e909ab1b1058df8942d6ed9cf7de2350f1f396c1a905513a5d2313c62c3156f1"
PRELAUNCH_AMENDMENT_RELATIVE_PATH = (
    "reviewer_appendix_campaign_20260731/PRELAUNCH_AMENDMENT_20260731.json"
)
PRELAUNCH_AMENDMENT_SHA256 = (
    "6e6b30711fbd9a03af7d4b25003758888d990faa725e6b63090c2736e9530dc3"
)
PREDECESSOR_CAMPAIGN = "corrected_bcst_campaign_20260730_v2"
PREDECESSOR_SOURCE_ROOT = (
    "corrected_bcst_campaign_20260730_v2/source_v2/bcst_v2"
)

# Every Python module in the predecessor campaign's scientific source package
# is pinned.  The appendix implementation may be new, but its problem/model
# provenance must continue to point to these exact predecessor bytes.
PREDECESSOR_SOURCE_MANIFEST: Mapping[str, str] = MappingProxyType(
    {
        "adam400_capacity_amendment.py": "b3aeea77687d612f065cedd6c6eb79f3e5afa9ba53cde6ad860a997ae09b6914",
        "adam400_depth_rescue.py": "4e777e6ae24461f82c8e6b01773ace14bf90c3add353c6312c8cf5117aa2b123",
        "campaign_controller.py": "5001d6dac0de66dec32a99905c0829ccfe02ea8b2457f4ccae6c32de50358859",
        "campaign_report.py": "45db66372ad11ffd8e02d24c4623a19f02a23d8add6369f1ee0b6ab015c4b833",
        "capacity_benchmark.py": "1628992d1efdf7c4007df732996d6a89305d3e89f158564a94db7ea4ac995b4d",
        "confirmation_analysis.py": "bdbaff980d281ed0581fbff797e0e3efbb988ccfbd11521ce5c68a8f42fabcd8",
        "dynamics.py": "9f9a4e6afdce5452205b55f287362fcb3136850d82b6c44e58d91e6799d0105d",
        "instance_core.py": "f58c0a988952bd2082a7c9fbb04370babf5bf61d7ddede14c4cf6127a0738524",
        "optimizer_protocol.py": "6d46851537bca87e86d9b40b3c5bdddd4f1cdea9bee0c88c2babae892f6ce089",
        "production_runner.py": "1cb625f7df54f437efda23030d79f075f5097d3b12aa4ec4f1dae3b9badda74c",
        "protocol_lifecycle.py": "27a8cad6f50c97ffe25ae0ba2a47ade9fc09066ef61e0122553c9f370e5ef0c1",
        "setting_rescue.py": "a0aeb9368ec0054014a198b04c41d2eaa892c8f3906d8bc6904782ac299662f9",
        "torch_adapter.py": "1228060123ab922d4158e5dfe82b21d22b2e568b44c1fbea5b2927b373d11c1c",
        "tuning_analysis.py": "ba41cbcb7eae819784afb9775af7ce0e7e1a9769f9f160d1f795502e0d51b196",
    }
)


class ProvenanceError(RuntimeError):
    """Raised when frozen plan or predecessor source bytes do not match."""


def canonical_json_bytes(value: Any) -> bytes:
    """Return the campaign's compact, sorted, finite canonical JSON bytes."""

    try:
        encoded = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise ProvenanceError(f"value is not canonical-JSON compatible: {exc}") from exc
    return encoded.encode("utf-8")


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def predecessor_manifest_record() -> dict[str, Any]:
    """Return the immutable predecessor source manifest in canonical order."""

    return {
        "schema": "lp-qaoa-predecessor-source-manifest-v1",
        "campaign": PREDECESSOR_CAMPAIGN,
        "source_root": PREDECESSOR_SOURCE_ROOT,
        "files": dict(sorted(PREDECESSOR_SOURCE_MANIFEST.items())),
    }


PREDECESSOR_SOURCE_MANIFEST_SHA256 = sha256_bytes(
    canonical_json_bytes(predecessor_manifest_record())
)


def _default_workspace_root() -> Path:
    # .../<workspace>/reviewer_appendix_campaign_20260731/source/appendix_v3
    return Path(__file__).resolve().parents[3]


def assert_frozen_provenance(repo_root: str | Path | None = None) -> dict[str, Any]:
    """Verify the V3 plan chain and every pinned predecessor source file."""

    root = Path(repo_root).resolve() if repo_root is not None else _default_workspace_root()
    plan_path = root / PLAN_RELATIVE_PATH
    if not plan_path.is_file():
        raise ProvenanceError(f"frozen plan is missing: {plan_path}")
    observed_plan = sha256_file(plan_path)
    if observed_plan != PLAN_SHA256:
        raise ProvenanceError(
            f"frozen plan hash mismatch: expected {PLAN_SHA256}, observed {observed_plan}"
        )

    frozen_chain = {
        "base_plan": (BASE_PLAN_RELATIVE_PATH, BASE_PLAN_SHA256),
        "prelaunch_amendment": (
            PRELAUNCH_AMENDMENT_RELATIVE_PATH,
            PRELAUNCH_AMENDMENT_SHA256,
        ),
        "postlaunch_amendment": (AMENDMENT_RELATIVE_PATH, AMENDMENT_SHA256),
        "v2_exclusion_manifest": (EXCLUSION_RELATIVE_PATH, EXCLUSION_SHA256),
    }
    verified_chain: dict[str, dict[str, str]] = {}
    for label, (relative, expected) in frozen_chain.items():
        artifact_path = root / relative
        if not artifact_path.is_file():
            raise ProvenanceError(f"frozen {label} is missing: {artifact_path}")
        observed = sha256_file(artifact_path)
        if observed != expected:
            raise ProvenanceError(
                f"frozen {label} hash mismatch: expected {expected}, observed {observed}"
            )
        verified_chain[label] = {"relative_path": relative, "sha256": observed}

    source_root = root / PREDECESSOR_SOURCE_ROOT
    verified: dict[str, str] = {}
    for relative, expected in sorted(PREDECESSOR_SOURCE_MANIFEST.items()):
        source_path = source_root / relative
        if not source_path.is_file():
            raise ProvenanceError(f"predecessor source is missing: {source_path}")
        observed = sha256_file(source_path)
        if observed != expected:
            raise ProvenanceError(
                "predecessor source hash mismatch for "
                f"{relative}: expected {expected}, observed {observed}"
            )
        verified[relative] = observed

    return {
        "campaign": CAMPAIGN_ID,
        "plan_relative_path": PLAN_RELATIVE_PATH,
        "plan_sha256": observed_plan,
        "frozen_protocol_chain": verified_chain,
        "predecessor_campaign": PREDECESSOR_CAMPAIGN,
        "predecessor_source_manifest_sha256": (
            PREDECESSOR_SOURCE_MANIFEST_SHA256
        ),
        "predecessor_source_files": verified,
    }


__all__ = [
    "APPENDIX_PLAN_SHA256",
    "AMENDMENT_RELATIVE_PATH",
    "AMENDMENT_SHA256",
    "BASE_PLAN_RELATIVE_PATH",
    "BASE_PLAN_SHA256",
    "CAMPAIGN_ID",
    "EXCLUSION_RELATIVE_PATH",
    "EXCLUSION_SHA256",
    "PLAN_SHA256",
    "PRELAUNCH_AMENDMENT_RELATIVE_PATH",
    "PRELAUNCH_AMENDMENT_SHA256",
    "PREDECESSOR_CAMPAIGN",
    "PREDECESSOR_SOURCE_MANIFEST",
    "PREDECESSOR_SOURCE_MANIFEST_SHA256",
    "ProvenanceError",
    "assert_frozen_provenance",
    "canonical_json_bytes",
    "predecessor_manifest_record",
    "sha256_bytes",
    "sha256_file",
]
