"""One-cell entry point that activates the new-plan binding before V3 CLI use."""

from __future__ import annotations

import os

from appendix_v3 import cell_cli

from .runtime import activate


def main() -> int:
    plan = os.environ.get("BCST_UNIQUE_PLAN_PATH")
    certificate = os.environ.get("BCST_UNIQUE_STAGE1_CERT_PATH")
    certificate_sha = os.environ.get("BCST_UNIQUE_STAGE1_CERT_SHA256")
    if not plan or not certificate or not certificate_sha:
        raise RuntimeError("worker lacks exact plan/Stage-1 certificate environment")
    activate(
        plan,
        import_certificate_path=certificate,
        import_certificate_sha256=certificate_sha,
    )
    return cell_cli.main()


if __name__ == "__main__":
    raise SystemExit(main())
