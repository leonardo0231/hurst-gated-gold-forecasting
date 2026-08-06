from __future__ import annotations

import json
import shutil
import zipfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from .artifacts import artifact_from_path, status_artifact, write_manifest
from .config import PipelineConfig
from .io import atomic_json, decision_hash, sha256_file, write_csv


def _write_decision(path: Path, payload: dict[str, Any]) -> Path:
    payload["decision_hash"] = decision_hash(payload)
    atomic_json(path, payload)
    return path


def _empty_registry(path: Path, registry_id: str, list_field: str) -> Path:
    atomic_json(
        path,
        {
            f"{registry_id}_id": f"{registry_id}_v1",
            "registry_status": "CREATED_EMPTY",
            list_field: [],
            "created_at_utc": datetime.now(UTC).isoformat(),
        },
    )
    return path


def run_phase6(config: PipelineConfig, phase5_decision: Path) -> dict[str, Path]:
    paths = config.paths()
    metadata = paths.artifacts / "metadata"
    report_dir = paths.reports / "phase6"
    table_dir = report_dir / "tables"
    figure_dir = report_dir / "figures"
    manuscript_dir = report_dir / "manuscript"
    for directory in [table_dir, figure_dir, manuscript_dir]:
        directory.mkdir(parents=True, exist_ok=True)
    metrics = pd.read_csv(metadata / "phase5_locked_test_metrics_report.csv")
    comparison = pd.read_csv(metadata / "phase5_baseline_comparison_report.csv")
    claims = json.loads((metadata / "phase5_claim_registry.json").read_text(encoding="utf-8"))[
        "claims"
    ]
    table1 = table_dir / "table_T1_locked_test_performance.csv"
    table2 = table_dir / "table_T2_baseline_comparison.csv"
    write_csv(table1, metrics)
    write_csv(table2, comparison)
    figure = figure_dir / "figure_F1_locked_test_vs_baseline.png"
    plot_data = comparison.copy()
    plot_data["label"] = (
        plot_data["task"].str.replace("_", " ") + " h" + plot_data["horizon"].astype(str)
    )
    ax = plot_data.plot(
        x="label",
        y=["selected_model_metric", "baseline_metric"],
        kind="bar",
        figsize=(14, 6),
        rot=70,
    )
    ax.set_title("Locked-test selected model vs preregistered static baseline")
    ax.set_ylabel("Task-specific primary metric")
    ax.figure.tight_layout()
    ax.figure.savefig(figure, dpi=150)
    plt.close(ax.figure)

    supported = sum(item["status"] == "SUPPORTED" for item in claims)
    conditional = sum(item["status"] == "CONDITIONALLY_SUPPORTED" for item in claims)
    unsupported = sum(item["status"] == "NOT_SUPPORTED" for item in claims)
    sections = {
        "00_title_abstract_keywords.md": (
            "# HGE-Hybrid Gold Forecasting Framework\n\n"
            "This reproducible MVP evaluates a leakage-safe, gold-only forecasting pipeline "
            "on non-confidential deterministic sample data. Results are implementation "
            "evidence, not paper-grade market evidence.\n"
        ),
        "01_introduction.md": (
            "# Introduction\n\nThe project tests a multi-horizon, Hurst-aware forecasting "
            "architecture under strict temporal validation and artifact governance.\n"
        ),
        "02_data_targets_features.md": (
            "# Data, targets, and features\n\nTargets use forward log returns for 1, 5, 10, "
            "and 20 trading days. Direction thresholds are volatility-scaled and trade labels "
            "use a five-basis-point round-trip economic threshold. Features are causal and "
            "gold-only.\n"
        ),
        "03_methods.md": (
            "# Methods\n\nModels are trained with purged walk-forward validation. The final "
            "chronological block is locked until model selection is frozen. Preprocessing is "
            "fit only on each training partition.\n"
        ),
        "04_results.md": (
            f"# Results\n\nThe locked evaluation contains {len(comparison)} task-horizon "
            f"comparisons: {supported} supported, {conditional} conditionally supported, and "
            f"{unsupported} not supported against preregistered baselines. All negative "
            "results remain in the tables.\n"
        ),
        "05_discussion.md": (
            "# Discussion\n\nThe execution verifies the software and research controls. "
            "Statistical results from the deterministic fixture must not be generalized to "
            "financial markets.\n"
        ),
        "06_limitations.md": (
            "# Limitations\n\nThe run uses deterministic sample data, close-based targets, "
            "gold-only features, deferred GARCH, and no paper-grade vendor provenance.\n"
        ),
        "07_conclusion.md": (
            "# Conclusion\n\nThe implementation demonstrates a leakage-safe end-to-end research "
            "workflow. It does not establish paper-grade predictive or trading claims.\n"
        ),
        "08_reproducibility.md": (
            "# Reproducibility\n\nConfiguration, lockfile, manifests, model hashes, deterministic "
            "seeds, tests, and reports are included. Raw proprietary data is not included.\n"
        ),
        "09_appendix.md": (
            "# Appendix\n\nSee artifact manifests and machine-readable decisions for exact "
            "provenance and status semantics.\n"
        ),
    }
    section_paths: list[Path] = []
    for name, content in sections.items():
        path = manuscript_dir / name
        path.write_text(content, encoding="utf-8")
        section_paths.append(path)
    full = manuscript_dir / "full_manuscript_draft.md"
    full.write_text(
        "\n\n".join(path.read_text(encoding="utf-8") for path in section_paths), encoding="utf-8"
    )
    evidence_map = metadata / "phase6_manuscript_evidence_map.json"
    atomic_json(
        evidence_map,
        {
            "evidence_items": [
                {
                    "evidence_id": "E_METRICS",
                    "source_artifact": str(
                        (metadata / "phase5_locked_test_metrics_report.csv").relative_to(paths.root)
                    ),
                    "source_hash": sha256_file(metadata / "phase5_locked_test_metrics_report.csv"),
                    "used_in_sections": ["Results"],
                    "used_in_tables": ["T1"],
                    "evidence_valid": True,
                },
                {
                    "evidence_id": "E_BASELINES",
                    "source_artifact": str(
                        (metadata / "phase5_baseline_comparison_report.csv").relative_to(paths.root)
                    ),
                    "source_hash": sha256_file(metadata / "phase5_baseline_comparison_report.csv"),
                    "used_in_sections": ["Results"],
                    "used_in_tables": ["T2"],
                    "evidence_valid": True,
                },
            ],
            "built_after": {
                "tables_generated": True,
                "figures_generated": True,
                "manuscript_sections_generated": True,
            },
        },
    )
    decision = _write_decision(
        metadata / "phase6_final_manuscript_decision.json",
        {
            "phase6_manuscript_status": "CLOSED",
            "permission_to_enter_phase7": "CONDITIONAL",
            "full_manuscript_draft_path": str(full.relative_to(paths.root)),
            "full_manuscript_draft_hash": sha256_file(full),
            "paper_grade_claims": "NO",
            "new_evidence_created": False,
            "permitted_phase7_scope": {
                "submission_package": "YES",
                "journal_formatting": "YES",
                "language_editing": "YES",
                "reviewer_response": "CONDITIONAL",
                "new_analysis": "NO",
                "claim_upgrade": "NO",
            },
            "created_at_utc": datetime.now(UTC).isoformat(),
        },
    )
    artifacts = [
        artifact_from_path(paths.root, path)
        for path in [table1, table2, figure, full, evidence_map, decision]
    ]
    write_manifest(paths.root, 6, artifacts)
    return {"decision": decision, "manuscript": full, "figure": figure}


def run_phase7(config: PipelineConfig, phase6: dict[str, Path]) -> dict[str, Path]:
    paths = config.paths()
    metadata = paths.artifacts / "metadata"
    submission = paths.reports / "phase7" / "submission"
    submission.mkdir(parents=True, exist_ok=True)
    manuscript = submission / "manuscript_submission_ready.md"
    shutil.copyfile(phase6["manuscript"], manuscript)
    cover = submission / "cover_letter.md"
    cover.write_text(
        "# Cover letter\n\nThis generic submission package contains an audit-controlled MVP "
        "manuscript. No target journal has been selected and no paper-grade claim is made.\n",
        encoding="utf-8",
    )
    availability = submission / "availability_statements.md"
    availability.write_text(
        "# Availability\n\nCode and non-confidential sample data are included. Proprietary "
        "market data, credentials, and restricted model artifacts are not promised for public "
        "release.\n",
        encoding="utf-8",
    )
    package_zip = submission / "phase7_generic_submission_package.zip"
    with zipfile.ZipFile(package_zip, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for file in [manuscript, cover, availability, phase6["figure"]]:
            archive.write(file, arcname=file.name)
    components = [
        artifact_from_path(paths.root, file)
        for file in [manuscript, cover, availability, package_zip]
    ]
    package_manifest = metadata / "phase7_submission_package_manifest.json"
    atomic_json(
        package_manifest,
        {
            "submission_package_id": "phase7_submission_package_v1",
            "target_journal": None,
            "journal_specific_formatting_status": "CONDITIONAL",
            "components": [item.__dict__ for item in components],
            "package_hash": decision_hash({"components": [item.__dict__ for item in components]}),
            "zip_binary_hash": sha256_file(package_zip),
            "zip_binary_hash_used_for_audit_only": True,
        },
    )
    reviewer_registry = _empty_registry(
        metadata / "phase7_reviewer_request_registry.json",
        "reviewer_request_registry",
        "reviewer_requests",
    )
    extension_registry = _empty_registry(
        metadata / "phase7_extension_request_registry.json",
        "extension_request_registry",
        "extension_requests",
    )
    decision = _write_decision(
        metadata / "phase7_final_submission_decision.json",
        {
            "phase7_submission_status": "CONDITIONAL",
            "permission_to_enter_phase8": "CONDITIONAL",
            "target_journal": None,
            "journal_specific_compliance_status": "CONDITIONAL",
            "submission_package_manifest_hash": sha256_file(package_manifest),
            "new_analysis_executed": False,
            "claim_upgrade_detected": False,
            "actual_submission_executed": False,
            "permitted_phase8_scope": {
                "target_journal_resolution": "YES",
                "journal_submission_execution": "CONDITIONAL",
                "editorial_revision": "YES",
                "new_analysis": "NO",
                "claim_upgrade": "NO",
            },
            "created_at_utc": datetime.now(UTC).isoformat(),
        },
    )
    write_manifest(
        paths.root,
        7,
        components
        + [
            artifact_from_path(paths.root, item)
            for item in [package_manifest, reviewer_registry, extension_registry, decision]
        ],
    )
    return {"decision": decision, "package": package_zip, "package_manifest": package_manifest}


def _lifecycle_phase(
    config: PipelineConfig, phase: int, previous_decision: Path
) -> dict[str, Path]:
    paths = config.paths()
    metadata = paths.artifacts / "metadata"
    prefix = f"phase{phase}"
    registries: list[Path] = []
    for stem, field in [
        ("editorial_status_registry", "status_events"),
        ("editorial_communication_registry", "communications"),
        ("reviewer_request_registry", "reviewer_requests"),
        ("revision_action_registry", "revision_actions"),
        ("extension_request_registry", "extension_requests"),
        ("transfer_offer_registry", "transfer_offers"),
    ]:
        path = metadata / f"{prefix}_{stem}.json"
        _empty_registry(path, stem, field)
        registries.append(path)
    receipt = metadata / f"{prefix}_submission_receipt.json"
    atomic_json(
        receipt,
        {
            "submission_receipt_id": f"{prefix}_submission_receipt_v1",
            "submission_executed": False,
            "submission_status": "BLOCKED_PENDING_JOURNAL_SELECTION",
            "submission_receipt_record_status": "CREATED",
            "portal_receipt_file_required": False,
            "portal_receipt_file_path": None,
            "portal_receipt_file_hash": None,
            "manuscript_id": None,
            "reason": "external_submission_not_authorized_or_requested",
            "created_at_utc": datetime.now(UTC).isoformat(),
        },
    )
    privacy = metadata / f"{prefix}_portal_privacy_policy.json"
    atomic_json(
        privacy,
        {
            "store_portal_credentials": False,
            "store_passwords": False,
            "store_session_tokens": False,
            "store_private_account_secrets": False,
            "store_reviewer_identity_if_anonymized": False,
        },
    )
    operating_modes = metadata / f"{prefix}_operating_mode_registry.json"
    mode_names = (
        [
            "submission_followup",
            "post_submission_tracking",
            "reviewer_response",
            "editorial_revision",
            "transfer_governance",
            "publication_archiving",
        ]
        if phase == 9
        else [
            "continued_submission_followup",
            "controlled_submission_execution",
            "post_submission_tracking",
            "reviewer_response",
            "editorial_revision",
            "transfer_package_execution",
            "resubmission_governance",
            "publication_archiving",
            "extension_dispatch",
            "project_closure",
        ]
        if phase == 10
        else [
            "continued_lifecycle_monitoring",
            "controlled_submission_execution",
            "post_submission_tracking",
            "reviewer_response_continuation",
            "editorial_revision_continuation",
            "transfer_package_execution_continuation",
            "resubmission_governance_continuation",
            "publication_archiving_continuation",
            "extension_orchestration",
            "extension_result_reentry",
            "post_closure_audit",
            "publication_metadata_update",
            "project_closure_confirmation",
        ]
    )
    modes: dict[str, Any] = {}
    for name in mode_names:
        active = name in {
            "submission_followup",
            "continued_submission_followup",
            "continued_lifecycle_monitoring",
        }
        modes[name] = {
            "active": active,
            "activation_reason": "generic_package_requires_target_journal"
            if active
            else "trigger_absent",
            "required_registries": [],
            "mode_status": "ACTIVE" if active else "NOT_ACTIVE",
        }
    atomic_json(
        operating_modes,
        {
            "operating_mode_registry_id": f"{prefix}_operating_mode_registry_v1",
            "modes": modes,
            "mode_conflicts_detected": False,
        },
    )
    final_name = {
        8: "final_submission_execution_decision",
        9: "final_editorial_workflow_decision",
        10: "final_lifecycle_decision",
        11: "final_continuation_decision",
    }[phase]
    status_key = {
        8: "phase8_submission_execution_status",
        9: "phase9_editorial_workflow_status",
        10: "phase10_lifecycle_status",
        11: "phase11_continuation_status",
    }[phase]
    permission_key = f"permission_to_enter_phase{phase + 1}"
    decision = _write_decision(
        metadata / f"{prefix}_{final_name}.json",
        {
            status_key: "CONDITIONAL",
            permission_key: "CONDITIONAL",
            "source_decision_hash": sha256_file(previous_decision),
            "actual_submission_executed": False,
            "submission_status": "BLOCKED_PENDING_JOURNAL_SELECTION",
            "current_editorial_status": "NOT_SUBMITTED",
            "operating_mode_registry_hash": sha256_file(operating_modes),
            "operating_mode_status": {
                name: {
                    "active": item["active"],
                    "final_mode_status": "OPEN" if item["active"] else "NOT_ACTIVE",
                }
                for name, item in modes.items()
            },
            "new_analysis_executed": False,
            "new_evidence_created": False,
            "claim_upgrade_detected": False,
            "portal_credentials_stored": False,
            "anonymized_reviewer_identity_stored": False,
            "created_at_utc": datetime.now(UTC).isoformat(),
        },
    )
    artifacts = [
        artifact_from_path(paths.root, item)
        for item in [*registries, receipt, privacy, operating_modes, decision]
    ]
    artifacts += [
        status_artifact(
            f"phase{phase}_external_submission",
            "DEFERRED",
            "manual_authorization_and_target_journal_required",
        )
    ]
    write_manifest(paths.root, phase, artifacts)
    return {"decision": decision, "receipt": receipt, "modes": operating_modes}


def run_lifecycle_phases(
    config: PipelineConfig, phase7: dict[str, Path]
) -> dict[int, dict[str, Path]]:
    outputs: dict[int, dict[str, Path]] = {}
    previous = phase7["decision"]
    for phase in range(8, 12):
        outputs[phase] = _lifecycle_phase(config, phase, previous)
        previous = outputs[phase]["decision"]
    return outputs
