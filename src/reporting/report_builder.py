"""Evidence-backed, pre-adjudication verification report generation.

This report is for human review, not claim approval or rejection.
"""
from collections import Counter
from datetime import datetime, timezone
from uuid import uuid4

ALLOWED_STATUSES = {
    "passed", "issue_detected", "requires_review",
    "not_applicable", "not_evaluated",
}


def _finding(check_type, status, reason, *, evidence=None,
             rule=None, field_name=None, source_ids=None):
    if status not in ALLOWED_STATUSES:
        raise ValueError(f"Invalid finding status: {status}")
    return {
        "finding_id": str(uuid4()),
        "check_type": check_type,
        "field_name": field_name,
        "status": status,
        "reason": reason,
        "review_required": status in {"issue_detected", "requires_review"},
        "source_ids": source_ids or [],
        "evidence": evidence or [],
        "rule": rule,
    }


def build_verification_report(claim_id, verification, requirements,
                              *, requirement_source="demonstration_only",
                              rule_id=None, rule_version=None):
    """Build a JSON-serializable report from the existing verifier outputs."""
    if not str(claim_id).strip():
        raise ValueError("claim_id is required")
    findings = []

    for result in verification.get("results", []):
        field = result["field_name"]
        status = result["status"]
        observations = result.get("observations", [])
        sources = sorted({str(o["source_id"]) for o in observations})

        if status == "conflict":
            finding_status = "issue_detected"
            reason = f"Conflicting values for {field} across sources"
        elif status == "consistent":
            finding_status = "passed"
            reason = f"Matching values for {field} across independent sources"
        elif status == "insufficient_evidence":
            finding_status = "not_evaluated"
            reason = f"Insufficient independent sources to compare {field}"
        else:
            finding_status = "requires_review"
            reason = f"Unrecognized comparison status for {field}: {status}"

        # Retain the exact source object: page/text evidence, row/cell, or JSON path.
        evidence = [
            {
                "source_id": observation.get("source_id"),
                "record_id": observation.get("record_id"),
                "value": observation.get("value"),
                "normalized_value": observation.get("normalized_value"),
                "location": observation.get("source"),
            }
            for observation in observations
        ]
        findings.append(_finding(
            "cross_source_consistency", finding_status, reason,
            field_name=field, source_ids=sources, evidence=evidence,
        ))

    for issue in verification.get("review_issues", []):
        source_ids = issue.get("source_ids") or (
            [issue["source_id"]] if issue.get("source_id") else []
        )
        findings.append(_finding(
            "verification_review_issue", "requires_review",
            f"Verifier flagged: {issue.get('type', 'unknown_issue')}",
            field_name=issue.get("field_name"),
            source_ids=source_ids,
            evidence=[{"issue": issue}],
        ))

    rule = {
        "rule_id": rule_id,
        "rule_version": rule_version,
        "source": requirement_source,
        "authoritative": requirement_source != "demonstration_only"
                         and bool(rule_id and rule_version),
    }

    if requirements is None or "satisfied" not in requirements:
        findings.append(_finding(
            "documentation_requirements", "not_evaluated",
            "No usable documentation-requirement evaluation is available",
            evidence=[{"evaluation": requirements}], rule=rule,
        ))
    elif requirements["satisfied"]:
        findings.append(_finding(
            "documentation_requirements", "passed",
            "Configured demonstration documentation requirements satisfied"
            if not rule["authoritative"] else
            "Configured documentation requirements satisfied",
            evidence=[{"evaluation": requirements}], rule=rule,
        ))
    else:
        findings.append(_finding(
            "documentation_requirements", "requires_review",
            "Configured documentation requirements were not satisfied; "
            "review missing documents and rule applicability",
            evidence=[{"evaluation": requirements}], rule=rule,
        ))

    # Even successful demonstration-rule checks do not prove policy compliance.
    if not rule["authoritative"]:
        findings.append(_finding(
            "rule_authority", "not_evaluated",
            "Insurer-approved rule source and version are not established; "
            "demonstration results cannot establish claim readiness",
            rule=rule,
        ))

    counts = dict(Counter(f["status"] for f in findings))
    for status in ALLOWED_STATUSES:
        counts.setdefault(status, 0)

    unresolved = counts["issue_detected"] + counts["requires_review"]
    report_status = "requires_review" if unresolved else "not_evaluated"
    # This pilot report never issues an approval/denial or readiness decision.
    return {
        "schema_version": "1.0",
        "report_id": str(uuid4()),
        "claim_id": str(claim_id).strip(),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "overall_status": report_status,
        "requires_review": bool(unresolved),
        "adjudication_status": "not_evaluated",
        "rule_context": rule,
        "summary": {
            "finding_count": len(findings),
            "status_counts": counts,
            "source_count": verification.get("sources_checked"),
        },
        "findings": findings,
    }
