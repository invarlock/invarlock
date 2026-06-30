#!/usr/bin/env bash
# release_review_policy.sh - Hardened evidence-pack release-review policy.

pack_apply_release_review_defaults() {
    PACK_REQUIRE_PASS="${PACK_REQUIRE_PASS:-1}"
    PACK_VERIFY_PROFILE="${PACK_VERIFY_PROFILE:-ci}"
    PACK_REPORT_ASSURANCE="${PACK_REPORT_ASSURANCE:-strict}"
    PACK_EVALUATE_ASSURANCE="${PACK_EVALUATE_ASSURANCE:-strict}"
    PACK_SIGN_MANIFEST="${PACK_SIGN_MANIFEST:-1}"
    PACK_REQUIRE_RUNTIME_MANIFESTS="${PACK_REQUIRE_RUNTIME_MANIFESTS:-1}"
    PACK_DEFER_REPORT_RENDERING="${PACK_DEFER_REPORT_RENDERING:-1}"
    PACK_RELEASE_REVIEW=1
    export PACK_REQUIRE_PASS PACK_VERIFY_PROFILE PACK_REPORT_ASSURANCE PACK_EVALUATE_ASSURANCE
    export PACK_SIGN_MANIFEST PACK_REQUIRE_RUNTIME_MANIFESTS
    export PACK_DEFER_REPORT_RENDERING PACK_RELEASE_REVIEW
}

pack_validate_release_review_settings() {
    if [[ "${PACK_RELEASE_REVIEW:-0}" != "1" ]]; then
        return 0
    fi
    if [[ "${PACK_REQUIRE_PASS:-0}" != "1" ]]; then
        echo "ERROR: release-review mode requires PACK_REQUIRE_PASS=1." >&2
        return 1
    fi
    if [[ "${PACK_SIGN_MANIFEST:-1}" == "0" ]]; then
        echo "ERROR: release-review mode requires PACK_SIGN_MANIFEST=1." >&2
        return 1
    fi
    if [[ "${PACK_REQUIRE_RUNTIME_MANIFESTS:-0}" != "1" ]]; then
        echo "ERROR: release-review mode requires PACK_REQUIRE_RUNTIME_MANIFESTS=1." >&2
        return 1
    fi
    if [[ -z "${PACK_VERIFY_PROFILE:-}" ]]; then
        echo "ERROR: release-review mode requires explicit PACK_VERIFY_PROFILE." >&2
        return 1
    fi
    if [[ "${PACK_VERIFY_PROFILE}" == "dev" ]]; then
        echo "ERROR: release-review mode rejects PACK_VERIFY_PROFILE=dev." >&2
        return 1
    fi
    if [[ "${PACK_REPORT_ASSURANCE:-}" != "strict" ]]; then
        echo "ERROR: release-review mode requires PACK_REPORT_ASSURANCE=strict." >&2
        return 1
    fi
    if [[ "${PACK_EVALUATE_ASSURANCE:-}" != "strict" ]]; then
        echo "ERROR: release-review mode requires PACK_EVALUATE_ASSURANCE=strict." >&2
        return 1
    fi
}
