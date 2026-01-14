#!/usr/bin/env bash

test_run_pack_build_pack_collects_artifacts() {
    mock_reset

    source ./scripts/proof_packs/run_pack.sh

    local run_dir="${TEST_TMPDIR}/run"
    mkdir -p "${run_dir}/reports" "${run_dir}/analysis" "${run_dir}/state"
    mkdir -p "${run_dir}/modelA/certificates/edit/run_1"

    echo "verdict" > "${run_dir}/reports/final_verdict.txt"
    echo "{}" > "${run_dir}/reports/final_verdict.json"
    echo "model,score" > "${run_dir}/analysis/eval_results.csv"
    echo "model,metric" > "${run_dir}/analysis/guard_sensitivity_matrix.csv"
    echo '{"model_list": ["org/model"], "models": {"org/model": {"revision": "abc"}}}' > "${run_dir}/state/model_revisions.json"
    echo "{}" > "${run_dir}/modelA/certificates/edit/run_1/evaluation.cert.json"

    local bin_dir="${TEST_TMPDIR}/bin"
    mkdir -p "${bin_dir}"
    cat > "${bin_dir}/invarlock" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
cmd="${1:-}"
shift || true
case "${cmd}" in
    report)
        sub="${1:-}"
        if [[ "${sub}" == "html" ]]; then
            out=""
            while [[ $# -gt 0 ]]; do
                case "$1" in
                    --output|-o)
                        out="$2"
                        shift 2
                        ;;
                    *)
                        shift
                        ;;
                esac
            done
            mkdir -p "$(dirname "${out}")"
            printf '<html>ok</html>\n' > "${out}"
            exit 0
        fi
        ;;
    verify)
        echo '{"ok": true}'
        exit 0
        ;;
esac
echo '{}'
EOF
    chmod +x "${bin_dir}/invarlock"
    export PATH="${bin_dir}:${PATH}"

    PACK_GPG_SIGN=0

    local pack_dir="${TEST_TMPDIR}/pack"
    pack_build_pack "${run_dir}" "${pack_dir}"

    assert_file_exists "${pack_dir}/results/final_verdict.txt" "verdict copied"
    assert_file_exists "${pack_dir}/results/eval_results.csv" "eval results copied"
    assert_file_exists "${pack_dir}/certs/modelA/edit/run_1/evaluation.cert.json" "cert copied"
    assert_file_exists "${pack_dir}/certs/modelA/edit/run_1/verify.json" "verify output captured"
    assert_file_exists "${pack_dir}/manifest.json" "manifest written"
    assert_file_exists "${pack_dir}/checksums.sha256" "checksums written"
    assert_file_exists "${pack_dir}/certs/modelA/edit/run_1/evaluation.html" "html rendered"
    assert_file_exists "${pack_dir}/README.md" "readme written"
}
