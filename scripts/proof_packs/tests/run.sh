#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
HELPERS_SH="${SCRIPT_DIR}/helpers.sh"
MOCK_BIN_DIR="${SCRIPT_DIR}/mocks/bin"

FILTER_REGEX=""
DO_BRANCH_COVERAGE="false"
DO_LINE_COVERAGE="false"
COVERAGE_DIR=""
COVERAGE_RAW_HITS=""

usage() {
    cat <<'EOF'
Usage: scripts/proof_packs/tests/run.sh [--filter REGEX] [--coverage] [--line-coverage]

Options:
  --filter REGEX     Run only tests whose id matches REGEX (id: test_file::test_fn)
  --coverage         Run tests under xtrace and enforce 100% branch coverage for proof pack bash scripts
  --line-coverage    Run tests under xtrace and enforce 100% executable-line coverage for proof pack bash scripts
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --filter)
            FILTER_REGEX="${2:-}"
            shift 2
            ;;
        --coverage)
            DO_BRANCH_COVERAGE="true"
            shift
            ;;
        --line-coverage)
            DO_LINE_COVERAGE="true"
            shift
            ;;
        --help|-h)
            usage
            exit 0
            ;;
        *)
            echo "Unknown arg: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
done

REAL_PYTHON3="$(command -v python3 2>/dev/null || true)"

coverage_owner_hint() {
    local rel="$1"
    case "${rel}" in
        scripts/proof_packs/lib/config_generator.sh) echo "scripts/proof_packs/tests/test_config_generator.sh" ;;
        scripts/proof_packs/lib/task_serialization.sh) echo "scripts/proof_packs/tests/test_task_serialization.sh" ;;
        scripts/proof_packs/lib/queue_manager.sh) echo "scripts/proof_packs/tests/test_queue_manager.sh" ;;
        scripts/proof_packs/lib/scheduler.sh) echo "scripts/proof_packs/tests/test_scheduler.sh" ;;
        scripts/proof_packs/lib/gpu_worker.sh) echo "scripts/proof_packs/tests/test_gpu_worker.sh" ;;
        scripts/proof_packs/lib/fault_tolerance.sh) echo "scripts/proof_packs/tests/test_fault_tolerance.sh" ;;
        scripts/proof_packs/lib/task_functions.sh) echo "scripts/proof_packs/tests/test_task_functions.sh" ;;
        scripts/proof_packs/lib/model_creation.sh) echo "scripts/proof_packs/tests/test_model_creation.sh" ;;
        scripts/proof_packs/lib/runtime.sh) echo "scripts/proof_packs/tests/test_runtime.sh" ;;
        scripts/proof_packs/lib/validation_suite.sh) echo "scripts/proof_packs/tests/test_validation_suite.sh" ;;
        scripts/proof_packs/lib/setup_remote.sh) echo "scripts/proof_packs/tests/test_setup_remote.sh" ;;
        scripts/proof_packs/suites.sh) echo "scripts/proof_packs/tests/test_suites.sh" ;;
        scripts/proof_packs/run_suite.sh) echo "scripts/proof_packs/tests/test_run_suite.sh" ;;
        scripts/proof_packs/run_pack.sh) echo "scripts/proof_packs/tests/test_run_pack.sh" ;;
        scripts/proof_packs/verify_pack.sh) echo "scripts/proof_packs/tests/test_verify_pack.sh" ;;
        *) echo "scripts/proof_packs/tests/<add_test>.sh" ;;
    esac
}

coverage_target_files() {
    (
        cd "${ROOT_DIR}"
        find scripts/proof_packs -maxdepth 1 -type f -name '*.sh' -print
        find scripts/proof_packs/lib -maxdepth 1 -type f -name '*.sh' -print
    ) | sort
}

coverage_generate_inventory() {
    local out_file="$1"
    : >"${out_file}"

    local rel abs hint
    while IFS= read -r rel; do
        [[ -n "${rel}" ]] || continue
        abs="${ROOT_DIR}/${rel}"
        hint="$(coverage_owner_hint "${rel}")"

        awk -v rel="${rel}" -v hint="${hint}" '
function trim(s) {
    sub(/^[[:space:]]+/, "", s)
    sub(/[[:space:]]+$/, "", s)
    return s
}
function is_blank_or_comment(s) {
    s = trim(s)
    return (s == "" || s ~ /^#/)
}
function is_if_start(s) { return s ~ /^[[:space:]]*if[[:space:]]/ }
function is_elif_start(s) { return s ~ /^[[:space:]]*elif[[:space:]]/ }
function is_else(s) { return s ~ /^[[:space:]]*else([[:space:]]|$)/ }
function is_fi(s) { return s ~ /^[[:space:]]*fi([[:space:]]|$)/ }
function is_then_line(s) {
    return (s ~ /^[[:space:]]*then([[:space:]]|$)/ || s ~ /;[[:space:]]*then([[:space:]]*#.*)?$/)
}
function is_case_start(s) { return s ~ /^[[:space:]]*case[[:space:]]/ }
function is_esac(s) { return s ~ /^[[:space:]]*esac([[:space:]]|$)/ }
function is_case_terminator(s) { return s ~ /^[[:space:]]*(;;|;;&|;&)([[:space:]]|$)/ }
function is_heredoc_line(n) { return (n in heredoc && heredoc[n] == 1) }
function strip_inline_comment(s) {
    sub(/[[:space:]]+#.*/, "", s)
    return s
}
function first_error_return_or_exit_kind(s,    t, parts, n, k, word, arg) {
    t = strip_inline_comment(s)
    n = split(t, parts, /[;[:space:]]+/)
    for (k = 1; k <= n; k++) {
        word = parts[k]
        if (word == "return" || word == "exit") {
            arg = (k + 1 <= n ? parts[k + 1] : "")
            if (arg == "0") return ""
            return word
        }
    }
    return ""
}
function find_error_return_or_exit_in_or_block(start,    j, t, kind) {
    for (j = start; j <= NR; j++) {
        if (is_heredoc_line(j)) continue
        t = strip_inline_comment(lines[j])
        kind = first_error_return_or_exit_kind(t)
        if (kind != "") return j "@" kind
        if (j > start) {
            t = trim(t)
            if (t ~ /^\}[[:space:]]*(;|$)/) return ""
        }
    }
    return ""
}
function is_keyword_only(s) {
    s = trim(s)
    if (s ~ /^(then|do|done|fi|else|esac|in)$/) return 1
    if (s ~ /^(;;|;;&|;&)$/) return 1
    if (s ~ /^\}$/) return 1
    if (s ~ /^elif[[:space:]]/) return 1
    return 0
}
function find_exec_in_if_arm(start,    j, depth, line) {
    depth = 0
    for (j = start; j <= NR; j++) {
        if (is_heredoc_line(j)) continue
        line = lines[j]
        if (is_if_start(line)) depth++
        if (is_fi(line) && depth > 0) depth--

        if (depth == 0 && (is_elif_start(line) || is_else(line) || is_fi(line))) return 0

        if (is_blank_or_comment(line) || is_keyword_only(line)) continue
        return j
    }
    return 0
}
function is_case_pattern_line(s) {
    s = trim(s)
    if (s == "" || s ~ /^#/) return 0
    if (s ~ /^(case|esac|in)[[:space:]]/) return 0
    if (s ~ /^(;;|;;&|;&)/) return 0
    if (s ~ /\$\(/) return 0
    if (s ~ /[$:{=}]/) return 0
    return (s ~ /^[^)]*\)/)
}
function case_pattern_has_inline_cmd(line,    pos, rest) {
    pos = index(line, ")")
    if (pos == 0) return 0
    rest = substr(line, pos + 1)
    sub(/[[:space:]]+#.*/, "", rest)
    rest = trim(rest)
    if (rest == "" || rest ~ /^(;;|;;&|;&)$/) return 0
    return 1
}
function find_exec_in_case_arm(start, base_depth,    j, depth, line) {
    depth = base_depth
    for (j = start; j <= NR; j++) {
        if (is_heredoc_line(j)) continue
        line = lines[j]
        if (is_case_start(line)) depth++
        if (is_esac(line)) {
            if (depth > base_depth) depth--
            else return 0
        }

        if (depth == base_depth) {
            if (is_case_terminator(line)) return 0
            if (is_case_pattern_line(line)) return 0
        }

        if (is_blank_or_comment(line) || is_keyword_only(line) || is_case_terminator(line)) continue
        return j
    }
    return 0
}
BEGIN {
    OFS = "\t"
    hd_active = 0
    hd_pending = 0
    hd_delim = ""
    hd_strip_tabs = 0
}
{
    line = $0
    lines[NR] = line

    # Enter heredoc body starting on the line after the opener.
    if (hd_pending) {
        hd_active = 1
        hd_pending = 0
    }

    heredoc[NR] = (hd_active ? 1 : 0)

    if (hd_active) {
        check = line
        if (hd_strip_tabs) sub(/^\t+/, "", check)
        if (check == hd_delim) {
            hd_active = 0
            hd_delim = ""
            hd_strip_tabs = 0
        }
        next
    }

    # Detect heredoc openers (<<EOF, <<'EOF', <<-EOF, ...). Ignore here-strings (<<<).
    if (line ~ /^[[:space:]]*#/) next
    if (match(line, /<<-?[[:space:]]*[^[:space:]]+/)) {
        token = substr(line, RSTART, RLENGTH)
        if (substr(token, 1, 3) == "<<<") next
        if (token ~ /^<</) {
            hd_strip_tabs = (token ~ /^<<-/ ? 1 : 0)
            gsub(/^<<-?[[:space:]]*/, "", token)
            sq = sprintf("%c", 39)
            dq = "\""
            if (token != "" && (substr(token, 1, 1) == sq || substr(token, 1, 1) == dq)) {
                token = substr(token, 2)
            }
            if (token != "" && (substr(token, length(token), 1) == sq || substr(token, length(token), 1) == dq)) {
                token = substr(token, 1, length(token) - 1)
            }
            if (token != "") {
                hd_delim = token
                hd_pending = 1
            }
        }
    }
}
END {
    case_depth = 0
    pending_cond_line = 0
    pending_cond_kind = ""

    for (i = 1; i <= NR; i++) {
        if (is_heredoc_line(i)) continue
        line = lines[i]

        if (is_if_start(line)) { pending_cond_line = i; pending_cond_kind = "if" }
        else if (is_elif_start(line)) { pending_cond_line = i; pending_cond_kind = "elif" }

        if (is_then_line(line)) {
            id = rel ":" pending_cond_kind ":" pending_cond_line ":then"
            rep1 = find_exec_in_if_arm(i + 1)
            if (rep1 == 0) {
                printf("ERROR: %s has an empty then-arm near line %d\\n", rel, i) > "/dev/stderr"
                exit 3
            }
            rep2 = find_exec_in_if_arm(rep1 + 1)
            reps = rep1
            if (rep2 != 0 && rep2 != rep1) reps = reps "," rep2
            print id, rel, reps, hint
        }
        if (is_else(line)) {
            id = rel ":else:" i
            rep1 = find_exec_in_if_arm(i + 1)
            if (rep1 == 0) {
                printf("ERROR: %s has an empty else-arm near line %d\\n", rel, i) > "/dev/stderr"
                exit 3
            }
            rep2 = find_exec_in_if_arm(rep1 + 1)
            reps = rep1
            if (rep2 != 0 && rep2 != rep1) reps = reps "," rep2
            print id, rel, reps, hint
        }

        if (is_case_start(line)) case_depth++
        if (case_depth > 0 && is_case_pattern_line(line)) {
            id = rel ":case:" i
            if (case_pattern_has_inline_cmd(line)) rep1 = i
            else rep1 = find_exec_in_case_arm(i + 1, case_depth)
            if (rep1 == 0) {
                printf("ERROR: %s has an empty case-arm near line %d\\n", rel, i) > "/dev/stderr"
                exit 3
            }
            rep2 = find_exec_in_case_arm(rep1 + 1, case_depth)
            reps = rep1
            if (rep2 != 0 && rep2 != rep1) reps = reps "," rep2
            print id, rel, reps, hint
        }
        if (is_esac(line) && case_depth > 0) case_depth--

        # Explicit short-circuit error paths: "cmd || return/exit N" (N != 0).
        tmp = strip_inline_comment(line)
        if (tmp ~ /\|\|[[:space:]]*(return|exit)([[:space:]]|$)/) {
            pos = index(tmp, "||")
            after = trim(substr(tmp, pos + 2))
            split(after, parts, /[[:space:];]+/)
            kw = parts[1]
            arg = parts[2]
            if ((kw == "return" || kw == "exit") && arg != "0") {
                id = rel ":or_" kw ":" i
                print id, rel, i "@" kw, hint
            }
        }

        # Explicit error-path blocks: "... || { ... return/exit N ... }" (N != 0).
        if (tmp ~ /\|\|[[:space:]]*\{([[:space:]]|$)/) {
            rep = find_error_return_or_exit_in_or_block(i)
            if (rep != "") {
                id = rel ":or_block:" i
                print id, rel, rep, hint
            }
        }
    }
}
        ' "${abs}" >>"${out_file}"
    done < <(coverage_target_files)
}

coverage_append_trace_hits() {
    local trace_file="$1"

    awk -v root="${ROOT_DIR}/" '
function ltrim(s) {
    sub(/^[[:space:]]+/, "", s)
    return s
}
$0 ~ /^_+XTRACE__:/ {
    # __XTRACE__:/abs/path:123: command (bash may prefix extra leading "_" chars)
    sub(/^_+XTRACE__:/, "", $0)

    # Split on the first two ":" delimiters only (command text may contain ":").
    pos1 = index($0, ":")
    if (pos1 == 0) next
    rest = substr($0, pos1 + 1)
    pos2 = index(rest, ":")
    if (pos2 == 0) next

    file = substr($0, 1, pos1 - 1)
    line = substr(rest, 1, pos2 - 1)
    cmd = substr(rest, pos2 + 1)

    if (file ~ /^\// && index(file, root) == 1) file = substr(file, length(root) + 1)
    sub(/^\.\//, "", file)
    if (index(file, "scripts/") != 1) next

    gsub(/[^0-9]/, "", line)
    if (line == "") next

    cmd = ltrim(cmd)
    token = cmd
    sub(/[[:space:]].*$/, "", token)
    if (token == "") token = "-"
    print file "\t" line "\t" token
}
    ' "${trace_file}" >>"${COVERAGE_RAW_HITS}"
}

coverage_append_trace_hits_from_logs() {
    local dir="$1"
    local f
    while IFS= read -r f; do
        [[ -f "${f}" ]] || continue
        grep -qE '^_+XTRACE__:' "${f}" 2>/dev/null || continue
        coverage_append_trace_hits "${f}"
    done < <(find "${dir}" -type f -name '*.log' -print 2>/dev/null || true)
}

coverage_ensure_executed() {
    local executed_file="${COVERAGE_DIR}/executed.tsv"
    if [[ -z "${COVERAGE_DIR}" || -z "${COVERAGE_RAW_HITS}" ]]; then
        echo "ERROR: coverage not initialized" >&2
        return 1
    fi
    if [[ ! -f "${executed_file}" ]]; then
        sort -u "${COVERAGE_RAW_HITS}" >"${executed_file}"
    fi
}

coverage_check() {
    local inventory_file="${COVERAGE_DIR}/branch_inventory.tsv"
    local executed_file="${COVERAGE_DIR}/executed.tsv"
    local missing_file="${COVERAGE_DIR}/missing.tsv"

    coverage_generate_inventory "${inventory_file}"
    coverage_ensure_executed

    if ! awk -v executed="${executed_file}" '
BEGIN {
    FS = "\t"
    while ((getline < executed) > 0) {
        file = $1
        line = $2
        cmd = $3
        key = file ":" line
        hits_line[key] = 1
        if (cmd != "") {
            hits_cmd[key ":" cmd] = 1
        }
    }
}
{
    id = $1
    file = $2
    reps = $3
    hint = $4
    n = split(reps, parts, /,/)
    ok = 0
    for (i = 1; i <= n; i++) {
        spec = parts[i]
        gsub(/^[[:space:]]+|[[:space:]]+$/, "", spec)
        if (spec == "") continue

        split(spec, segs, /@/)
        line = segs[1]
        gsub(/[^0-9]/, "", line)
        if (line == "") continue

        key = file ":" line
        if (length(segs) >= 2 && segs[2] != "") {
            cmd = segs[2]
            if ((key ":" cmd) in hits_cmd) {
                ok = 1
                break
            }
        } else {
            if (key in hits_line) {
                ok = 1
                break
            }
        }
    }
    if (!ok) {
        print id "\t" file "\t" reps "\t" hint
        missing++
    }
}
END {
    if (missing > 0) exit 7
}
    ' "${inventory_file}" >"${missing_file}"
    then
        local missing_count
        missing_count="$(wc -l <"${missing_file}" | tr -d ' ')"
        echo "COVERAGE FAIL: missing ${missing_count} branch arm(s)" >&2
        awk -F'\t' 'NR<=80 {printf("  - %s (%s:%s) -> %s\n", $1, $2, $3, $4)}' "${missing_file}" >&2
        if [[ ${missing_count} -gt 80 ]]; then
            echo "  ... (see ${missing_file})" >&2
        fi
        echo "coverage artifacts: ${COVERAGE_DIR}" >&2
        return 1
    fi
    return 0
}

line_generate_inventory() {
    local out_file="$1"
    : >"${out_file}"

    local rel abs hint
    while IFS= read -r rel; do
        [[ -n "${rel}" ]] || continue
        abs="${ROOT_DIR}/${rel}"
        hint="$(coverage_owner_hint "${rel}")"

        awk -v rel="${rel}" -v hint="${hint}" '
function trim(s) {
    sub(/^[[:space:]]+/, "", s)
    sub(/[[:space:]]+$/, "", s)
    return s
}
function is_keyword_only(s) {
    s = trim(s)
    if (s ~ /^(then|do|done|fi|else|esac|in)$/) return 1
    # Compound closer lines with redirections / process substitutions are not traced by bash xtrace.
    # Examples: "} 200>file", "done < <(...)".
    if (s ~ /^\}[[:space:]]*[0-9]*[<>]/) return 1
    if (s ~ /^\)[[:space:]]*[0-9]*[<>]/) return 1
    if (s ~ /^\($/) return 1
    if (s ~ /^\)[[:space:]]*&$/) return 1
    if (s ~ /^done[[:space:]]+<[^<]/ || s ~ /^done[[:space:]]+<+[[:space:]]*<\(/) return 1
    if (s ~ /^(;;|;;&|;&)$/) return 1
    if (s ~ /^\}$/) return 1
    if (s ~ /^\{$/) return 1
    if (s ~ /^\)$/) return 1
    return 0
}
function is_case_pattern_line(s) {
    s = trim(s)
    if (s == "" || s ~ /^#/) return 0
    if (s ~ /^(case|esac|in)[[:space:]]/) return 0
    if (s ~ /^(;;|;;&|;&)/) return 0
    if (s ~ /\$\(/) return 0
    if (s ~ /[$:{=}]/) return 0
    return (s ~ /^[^)]*\)/)
}
function case_pattern_has_inline_cmd(line,    pos, rest) {
    pos = index(line, ")")
    if (pos == 0) return 0
    rest = substr(line, pos + 1)
    sub(/[[:space:]]+#.*/, "", rest)
    rest = trim(rest)
    if (rest == "" || rest ~ /^(;;|;;&|;&)$/) return 0
    return 1
}
function is_function_def_line(s,    t) {
    t = trim(s)
    if (t ~ /^function[[:space:]]+[A-Za-z_][A-Za-z0-9_]*([[:space:]]*\(\))?[[:space:]]*\{/) return 1
    if (t ~ /^function[[:space:]]+[A-Za-z_][A-Za-z0-9_]*([[:space:]]*\(\))?[[:space:]]*$/) return 1
    if (t ~ /^[A-Za-z_][A-Za-z0-9_]*[[:space:]]*\(\)[[:space:]]*\{/) return 1
    if (t ~ /^[A-Za-z_][A-Za-z0-9_]*[[:space:]]*\(\)[[:space:]]*$/) return 1
    return 0
}
function ends_with_backslash(s) {
    s = trim(s)
    return (s ~ /\\$/)
}
	function is_array_start(s,    tmp) {
	    tmp = trim(s)
	    sub(/[[:space:]]+#.*/, "", tmp)
	    tmp = trim(tmp)
	    if (tmp !~ /(^|[[:space:]])[A-Za-z_][A-Za-z0-9_]*(\+)?=[[:space:]]*\(/) return 0
	    # Single-line array literal (closes on the same line) is traced on this line.
	    if (tmp ~ /\)[[:space:]]*$/) return 0
	    # Multi-line array literal: bash xtrace attribution differs depending on whether the
	    # assignment is wrapped in `local`/`declare` (which traces the opener line).
	    if (tmp ~ /^(local|declare|typeset)[[:space:]]/) return 1
	    return 2
	}
	function find_multiline_quote_start(s,    i, ch, prev, state, start, kind) {
	    state = 0
	    start = 0
	    kind = ""
	    for (i = 1; i <= length(s); i++) {
	        ch = substr(s, i, 1)

	        if (state == 1) {
	            if (ch == "'\''") { state = 0; start = 0; kind = "" }
	            continue
	        }
	        if (state == 2) {
	            if (ch == "\\") { i++; continue }
	            if (ch == "\"") { state = 0; start = 0; kind = "" }
	            continue
	        }

	        if (ch == "\\") { i++; continue }
	        if (ch == "'\''") { state = 1; start = i; kind = "sq"; continue }
	        if (ch == "\"") { state = 2; start = i; kind = "dq"; continue }

	        if (ch == "#") {
	            if (i == 1) break
	            prev = substr(s, i - 1, 1)
	            if (prev ~ /[[:space:]]/) break
	        }
	    }
	    if (state != 0 && start > 0) return start "\t" kind
	    return ""
	}
	function should_shift_quote_to_close(line, start,    prev_ch) {
	    if (start <= 1) return 0
	    prev_ch = substr(line, start - 1, 1)
	    # Quotes that begin mid-word (e.g., var='a\nb') are attributed to the closing line.
	    return (prev_ch !~ /[[:space:]]/)
	}
	function update_quote_state(s,    i, ch, prev) {
	    for (i = 1; i <= length(s); i++) {
	        ch = substr(s, i, 1)

        if (sq) {
            if (ch == "'\''") sq = 0
            continue
        }
        if (dq) {
            if (ch == "\\") { i++; continue }
            if (ch == "\"") { dq = 0; continue }
            continue
        }

        if (ch == "\\") { i++; continue }
        if (ch == "'\''") { sq = 1; continue }
        if (ch == "\"") { dq = 1; continue }

        # Track multi-line command substitutions across lines; xtrace attribution depends on
        # whether the opener is wrapped in `local`/`declare` (see END block logic).
        if (ch == "$" && i < length(s) && substr(s, i + 1, 1) == "(") {
            # Ignore arithmetic expansion "$((".
            if (i + 2 <= length(s) && substr(s, i + 2, 1) == "(") continue
            cmdsub_depth++
            i++
            continue
        }
        if (ch == ")" && cmdsub_depth > 0) {
            cmdsub_depth--
            continue
        }

        if (ch == "#") {
            if (i == 1) break
            prev = substr(s, i - 1, 1)
            if (prev ~ /[[:space:]]/) break
        }
    }
}
	BEGIN {
	    OFS = "\t"
	    sq = 0
	    dq = 0
	    cmdsub_depth = 0
	    cmdsub_trace_mode = 0
	    hd_active = 0
	    hd_pending = 0
	    hd_delim = ""
	    hd_strip_tabs = 0
	}
{
    line = $0
    lines[NR] = line

    if (hd_pending) {
        hd_active = 1
        hd_pending = 0
    }

    heredoc[NR] = (hd_active ? 1 : 0)

    if (hd_active) {
        check = line
        if (hd_strip_tabs) sub(/^\t+/, "", check)
        if (check == hd_delim) {
            hd_active = 0
            hd_delim = ""
            hd_strip_tabs = 0
        }
        next
    }

    if (line ~ /^[[:space:]]*#/) next
    if (match(line, /<<-?[[:space:]]*[^[:space:]]+/)) {
        token = substr(line, RSTART, RLENGTH)
        if (substr(token, 1, 3) == "<<<") next
        if (token ~ /^<</) {
            hd_strip_tabs = (token ~ /^<<-/ ? 1 : 0)
            gsub(/^<<-?[[:space:]]*/, "", token)
            sqc = sprintf("%c", 39)
            dqv = "\""
            if (token != "" && (substr(token, 1, 1) == sqc || substr(token, 1, 1) == dqv)) {
                token = substr(token, 2)
            }
            if (token != "" && (substr(token, length(token), 1) == sqc || substr(token, length(token), 1) == dqv)) {
                token = substr(token, 1, length(token) - 1)
            }
            if (token != "") {
                hd_delim = token
                hd_pending = 1
            }
        }
    }
	}
		END {
		    cont_mode = 0
		    quote_shift = 0
		    in_array = 0
		    array_trace_close = 0
		    in_proc_sub = 0

	    for (i = 1; i <= NR; i++) {
	        if (heredoc[i]) continue
	        line = lines[i]

	        t = trim(line)

        # Multi-line process substitutions ("< <(" ... ")" on their own lines) do not produce
        # stable xtrace line attribution for the inner commands. Exclude their bodies from
        # executable-line inventory.
        if (in_proc_sub) {
            update_quote_state(line)
            if (!(sq || dq) && cmdsub_depth == 0 && t ~ /^\)[[:space:]]*(#.*)?$/) {
                in_proc_sub = 0
            }
            continue
        }
        if (t ~ /<[[:space:]]*<[[:space:]]*\($/) {
            in_proc_sub = 1
            update_quote_state(line)
            continue
        }

	        if (in_array) {
	            if (t == "" || t ~ /^#/) {
	                update_quote_state(line)
	                continue
	            }
	            t2 = t
	            sub(/[[:space:]]+#.*/, "", t2)
	            t2 = trim(t2)
	            if (t2 ~ /\)[[:space:]]*$/) {
	                if (array_trace_close) {
	                    print rel ":" i, rel, i, hint
	                }
	                in_array = 0
	                array_trace_close = 0
	            }
	            update_quote_state(line)
	            continue
	        }

	        prev_sq = sq
	        prev_dq = dq
	        prev_cmdsub = cmdsub_depth
	        update_quote_state(line)

	        # Multi-line "$(" ... ")" command substitutions: inventory either the opener line
	        # (for `local`/`declare` assignments) or the closing line (default).
	        if (prev_cmdsub == 0 && cmdsub_depth > 0) {
	            cmdsub_trace_mode = 2
	            t_cmd = trim(line)
	            sub(/[[:space:]]+#.*/, "", t_cmd)
	            t_cmd = trim(t_cmd)
		            if (t_cmd ~ /^(local|declare|typeset)[[:space:]]/ && t_cmd ~ /=[[:space:]]*[$][(]/) {
	                cmdsub_trace_mode = 1
	                if (t != "" && t !~ /^#/ && !is_keyword_only(line) && !(is_case_pattern_line(line) && !case_pattern_has_inline_cmd(line)) && !is_function_def_line(line)) {
	                    print rel ":" i, rel, i, hint
	                }
	            }
	            cont_mode = 0
	            continue
	        }
	        if (prev_cmdsub > 0 && cmdsub_depth == 0) {
	            if (cmdsub_trace_mode == 2) {
	                print rel ":" i, rel, i, hint
	            }
	            cmdsub_trace_mode = 0
	            cont_mode = 0
	            quote_shift = 0
	            continue
	        }
	        if (prev_cmdsub > 0 || cmdsub_depth > 0) {
	            cont_mode = 0
	            continue
	        }

	        # Multi-line quoted strings: bash xtrace sometimes attributes to the opener (when
	        # the quote is not the first arg), and sometimes to the closing line (first-arg
	        # quotes and mid-word quotes like var='a\nb'). Track which behavior to expect.
	        if (prev_sq || prev_dq) {
	            if (!(sq || dq)) {
	                if (quote_shift) {
	                    if (t != "" && t !~ /^#/ && !is_keyword_only(line) && !(is_case_pattern_line(line) && !case_pattern_has_inline_cmd(line)) && !is_function_def_line(line)) {
	                        print rel ":" i, rel, i, hint
	                    }
	                }
	                quote_shift = 0
	            }
	            cont_mode = 0
	            continue
	        }
	        if (!(prev_sq || prev_dq) && (sq || dq)) {
	            info = find_multiline_quote_start(line)
	            split(info, parts, /\t/)
	            qpos = parts[1] + 0
	            quote_shift = should_shift_quote_to_close(line, qpos)
	            if (!quote_shift) {
	                if (t != "" && t !~ /^#/ && !is_keyword_only(line) && !(is_case_pattern_line(line) && !case_pattern_has_inline_cmd(line)) && !is_function_def_line(line)) {
	                    print rel ":" i, rel, i, hint
	                }
	            }
	            cont_mode = 0
	            continue
	        }

	        # Backslash continuations: bash xtrace attributes to the opener line.
	        if (cont_mode != 0) {
	            if (!ends_with_backslash(line)) cont_mode = 0
	            continue
	        }

	        if (t == "" || t ~ /^#/) continue
	        if (is_keyword_only(line)) continue
	        if (is_case_pattern_line(line) && !case_pattern_has_inline_cmd(line)) continue
	        if (is_function_def_line(line)) continue

	        arr_mode = is_array_start(line)
	        if (arr_mode != 0) {
	            if (arr_mode == 1) {
	                print rel ":" i, rel, i, hint
	                array_trace_close = 0
	            } else {
	                array_trace_close = 1
	            }
	            in_array = 1
	            continue
	        }

	        if (ends_with_backslash(line)) {
	            print rel ":" i, rel, i, hint
	            cont_mode = 1
	            continue
	        }

	        print rel ":" i, rel, i, hint
		    }
		}
		        ' "${abs}" >>"${out_file}"
		    done < <(coverage_target_files)
}

line_coverage_check() {
    local inventory_file="${COVERAGE_DIR}/line_inventory.tsv"
    local executed_file="${COVERAGE_DIR}/executed.tsv"
    local missing_file="${COVERAGE_DIR}/missing_lines.tsv"

    line_generate_inventory "${inventory_file}"
    coverage_ensure_executed

    if ! awk -v executed="${executed_file}" '
BEGIN {
    FS = "\t"
    while ((getline < executed) > 0) {
        key = $1 ":" $2
        hits[key] = 1
    }
}
{
    id = $1
    file = $2
    line = $3
    hint = $4
    key = file ":" line
    if (!(key in hits)) {
        print id "\t" file "\t" line "\t" hint
        missing++
    }
}
END {
    if (missing > 0) exit 7
}
    ' "${inventory_file}" >"${missing_file}"
    then
        local missing_count
        missing_count="$(wc -l <"${missing_file}" | tr -d ' ')"
        echo "LINE COVERAGE FAIL: missing ${missing_count} line(s)" >&2
        awk -F'\t' 'NR<=80 {printf("  - %s (%s:%s) -> %s\n", $1, $2, $3, $4)}' "${missing_file}" >&2
        if [[ ${missing_count} -gt 80 ]]; then
            echo "  ... (see ${missing_file})" >&2
        fi
        echo "coverage artifacts: ${COVERAGE_DIR}" >&2
        return 1
    fi
    return 0
}

list_test_files() {
    find "${SCRIPT_DIR}" -maxdepth 1 -type f -name 'test_*.sh' -print | sort
}

list_tests_in_file() {
    local file="$1"
    bash -c '
set -euo pipefail
export TEST_ROOT="$1"
export TEST_TMPDIR="$(mktemp -d 2>/dev/null || mktemp -d -t invarlock_bash_tests.XXXXXXXX)"
trap "rm -rf \"${TEST_TMPDIR}\"" EXIT
source "$2"
source "$3"
declare -F | awk "{print \$3}" | grep "^test_" || true
' -- "${ROOT_DIR}" "${HELPERS_SH}" "${file}"
}

run_one_test() {
    local file="$1"
    local fn="$2"
    local id
    id="$(basename "${file}")::${fn}"

    if [[ -n "${FILTER_REGEX}" ]]; then
        if ! [[ "${id}" =~ ${FILTER_REGEX} ]]; then
            return 0
        fi
    fi

    local tmp_dir
    tmp_dir="$(mktemp -d 2>/dev/null || mktemp -d -t invarlock_bash_tests.XXXXXXXX)"

    local out_file err_file trace_file
    out_file="$(mktemp "${tmp_dir}/stdout.XXXXXX")"
    err_file="$(mktemp "${tmp_dir}/stderr.XXXXXX")"
    trace_file="${tmp_dir}/xtrace.log"
    local rc=0

	if [[ "${DO_BRANCH_COVERAGE}" == "true" || "${DO_LINE_COVERAGE}" == "true" ]]; then
	    bash -c '
	set -euo pipefail
	cd "$1"
export TEST_ROOT="."
export TEST_TMPDIR="$2"
export TEST_REAL_PYTHON3="$3"
export PATH="$4:$PATH"
source "$5"
source "$6"
# Keep xtrace prefixes short to avoid bash 3.2 truncation on long absolute paths.
# Note: bash 3.2 truncates long PS4 expansions; we rely on shorter prod script paths
# and ignore malformed trace lines from very long absolute paths (e.g., temp dirs).
export PS4="__XTRACE__:\${BASH_SOURCE[0]:-}:\${LINENO}: "
	set -x
	"$7"
	        ' -- "${ROOT_DIR}" "${tmp_dir}" "${REAL_PYTHON3}" "${MOCK_BIN_DIR}" "${HELPERS_SH}" "${file}" "${fn}" >"${out_file}" 2>"${err_file}" </dev/null
	    rc=$?
	    if [[ ${rc} -eq 0 ]]; then
            local safe_id trace_copy
            safe_id="$(echo "${id}" | tr -c 'A-Za-z0-9._-' '_')"
            trace_copy="${COVERAGE_DIR}/trace_${safe_id}.log"
            grep -E '^_+XTRACE__:' "${err_file}" >"${trace_copy}" 2>/dev/null || true
            coverage_append_trace_hits "${err_file}"
            coverage_append_trace_hits_from_logs "${tmp_dir}"
            rm -rf "${tmp_dir}"
            echo "ok  ${id}"
            return 0
        fi
	else
	    bash -c '
	set -euo pipefail
	cd "$1"
export TEST_ROOT="."
export TEST_TMPDIR="$2"
export TEST_REAL_PYTHON3="$3"
export PATH="$4:$PATH"
	source "$5"
	source "$6"
	"$7"
	        ' -- "${ROOT_DIR}" "${tmp_dir}" "${REAL_PYTHON3}" "${MOCK_BIN_DIR}" "${HELPERS_SH}" "${file}" "${fn}" >"${out_file}" 2>"${err_file}" </dev/null
	    rc=$?
	    if [[ ${rc} -eq 0 ]]; then
            rm -rf "${tmp_dir}"
            echo "ok  ${id}"
            return 0
        fi
    fi

    echo "not ok  ${id} (rc=${rc})" >&2
    sed 's/^/  | /' "${out_file}" >&2 || true
    sed 's/^/  | /' "${err_file}" >&2 || true
    echo "tmpdir: ${tmp_dir}" >&2
    return 1
}

main() {
    if [[ "${DO_BRANCH_COVERAGE}" == "true" || "${DO_LINE_COVERAGE}" == "true" ]]; then
        COVERAGE_DIR="${SCRIPT_DIR}/.coverage"
        rm -rf "${COVERAGE_DIR}"
        mkdir -p "${COVERAGE_DIR}"
        COVERAGE_RAW_HITS="${COVERAGE_DIR}/executed.raw.tsv"
        : >"${COVERAGE_RAW_HITS}"
    fi

    local failures=0
    local file
    while IFS= read -r file; do
        local fn
        while IFS= read -r fn; do
            [[ -n "${fn}" ]] || continue
            run_one_test "${file}" "${fn}" || failures=$((failures + 1))
        done < <(list_tests_in_file "${file}")
    done < <(list_test_files)

    if [[ ${failures} -gt 0 ]]; then
        echo "${failures} test(s) failed" >&2
        exit 1
    fi

    if [[ "${DO_BRANCH_COVERAGE}" == "true" ]]; then
        coverage_check
    fi
    if [[ "${DO_LINE_COVERAGE}" == "true" ]]; then
        line_coverage_check
    fi
}

main
