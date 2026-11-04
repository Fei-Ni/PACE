#!/usr/bin/env bash
# Release hygiene gate. Run from anywhere:  bash tools/release_check.sh
#
# Checks (any failure -> exit 1):
#   1. no internal paths / names, no secret-shaped strings, no unresolved review notes, no debugger hooks,
#      no CJK text
#      (THIRD_PARTY_NOTICES.md and this script are exempt from the path / name check)
#   2. `bash -n` on every shell script
#   3. every launcher runs with an empty environment (only PATH) in --dry-run mode and prints an EFFECTIVE line
#   4. every relative link / image in *.md resolves
#   5. every argparse `default=` that looks like a repository-relative file path exists
#   6. python -m compileall -q pace examples tools tests
#   7. pytest -q tests
# Environment: PYTHON (interpreter with pytest + omegaconf or pyyaml; default python3).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHON="${PYTHON:-python3}"
SELF="tools/release_check.sh"
cd "${REPO_ROOT}"

FAILURES=0
fail() {
  echo "FAIL: $*" >&2
  FAILURES=$((FAILURES + 1))
}
section() {
  echo
  echo "== $* =="
}

# Tracked-or-not text files of the release (no .git, no caches).
list_files() {
  find . -type f \
    -not -path './.git/*' -not -path '*/__pycache__/*' -not -path './.pytest_cache/*' -not -path './.ruff_cache/*' \
    | sed 's|^\./||' | sort
}

section "1a. internal paths and names"
# Absolute cluster-style paths. Maintainers can add their own internal host / user / project names, one extended
# regex per line, in an untracked `.release_denylist` at the repository root (it is git-ignored).
PATTERN='/projects/|/home/|/lus/|/mnt/|/data/|/scratch'
if [[ -f .release_denylist ]]; then
  while IFS= read -r line; do
    [[ -z "${line}" || "${line}" == \#* ]] && continue
    PATTERN="${PATTERN}|${line}"
  done < .release_denylist
  echo "(+ $(grep -cvE '^(#|$)' .release_denylist) patterns from .release_denylist)"
fi
HITS="$(list_files | grep -v -x -e "${SELF}" -e 'THIRD_PARTY_NOTICES.md' -e '.release_denylist' | tr '\n' '\0' \
  | xargs -0 grep -nIE "${PATTERN}" 2>/dev/null || true)"
if [[ -n "${HITS}" ]]; then
  echo "${HITS}"
  fail "internal paths / names found"
else
  echo "ok"
fi

section "1b. secret-shaped strings"
SECRETS='hf_[A-Za-z0-9]{20,}|ghp_|github_pat_|(^|[^A-Za-z0-9])sk-[A-Za-z0-9_-]{16,}|AKIA[0-9A-Z]{16}|PRIVATE KEY'
HITS="$(list_files | grep -v -x -e "${SELF}" | tr '\n' '\0' | xargs -0 grep -nIE "${SECRETS}" 2>/dev/null || true)"
if [[ -n "${HITS}" ]]; then
  echo "${HITS}"
  fail "secret-shaped strings found"
else
  echo "ok"
fi

section "1b2. review notes"
# Drafting notes for the authors (<!-- REVIEW ... -->) must be resolved and removed before publishing.
# ALLOW_REVIEW_NOTES=1 only reports them (local review builds).
REVIEW_HITS="$(grep -rn --include='*.md' '<!-- REVIEW' . 2>/dev/null | grep -v '^./.git/' || true)"
if [[ -n "${REVIEW_HITS}" ]]; then
  echo "${REVIEW_HITS}"
  if [[ "${ALLOW_REVIEW_NOTES:-0}" == "1" ]]; then
    echo "(ALLOW_REVIEW_NOTES=1: reported only)"
  else
    fail "unresolved review notes in markdown files"
  fi
else
  echo ok
fi

section "1b3. generic environment variable names"
# HOST is set automatically by zsh and by many clusters; PORT by some platforms. Launchers must not read them.
GENERIC_HITS="$(grep -rnE '\$\{(HOST|PORT)(\}|:-)' --include='*.sh' . | grep -v "^./${SELF}:" || true)"
if [[ -n "${GENERIC_HITS}" ]]; then
  echo "${GENERIC_HITS}"
  fail "launchers read generic HOST / PORT variables (use POLICY_HOST / POLICY_PORT / BIND_HOST)"
else
  echo ok
fi

section "1c. debugger hooks"
HITS="$(list_files | grep -v -x -e "${SELF}" | tr '\n' '\0' | xargs -0 grep -nIE 'debugpy|ipdb|pdb\.set_trace|breakpoint\(\)' 2>/dev/null || true)"
if [[ -n "${HITS}" ]]; then
  echo "${HITS}"
  fail "debugger hooks found"
else
  echo "ok"
fi

section "1d. CJK text"
HITS="$(list_files | tr '\n' '\0' | xargs -0 "${PYTHON}" -c '
import re, sys
cjk = re.compile("[\u3000-\u303f\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff\uff00-\uffef]")
for path in sys.argv[1:]:
    try:
        text = open(path, encoding="utf-8").read()
    except (UnicodeDecodeError, OSError):
        continue
    for n, line in enumerate(text.splitlines(), 1):
        if cjk.search(line):
            print(f"{path}:{n}: {line.strip()[:100]}")
' || true)"
if [[ -n "${HITS}" ]]; then
  echo "${HITS}"
  fail "CJK text found"
else
  echo "ok"
fi

section "2. bash -n"
SH_FILES=()
while IFS= read -r f; do SH_FILES+=("${f}"); done < <(list_files | grep '\.sh$')
for f in "${SH_FILES[@]}"; do
  if ! bash -n "${f}"; then fail "bash -n ${f}"; fi
done
echo "checked ${#SH_FILES[@]} scripts"

section "3. launchers: empty environment + --dry-run"
SHIM_DIR="$(mktemp -d)"
trap 'rm -rf "${SHIM_DIR}"' EXIT
PYTHON_BIN="$(command -v "${PYTHON}")"
printf '#!/bin/sh\nexec "%s" "$@"\n' "${PYTHON_BIN}" > "${SHIM_DIR}/python"
chmod +x "${SHIM_DIR}/python"
DRY_PATH="${SHIM_DIR}:/usr/bin:/bin"
for f in "${SH_FILES[@]}"; do
  [[ "${f}" == "${SELF}" ]] && continue
  if ! OUT="$(cd "${SHIM_DIR}" && env -i PATH="${DRY_PATH}" bash -u "${REPO_ROOT}/${f}" --dry-run 2>&1)"; then
    echo "${OUT}"
    fail "${f} --dry-run failed in an empty environment"
    continue
  fi
  if ! grep -q '^EFFECTIVE: ' <<<"${OUT}"; then
    echo "${OUT}"
    fail "${f} --dry-run printed no EFFECTIVE line"
    continue
  fi
  echo "ok  ${f}: $(grep '^EFFECTIVE: ' <<<"${OUT}" | head -1 | cut -c1-110)..."
done

section "4. relative links in *.md"
if ! list_files | grep '\.md$' | tr '\n' '\0' | xargs -0 "${PYTHON}" -c '
import os, re, sys
link = re.compile(r"!?\[[^\]]*\]\(([^)\s]+)(?:\s+\"[^\"]*\")?\)|<img[^>]+src=\"([^\"]+)\"")
bad = 0
for path in sys.argv[1:]:
    text = open(path, encoding="utf-8").read()
    text = re.sub(r"```.*?```", "", text, flags=re.S)
    for m in link.finditer(text):
        target = m.group(1) or m.group(2)
        if re.match(r"^(https?:|mailto:|#)", target):
            continue
        target = target.split("#", 1)[0]
        if not os.path.exists(os.path.join(os.path.dirname(path), target)):
            print(f"{path}: broken link {target}")
            bad += 1
sys.exit(1 if bad else 0)
'; then
  fail "broken relative links"
else
  echo "ok"
fi

section "5. argparse default paths"
if ! list_files | grep '\.py$' | grep -v '^third_party/' | tr '\n' '\0' | xargs -0 "${PYTHON}" -c '
import os, re, sys
pat = re.compile(r"default\s*=\s*[\"\x27]([^\"\x27]+)[\"\x27]")
bad = 0
for path in sys.argv[1:]:
    for n, line in enumerate(open(path, encoding="utf-8"), 1):
        for value in pat.findall(line):
            looks_like_file = "/" in value and re.search(r"\.(ya?ml|json|py|sh|pt|pth|txt|md)$", value)
            if not looks_like_file or value.startswith(("/", "~", "http")):
                continue
            if not os.path.exists(value):
                print(f"{path}:{n}: default path does not exist: {value}")
                bad += 1
sys.exit(1 if bad else 0)
'; then
  fail "argparse default paths missing"
else
  echo "ok"
fi

section "6. compileall"
if "${PYTHON}" -m compileall -q pace examples tools tests >/dev/null; then echo "ok"; else fail "compileall"; fi
find pace examples tools tests -name '__pycache__' -type d -prune -exec rm -rf {} + 2>/dev/null || true

section "7. pytest"
if "${PYTHON}" -m pytest -q -p no:cacheprovider tests; then echo "ok"; else fail "pytest"; fi
find pace examples tools tests -name '__pycache__' -type d -prune -exec rm -rf {} + 2>/dev/null || true

echo
if [[ "${FAILURES}" -gt 0 ]]; then
  echo "release_check: ${FAILURES} check(s) FAILED"
  exit 1
fi
echo "release_check: all checks passed"
