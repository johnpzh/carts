#!/usr/bin/env bash
# Suggest CARTS skills for source prompts before the model starts work.
# Exit 0 always: this is advisory, not a blocker.

set -u

if ! command -v jq >/dev/null 2>&1; then
  exit 0
fi

INPUT=$(cat)
PROMPT=$(printf '%s' "$INPUT" | jq -r '.prompt // .user_prompt // .message // empty')
CWD=$(printf '%s' "$INPUT" | jq -r '.cwd // .current_working_directory // empty')
HAYSTACK=$(printf '%s\n%s\n' "$PROMPT" "$CWD" | tr '[:upper:]' '[:lower:]')

case "$HAYSTACK" in
  *lib/carts*|*include/carts*|*tools/compile*) ;;
  *) exit 0 ;;
esac

if [ -n "${CARTS_PROJECT_DIR:-}" ]; then
  PROJECT_DIR=$CARTS_PROJECT_DIR
elif [ -n "${CLAUDE_PROJECT_DIR:-}" ]; then
  PROJECT_DIR=$CLAUDE_PROJECT_DIR
else
  PROJECT_DIR=$(git rev-parse --show-toplevel 2>/dev/null || pwd)
fi
PLUGIN_ROOT=${CARTS_PLUGIN_ROOT:-${CLAUDE_PLUGIN_ROOT:-$PROJECT_DIR/carts-plugin}}
SKILL_ROOT="$PLUGIN_ROOT/skills"

declare -a SKILLS=()
declare -a REASONS=()

skill_path() {
  case "$1" in
    check-utils) printf '%s/check-utils/SKILL.md' "$SKILL_ROOT" ;;
    carts-dialect-map) printf '%s/carts-dialect-map/SKILL.md' "$SKILL_ROOT" ;;
    carts-attr-consolidation)
      printf '%s/carts-attr-consolidation/SKILL.md' "$SKILL_ROOT" ;;
    carts-find-utils) printf '%s/carts-find-utils/SKILL.md' "$SKILL_ROOT" ;;
    refactor-utils) printf '%s/refactor-utils/SKILL.md' "$SKILL_ROOT" ;;
    carts-include-tier) printf '%s/carts-include-tier/SKILL.md' "$SKILL_ROOT" ;;
    pass-dev) printf '%s/pass-dev/SKILL.md' "$SKILL_ROOT" ;;
    carts-pipeline-map) printf '%s/carts-pipeline-map/SKILL.md' "$SKILL_ROOT" ;;
    test) printf '%s/test/SKILL.md' "$SKILL_ROOT" ;;
    create-test) printf '%s/create-test/SKILL.md' "$SKILL_ROOT" ;;
    debug) printf '%s/debug/SKILL.md' "$SKILL_ROOT" ;;
    miscompile-triage) printf '%s/miscompile-triage/SKILL.md' "$SKILL_ROOT" ;;
    reproducer) printf '%s/reproducer/SKILL.md' "$SKILL_ROOT" ;;
    runtime-first) printf '%s/runtime-first/SKILL.md' "$SKILL_ROOT" ;;
    runtime-triage) printf '%s/runtime-triage/SKILL.md' "$SKILL_ROOT" ;;
    distributed-triage) printf '%s/distributed-triage/SKILL.md' "$SKILL_ROOT" ;;
    carts-simplify) printf '%s/carts-simplify/SKILL.md' "$SKILL_ROOT" ;;
    carts-review) printf '%s/carts-review/SKILL.md' "$SKILL_ROOT" ;;
    carts-commit) printf '%s/carts-commit/SKILL.md' "$SKILL_ROOT" ;;
    *) return 1 ;;
  esac
}

has_skill() {
  local path
  path=$(skill_path "$1") || return 1
  [ -f "$path" ]
}

add_skill() {
  local name=$1
  local reason=$2

  has_skill "$name" || return 0
  for existing in "${SKILLS[@]}"; do
    [ "$existing" = "$name" ] && return 0
  done
  [ "${#SKILLS[@]}" -ge 5 ] && return 0
  SKILLS+=("$name")
  REASONS+=("$reason")
}

matches() {
  printf '%s' "$HAYSTACK" | grep -Eq "$1"
}

add_skill check-utils "utility/helper placement before CARTS source edits"
add_skill carts-dialect-map "SDE/CODIR/ARTS/ARTS-RT ownership boundaries"

if matches 'attr|attribute|attrnames|tablegen|\.td|ods|getattr|setattr'; then
  add_skill carts-attr-consolidation "ODS-first attribute and enum consolidation rules"
elif matches 'hasattr|removeattr|discardable'; then
  add_skill carts-attr-consolidation "ODS-first attribute and enum consolidation rules"
fi

if matches 'helper|util|utility|duplicate|dedup|refactor'; then
  add_skill carts-find-utils "existing helper discovery before adding utilities"
  add_skill refactor-utils "duplicate helper consolidation workflow"
elif matches 'static function|common helper'; then
  add_skill carts-find-utils "existing helper discovery before adding utilities"
  add_skill refactor-utils "duplicate helper consolidation workflow"
fi

if matches 'include/carts|header|internal\.h|utils\.h|include tier|lib header'; then
  add_skill carts-include-tier "public include versus pass-private header placement"
fi

if matches 'pass|transform|conversion|lowering|materializ'; then
  add_skill pass-dev "pass and transform implementation conventions"
  add_skill carts-pipeline-map "pipeline stage ownership and token map"
elif matches 'canonicaliz|pipeline|tools/compile'; then
  add_skill pass-dev "pass and transform implementation conventions"
  add_skill carts-pipeline-map "pipeline stage ownership and token map"
fi

if matches 'test|lit|regression|e2e|verify|check|fixture'; then
  add_skill test "focused CARTS verification commands"
  add_skill create-test "new regression fixture workflow"
fi

if matches 'debug|crash|fail|miscompile|wrong code|reproducer|reduce'; then
  add_skill debug "compiler debugging workflow"
  add_skill miscompile-triage "wrong-code triage"
  add_skill reproducer "reduced testcase workflow"
fi

if matches 'runtime|arts_rt|arts-rt|abi|edt|db|distributed|rdma|multinode'; then
  add_skill runtime-first "runtime/compiler contract checks"
  add_skill runtime-triage "runtime failure triage"
  add_skill distributed-triage "distributed execution triage"
fi

if matches 'commit|before commit|review|simplify'; then
  add_skill carts-simplify "pre-commit simplification pass"
  add_skill carts-review "pre-commit review pass"
  add_skill carts-commit "commit readiness workflow"
fi

add_skill pass-dev "default compiler-transform workflow"

printf 'CARTS source prompt detected. Consult relevant skills before editing:\n'
for i in "${!SKILLS[@]}"; do
  path=$(skill_path "${SKILLS[$i]}")
  printf -- '- %s: %s (%s)\n' "${SKILLS[$i]}" "${REASONS[$i]}" "$path"
done
