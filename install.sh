#!/usr/bin/env bash
set -euo pipefail

REPO="git+https://github.com/sharifli4/agent-verdict.git"
BOLD="\033[1m"
GREEN="\033[32m"
CYAN="\033[36m"
YELLOW="\033[33m"
RED="\033[31m"
RESET="\033[0m"

info()  { printf "${CYAN}▸${RESET} %s\n" "$1"; }
ok()    { printf "${GREEN}✓${RESET} %s\n" "$1"; }
warn()  { printf "${YELLOW}!${RESET} %s\n" "$1"; }
fail()  { printf "${RED}✗${RESET} %s\n" "$1"; exit 1; }

# --- detect provider and extras from args or env ---
PROVIDER=""
MCP=false
for arg in "$@"; do
    case "$arg" in
        anthropic|claude) PROVIDER="anthropic" ;;
        openai|gpt)       PROVIDER="openai" ;;
        all)               PROVIDER="anthropic,openai"; MCP=true ;;
        mcp)               MCP=true ;;
    esac
done

if [ -z "$PROVIDER" ]; then
    if [ -n "${ANTHROPIC_API_KEY:-}" ] && [ -n "${OPENAI_API_KEY:-}" ]; then
        PROVIDER="anthropic,openai"
    elif [ -n "${ANTHROPIC_API_KEY:-}" ]; then
        PROVIDER="anthropic"
    elif [ -n "${OPENAI_API_KEY:-}" ]; then
        PROVIDER="openai"
    else
        PROVIDER="anthropic"
    fi
fi

if [ "$MCP" = true ]; then
    EXTRAS="agent-verdict[${PROVIDER},mcp]"
else
    EXTRAS="agent-verdict[${PROVIDER}]"
fi

echo ""
printf "${BOLD}  agent-verdict installer${RESET}\n"
echo ""

# --- find package manager ---
if command -v uv >/dev/null 2>&1; then
    PM="uv"
    info "Using uv"
    uv pip install "${EXTRAS} @ ${REPO}" 2>/dev/null || \
    uv pip install --system "${EXTRAS} @ ${REPO}"
elif command -v pipx >/dev/null 2>&1; then
    PM="pipx"
    info "Using pipx"
    pipx install "${EXTRAS} @ ${REPO}"
elif command -v pip >/dev/null 2>&1; then
    PM="pip"
    info "Using pip"
    pip install "${EXTRAS} @ ${REPO}"
elif command -v pip3 >/dev/null 2>&1; then
    PM="pip3"
    info "Using pip3"
    pip3 install "${EXTRAS} @ ${REPO}"
else
    fail "No package manager found. Install uv, pip, or pipx first."
fi

echo ""
if [ "$MCP" = true ]; then
    ok "Installed agent-verdict with ${PROVIDER} provider + MCP server"
else
    ok "Installed agent-verdict with ${PROVIDER} provider"
fi
echo ""

# --- check API key ---
NEED_KEY=""
if echo "$PROVIDER" | grep -q "anthropic"; then
    if [ -z "${ANTHROPIC_API_KEY:-}" ]; then
        NEED_KEY="anthropic"
    fi
fi
if echo "$PROVIDER" | grep -q "openai"; then
    if [ -z "${OPENAI_API_KEY:-}" ]; then
        NEED_KEY="${NEED_KEY:+$NEED_KEY and }openai"
    fi
fi

if [ -n "$NEED_KEY" ]; then
    warn "Set your API key to get started:"
    echo ""
    if echo "$NEED_KEY" | grep -q "anthropic"; then
        printf "    export ANTHROPIC_API_KEY=sk-ant-...\n"
    fi
    if echo "$NEED_KEY" | grep -q "openai"; then
        printf "    export OPENAI_API_KEY=sk-...\n"
    fi
    echo ""
fi

printf "${BOLD}  Try it:${RESET}\n"
echo ""
printf "    agent-verdict evaluate \"your agent's answer\" -c \"what it should do\"\n"
echo ""

if [ "$MCP" = true ]; then
    printf "${BOLD}  Add to Claude Code:${RESET}\n"
    echo ""
    printf "    claude mcp add agent-verdict agent-verdict-mcp\n"
    echo ""
fi
