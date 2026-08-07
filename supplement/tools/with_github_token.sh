#!/bin/sh
# Run a command with GITHUB_ACCESS_TOKEN set from a file OUTSIDE the repo.
#
# LeanDojo resolves `leanprover/lean4` tags through the GitHub API on every `LeanGitRepo`
# construction. Unauthenticated that is 60 requests/hour, which four parallel workers exhaust
# immediately -- the failure mode is `403: rate limit exceeded` followed by a ~22 minute backoff,
# with no classes processed.
#
# The token is read here from a path given by GITHUB_TOKEN_FILE. It is never written into this
# repository, never passed on a command line where
# it would show up in `ps`, and never echoed. Only the length is reported so the caller can tell a
# successful load from an empty one.
#
# Usage:
#   tools/with_github_token.sh <command> [args...]
set -eu

: "${GITHUB_TOKEN_FILE:?set GITHUB_TOKEN_FILE to a readable token file}"
TOKEN_FILE="$GITHUB_TOKEN_FILE"

if [ ! -r "$TOKEN_FILE" ]; then
  echo "with_github_token: cannot read $TOKEN_FILE" >&2
  exit 1
fi

# Match the token by its own shape rather than by parsing the surrounding line. Stripping a
# `KEY=` prefix with sed was fragile: it silently returned the whole 66-char line, which the API
# rejected with 401 while still looking like a successful load.
GITHUB_ACCESS_TOKEN=$(grep -oE '(ghp_|github_pat_|gho_|ghs_)[A-Za-z0-9_]+' "$TOKEN_FILE" | head -n 1)
export GITHUB_ACCESS_TOKEN

if [ -z "$GITHUB_ACCESS_TOKEN" ]; then
  echo "with_github_token: no token found in $TOKEN_FILE" >&2
  exit 1
fi
echo "with_github_token: loaded token (${#GITHUB_ACCESS_TOKEN} chars) from outside the repo" >&2

exec "$@"
