#!/bin/bash
# filepath: docs/source/gen_api.sh

# This script generates API documentation for the gal3d package using Sphinx.
set -euo pipefail   # Enable strict error handling

cd "$(dirname "$0")" # Change to the directory where the script is located


# Check if uv is available and set the command accordingly
if command -v uv >/dev/null 2>&1; then
  RUNNER="uv run"
else
  RUNNER=""
fi

# Set the Sphinx commands, using uv if available
SPHINX_APIDOC="${SPHINX_APIDOC:-$RUNNER sphinx-apidoc}"
SPHINX_AUTOGEN="${SPHINX_AUTOGEN:-$RUNNER sphinx-autogen}"

# Remove old API documentation
rm -rf reference/_autosummary

# Generate API documentation rst files into the reference/_autosummary directory
$SPHINX_APIDOC --separate -t _templates/apidoc -o reference/_autosummary ../../src/gal3d

# Remove the generated modules.rst file since we will use index.rst to control the API documentation structure
rm -f reference/_autosummary/modules.rst
rm -f reference/_autosummary/gal3d.rst

# Generate autosummary stub files from index.rst
find reference -name "*.rst" -print0 | xargs -0 $SPHINX_AUTOGEN