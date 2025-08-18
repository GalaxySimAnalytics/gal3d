#!/bin/bash
# filepath: docs/source/gen_api.sh

# Exit immediately if a command exits with a non-zero status
set -e

# Remove old API documentation
rm -rf reference/_autosummary

# Generate API documentation rst files into the reference/_autosummary directory
sphinx-apidoc -o reference/_autosummary ../../src/gal3d

# Generate autosummary stub files from index.rst
find reference -name "*.rst" | xargs sphinx-autogen