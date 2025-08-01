#!/bin/bash
# filepath: docs/source/gen_api.sh

# Generate API documentation rst files into the reference/_autosummary directory
sphinx-apidoc -o reference/_autosummary ../../src/gal3d