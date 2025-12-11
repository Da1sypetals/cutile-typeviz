#!/bin/bash
# Build script for cuTile typeviz extension (Python backend)
# Run this script from the root directory of the project

cd vscode_extension

# Install Python server dependencies
pip install -r server_py/requirements.txt

# Compile TypeScript client
npm run compile

# Package the extension
yes | vsce package --allow-missing-repository

# Install the extension
code --install-extension cutile-typeviz-1.0.0.vsix

cd ..
