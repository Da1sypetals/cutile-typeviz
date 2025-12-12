#!/bin/bash

pip install -e .

cd vscode_extension

# Compile TypeScript client
npm run compile

# Package the extension
yes | vsce package --allow-missing-repository

# Install the extension
code --install-extension cutile-typeviz-1.0.0.vsix

cd ..
