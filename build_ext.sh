cd vscode_extension/server
npm run compile
cd ..
npm run compile
yes | vsce package --allow-missing-repository 
code --install-extension cutile-typeviz-1.0.0.vsix
cd ..