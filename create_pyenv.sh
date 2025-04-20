#!/bin/bash

PY_VERSION="3.12.10"
ENV_NAME=".venv"

# Ensure pyenv is loaded
export PYENV_ROOT="$HOME/.pyenv"
command -v pyenv >/dev/null || export PATH="$PYENV_ROOT/bin:$PATH"
eval "$(pyenv init -)"

if ! pyenv versions --bare | grep -qx "$PY_VERSION"; then
    pyenv install "$PY_VERSION"
else
    echo "Python $PY_VERSION already installed. Skipping."
fi

pyenv local "$PY_VERSION"
echo "Using Python: $(python --version)"

python -m venv "$ENV_NAME"
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

echo -e "\033[1;36mVirtual environment created, use: \033[1;92msource $ENV_NAME/bin/activate\033[1;36m to activate it.\033[00m"
