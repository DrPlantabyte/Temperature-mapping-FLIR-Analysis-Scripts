#!/bin/bash
set -e
VENV_NAME="python-venv"
REQ_FILE="requirements.txt"
PY_VER="3"

if test ! -d "$VENV_NAME"; then
	"python${PY_VER}" -m venv "$VENV_NAME" &&
	echo "*" > "$VENV_NAME"/.gitignore &&
	source "$VENV_NAME"/bin/activate &&
	pip install --upgrade pip &&
	pip install -r "$REQ_FILE" &&
	pip freeze > "$VENV_NAME/$REQ_FILE" &&
	deactivate
fi

