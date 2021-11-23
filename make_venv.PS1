#!/bin/bash
Set-Location $PSScriptRoot
$VENV_NAME="python-venv"
$REQ_FILE="requirements.txt"
$PY_VER="3"

if ( Test-Path -Path "$VENV_NAME" -PathType Container ) { 
	echo "Virtual environment $VENV_NAME already exists!"
} else {
	echo "Virtual environment $VENV_NAME..."
	&python.exe -m venv "$VENV_NAME"
	echo "*" | Set-Content -NoNewline  "$VENV_NAME/.gitignore"
	&"$VENV_NAME\Scripts\python.exe" -m pip install --upgrade pip
	&"$VENV_NAME\Scripts\pip.exe" install -r "$REQ_FILE"
	&"$VENV_NAME\Scripts\pip.exe" freeze > "$VENV_NAME/$REQ_FILE"
	echo "...Done."
}


