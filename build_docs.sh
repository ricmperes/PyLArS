#!/bin/bash

rm -r ./docs/*
pdoc --force --html --output-dir ./docs pylars
mv ./docs/pylars/* ./docs/
rm -r ./docs/pylars

echo "Docs built with pdoc3!"
