#!/bin/bash
set -x
pip install twine
echo $TWINE_USERNAME
echo $TWINE_PASSWORD
twine upload --username $TWINE_USERNAME --password $TWINE_PASSWORD wheelhouse/*
