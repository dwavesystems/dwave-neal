#!/bin/bash
set -x
pip install twine
twine upload wheelhouse/*
