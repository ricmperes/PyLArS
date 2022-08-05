#!/bin/bash
autopep8 --aggressive -r --in-place ./pylars
autopep8 --aggressive -r --in-place ./tests
autopep8 --aggressive -r --in-place ./scripts