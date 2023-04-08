#!/bin/bash
cd Common_cython
sh compile.sh
cd ../
python3 NEOS/PWSterileFitTable1.py
