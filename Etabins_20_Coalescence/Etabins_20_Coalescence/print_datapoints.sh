#!/bin/bash

for energy in "7GeV" "9GeV" "11GeV" "14GeV" "17GeV" "19GeV" "27GeV"; do
    for particle in pions kaons protons; do
        root -b -q "PrintDatapoints.C(\"${energy}\", \"${particle}\");"
    done
done

