#!/bin/bash

# force a1 plots to be generated without going through the full pipeline

if [ $# -ne 1 ]; then
    echo "Usage: sh $0 <energy>"
    echo "Example: sh $0 19p6GeV"
    exit 1
fi
energy=$1
snakemake --cores all result/sys_tag_0/fit_Lambda_a1_${energy}.csv --touch
snakemake --cores all result/sys_tag_0/fit_Lambdabar_a1_${energy}.csv --touch
snakemake --cores all plots/sys_tag_0/a1_cen_${energy}.pdf