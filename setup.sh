#!/bin/bash
#LINK TO GOOD RESEARCH CODEBOOK https://goodresearch.dev/setup

pip install cookiecutter
cookiecutter gh:patrickmineault/true-neutral-cookiecutter
cd cookbook
pip install -e .
###### Start by creating a new repository on github website, then follow commands below############
git init
git add README.md
git commit -m "first commit"
git branch -M main
git remote add origin https://github.com/shivam-chandhok/cookbook.git
git push -u origin main
###############################################################


######### Create conda environment, install packages and create yml file################
#conda create --name codebook python=3.8
#conda activate codebook
#conda install pandas numpy scipy matplotlib seaborn
#conda env export > environment.yml
###############################################################

#######Format files or all files in a folder#####
#black file/foldername
################################################