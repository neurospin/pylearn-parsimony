#!/bin/bash
# If you have NEVER used this file before:
#     Please run it command by command manually in order to avoid destroying
#     your files. This script has only been tested on Ubuntu 12.04.
#
# This script requires you to install sphinxtogithub:
#
#     $ sudo pip install sphinxtogithub
#
# and epydoc:
#
#     $ sudo apt-get install python-epydoc
#
# and that Parsimony is in your PYTHONPATH:
#
#    export PYTHONPATH=$PYTHONPATH:/path/to/pylearn-parsimony/

# Build html page on your machine:
make html
epydoc -v --html parsimony -o ./_build/html/epydoc_api

# Build tmp directory:
outdir="$(mktemp -d)"
curdir="$(pwd)"

# Download Parsimony from github to upload pages:
#git clone git@github.com:neurospin/pylearn-parsimony.git $outdir
git clone https://github.com/neurospin/pylearn-parsimony $outdir

# Checkout gh-pages which is the parsimony webpage on Github and commit it:
cd $outdir
git fetch origin
git checkout -b gh-pages origin/gh-pages
alias cp='cp'
cp -r $curdir/_build/html/* $outdir
git add .
git commit -a -m "DOC: update pages"
git push origin gh-pages
cd $curdir
rm -rf $outdir

make clean