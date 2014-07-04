Required Softwares
------------------
Make sure the below softwares have been installed on your machine:

* http://sphinx-doc.org/latest/install.html
* https://pypi.python.org/pypi/sphinxtogithub
* Latex (for example $ sudo apt-get install texlive-full)
* dvipng which is usually included in latex package


Build pages
-----------

We assume that your parsimony repository is located in "~/pylearn-parsimony". Goto the directory which contains the doc source, and then build the doc.

```
$ PARSIMONYDIR=~/pylearn-parsimony
$ cd $PARSIMONYDIR/doc/source
$ make html
$ mkdir -p ./_build/html/epydoc_api
$ epydoc -v --html parsimony -o ./_build/html/epydoc_api
```

The website has been built in "~/pylearn-parsimony/doc/source/_build/html". You can open "index.html" by firefox to test.

```
$ firefox  $PARSIMONYDIR/doc/source/_build/html/index.html
```

Upload to github
----------------
"$PARSIMONYDIR/doc/source/_build/html" contains the parsimony website. Now we start to upload to github server. Clone parsimony from github to a temporary directory, and checkout gh-pages branch

```
$ cd /tmp
$ git clone git@github.com:neurospin/pylearn-parsimony.git parsimony_doc
$ cd parsimony_doc
$ git fetch origin
$ git checkout -b gh-pages origin/gh-pages
```

Copy the built website and push to the gh-pages branch on github.

```
$ cp -r $PARSIMONYDIR/doc/source/_build/html/* ./
$ git add .
$ git commit -a -m "DOC: update pages"
$ git push origin gh-pages
```

Now, you can visit your updated website at http://neurospin.github.io/pylearn-parsimony.
