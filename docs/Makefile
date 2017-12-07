# Minimal makefile for Sphinx documentation
#

# You can set these variables from the command line.
BUILD_CMD   = build
DOCS        = docs
DOCS_ALL    = _build/_allversions
MASTER      = master
PYTHON      = python
SPHINXOPTS  =
SPHINXBUILD =  -msphinx
SPHINXPROJ  = ReFrame
SPHINX_VERS = sphinx-versioning
SOURCEDIR   = .
BUILDDIR    = $(VERSION)
PANDOC      = pandoc
PANDOCOPTS  = --columns 1000
RM          = /bin/rm -rf
TAG_VERS    = '^v\d+(\.\d+)*[a-z]*'

MDS = about.md \
	advanced.md \
	configure.md \
	deferrables.md \
	pipeline.md \
	running.md \
	started.md \
	tutorial.md \
	usecases.md

RSTS = $(MDS:.md=.rst)

TARGET_DOCS := \
	help \
	html \
	dirhtml \
	singlehtml \
	pickle \
	json \
	htmlhelp \
	qthelp \
	devhelp \
	epub \
	latex \
	latexpdf \
	latexpdfja \
	text \
	man \
	textinfo \
	info \
	gettext \
	changes \
	xml \
	pseudoxml \
	linkcheck \
	doctest \
	coverage

all: $(RSTS)
	@$(SPHINX_VERS) -l conf.py build docs/ html/
	@rsync -az old/ html/_old/
	@./link_old_docs.sh

latest: $(RSTS)
	@make html
	@touch html/.nojekyll
	@rsync -az old/ html/_old/

%.rst: %.md
	$(PANDOC) $(PANDOCOPTS) --from=markdown --to=rst --output=$(@) $(@:.rst=.md)

clean:
	@echo 'Removing md files'
	-$(RM) $(RSTS)

distclean: clean
	@echo 'Removing directories'
	-$(RM) $(TARGET_DOCS) doctrees

$(TARGET_DOCS): Makefile
	@$(PYTHON) $(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)



.PHONY: all distclean clean latest Makefile
