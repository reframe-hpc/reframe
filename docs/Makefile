# Minimal makefile for Sphinx documentation
#

# You can set these variables from the command line.
DOCS        = docs
PYTHON      = python3
SPHINXOPTS  =
SPHINXBUILD =  -msphinx
SPHINXPROJ  = ReFrame
SPHINX_VERS = sphinx-versioning
SOURCEDIR   = .
BUILDDIR    = $(VERSION)
RM          = /bin/rm -rf
TAG_VERS    = '^v\d+(\.\d+)*[a-z]*'

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

all:
	@$(SPHINX_VERS) -l conf.py build docs/ html/

latest:
	@make html
	@touch html/.nojekyll

clean:
	@echo 'Removing directories'
	-$(RM) $(TARGET_DOCS) doctrees

$(TARGET_DOCS): Makefile
	@$(PYTHON) $(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)



.PHONY: all clean latest Makefile
