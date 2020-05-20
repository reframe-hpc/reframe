# Minimal makefile for Sphinx documentation
#

# You can set these variables from the command line.
DOCS        = docs
PYTHON      = python3
SPHINXOPTS  =
SPHINXBUILD =  -msphinx
SPHINXPROJ  = ReFrame
SOURCEDIR   = .
BUILDDIR    = $(VERSION)
RM          = /bin/rm -rf

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

latest:
	@make html man
	@touch html/.nojekyll

clean:
	@echo 'Removing directories'
	-$(RM) $(TARGET_DOCS) doctrees

$(TARGET_DOCS): Makefile
	@$(PYTHON) $(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)



.PHONY: clean latest Makefile
