# Minimal makefile for Sphinx documentation
#

# You can set these variables from the command line.
PYTHON      = python
SPHINXOPTS  =
SPHINXBUILD =  -msphinx
SPHINXPROJ  = ReFrame
SOURCEDIR   = .
BUILDDIR    = $(VERSION)
PANDOC      = pandoc
PANDOCOPTS  = --columns 1000
RM          = /bin/rm -rf

ifeq ($(finstring "darwin", $(OSTYPE)), "darwin")
SYMLINK_DIR=ln -sfh
else
# Assume a GNU/Linux system here
SYMLINK_DIR=ln -sfn
endif

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
	@make html
	@touch html/.nojekyll
	@echo 'Linking to old documentation pages'
	@cd html && $(SYMLINK_DIR) ../old _old && cd - > /dev/null

%.rst: %.md
	$(PANDOC) $(PANDOCOPTS) --from=markdown --to=rst --output=$(@) $(@:.rst=.md)

clean:
	@echo 'Removing md files'
	-$(RM) $(RSTS) doctrees

distclean: clean
	@echo 'Removing directories'
	-$(RM) $(TARGET_DOCS)

$(TARGET_DOCS): Makefile
	@$(PYTHON) $(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)


.PHONY: clean Makefile
