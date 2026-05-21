# Minimal makefile for Sphinx documentation
#

# You can set these variables from the command line, and also
# from the environment for the first two.
SPHINXJOBS      ?= auto
SPHINXOPTS      ?= -j $(SPHINXJOBS)
SPHINXBUILD     ?= uv run sphinx-build
SPHINXINTL      ?= uv run sphinx-intl
SPHINXAUTOBUILD ?= uv run sphinx-autobuild
SOURCEDIR       = source
BUILDDIR        = build

# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
	@echo ""
	@echo "Extra custom targets:"
	@echo "  html-en        Build English HTML documentation (output to build/en)"
	@echo "  html-zh        Build Chinese HTML documentation (output to build/zh_CN)"
	@echo "  i18n-en        Extract and update English translation files"
	@echo "  i18n-zh        Extract and update Chinese translation files"
	@echo "  i18n-all       Extract and update both English and Chinese translation files"
	@echo "  autobuild-en   Live preview of English documentation with auto-reload"
	@echo "  autobuild-zh   Live preview of Chinese documentation with auto-reload"
	@echo "  help           Show this help message"

.PHONY: help Makefile

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
%: Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
# Catch-all target: delegate unknown targets to sphinx-build using "make mode"

# Build English HTML documentation, output to build/en
html-en:
	SPHINX_LANGUAGE=en $(SPHINXBUILD) -b html -D language=en $(SOURCEDIR) $(BUILDDIR)/en

# Build Chinese HTML documentation, output to build/zh_CN
html-zh:
	SPHINX_LANGUAGE=zh_CN $(SPHINXBUILD) -b html -D language=zh_CN $(SOURCEDIR) $(BUILDDIR)/zh_CN


# Extract English translation strings and update English po files
i18n-en:
	$(SPHINXBUILD) -b gettext $(SOURCEDIR) $(BUILDDIR)/gettext
	$(SPHINXINTL) update -p $(BUILDDIR)/gettext -l en
	$(SPHINXINTL) build

# Extract Chinese translation strings and update Chinese po files
i18n-zh:
	$(SPHINXBUILD) -b gettext $(SOURCEDIR) $(BUILDDIR)/gettext
	$(SPHINXINTL) update -p $(BUILDDIR)/gettext -l zh_CN
	$(SPHINXINTL) build

# Extract and update both English and Chinese po files
i18n-all:
	$(SPHINXBUILD) -b gettext $(SOURCEDIR) $(BUILDDIR)/gettext
	$(SPHINXINTL) update -p $(BUILDDIR)/gettext -l en -l zh_CN
	$(SPHINXINTL) build


.PHONY: i18n-en i18n-zh i18n-all

# Live preview of English documentation with auto-reload
autobuild-en:
	SPHINX_LANGUAGE=en $(SPHINXAUTOBUILD) -b html -D language=en $(SOURCEDIR) $(BUILDDIR)/en

# Live preview of Chinese documentation with auto-reload
autobuild-zh:
	SPHINX_LANGUAGE=zh_CN $(SPHINXAUTOBUILD) -b html -D language=zh_CN $(SOURCEDIR) $(BUILDDIR)/zh_CN

.PHONY: autobuild-en autobuild-zh