# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import os
import sys
from datetime import datetime

sys.path.insert(0, os.path.abspath('.'))
sys.path.insert(0, os.path.abspath('../../src')) # Adjust this to your project's root
sys.path.append(os.path.abspath('sphinxext'))



project = 'Gal3D'
copyright = str(datetime.now().year)

language = os.environ.get('SPHINX_LANGUAGE') or 'zh_CN'
# Set author based on language
if language == 'zh_CN':
    author = '卢帅'
else:
    author = 'Shuai Lu'



import gal3d

version = ".".join(gal3d.__version__.split(".")[:2])
release = version



# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',      # Core library for autodoc
    'sphinx.ext.autosummary',  # Create neat summary tables
    'sphinx.ext.napoleon',     # Support for NumPy and Google style docstrings
    'sphinx.ext.coverage',     # Measure code coverage
    'sphinx.ext.mathjax',      # Support for LaTeX math
    'sphinx.ext.viewcode',     # Add links to highlighted source code , not include
    'sphinx.ext.todo',         # Support for todo items
    'sphinx_copybutton',       # Add a "copy" button to code blocks
    'numpydoc',                # Support for NumPy docstrings
   # 'nbsphinx',                # Support for Jupyter Notebooks
    'myst_nb',                  # Support for Jupyter Notebooks
    # Add other extensions here, e.g., 'myst_parser' for Markdown support
]

# nbsphinx settings
# nbsphinx_input_prompt = 'In [%s]:'
# nbsphinx_output_prompt = 'Out[%s]:'
# nbsphinx_execute = 'never' # the notebook is expensive to evaluate, so we need it pre-evaluated

nb_execution_mode = "off"
nb_number_source_lines = False  # Number code cell source lines
nb_merge_streams = True

# Enable MyST extensions
myst_enable_extensions = [
    "amsmath",
    "dollarmath"
    # Add other extensions if needed
]

ipython_warning_is_error = False # IPython warnings are not errors
ipython_savefig_dir = 'plots' # Directory to save IPython figures
plot_working_directory = '.' # Working directory for plots, when .. plot::
extensions+=['IPython.sphinxext.ipython_console_highlighting',
             'IPython.sphinxext.ipython_directive'] # IPython support


# The master toctree document.
master_doc = 'index'

# The suffix of source filenames.
source_suffix = '.rst'


# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']



# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = 'sphinx'



# The theme to use for HTML and HTML Help pages.
# Popular themes: 'sphinx_rtd_theme', 'furo', 'pydata_sphinx_theme', 'sphinx_book_theme'
html_theme = 'sphinx_book_theme'

# Theme options are theme-specific and customize the look and feel of a theme.
#html_theme_options = {
    # Repository and navigation settings
 #   "repository_provider": "github",
 #   "repository_url": "https://github.com/GalaxySimAnalytics/gal3d",
    
 #   "use_repository_button": True,      # Show repository button
   # "use_issues_button": True,          # Show issues button  
    #"use_edit_page_button": True,       # Show edit page button
    
    #"use_sidenotes": True,
   # "use_source_button": True,
 #   "path_to_docs": "docs/source",      # Path to documentation source in repo
   # "home_page_in_toc": True,           # Include home page in table of contents
    
    # Navigation and layout
    #"show_navbar_depth": 2,             # Depth of navigation bar
  #  "show_toc_level": 2,                # Depth of table of contents in sidebar
    #"collapse_navigation": False,        # Keep navigation expanded
    #"navigation_with_keys": True,        # Enable keyboard navigation
    
    # Logo and branding
  #  "logo": {
  #      "text": "Gal3D Documentation", # Logo text (if no image)
        # Add a subtitle/description
  #      "subtitle": "Galaxy 3D morphology model"
        # "image_light": "logo-light.png",  # Logo image for light mode
        # "image_dark": "logo-dark.png",    # Logo image for dark mode
  #  },
    
    # Search functionality
   # "search_bar_text": "Search the documentation...",  # Search bar placeholder
    
    # Launch buttons for interactive content
  #  "launch_buttons": {                 # Add launch buttons
  #      "notebook_interface": "jupyterlab",  # Default notebook interface
   #     "binderhub_url": "",            # Binder URL if using Binder
    #    "colab_url": "",                # Google Colab URL if using Colab
  #  },
    
    # Footer
    #"extra_footer": "",                 # Additional footer content
    
    # Announcement banner (optional)
  #  "announcement": "This documentation is under active development!",
    
    # Theme variants
    #"pygment_light_style": "default",   # Code highlighting style for light mode
    #"pygment_dark_style": "native",     # Code highlighting style for dark mode
    
    # Additional sphinx_book_theme specific options
    #"show_prev_next": True,             # Show previous/next navigation
   # "use_download_button": True,        # Show download button
    #"use_fullscreen_button": True,      # Show fullscreen button
#}

if language == 'zh_CN':
    html_theme_options = {
        "repository_provider": "github",
        "repository_url": "https://github.com/GalaxySimAnalytics/gal3d",
        "use_repository_button": True,
        "path_to_docs": "docs/source",
        "show_toc_level": 2,
        "announcement": "本文档正在建设中！",
    }
else:
    html_theme_options = {
        "repository_provider": "github",
        "repository_url": "https://github.com/GalaxySimAnalytics/gal3d",
        "use_repository_button": True,
        "path_to_docs": "docs/source",
        "show_toc_level": 2,
        "announcement": "This documentation is under active development!",
    }

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files.
html_static_path = ['_static']

# These paths are either relative to html_static_path
# or fully qualified paths (e.g. https://...)
html_css_files = ['gal3d.css']
html_logo = "_static/logo.png"

# -- Options for autodoc extension -------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/extensions/autodoc.html#configuration
autosummary_generate = True
#autodoc_typehints = "description"  # Show typehints in the description, not the signature
#autoclass_content = "both"         # Include both the class's and the __init__ docstring
#autodoc_member_order = 'bysource'  # Order members by source order


# -- Options for intersphinx extension ---------------------------------------
# https://www.sphinx-doc.org/en/master/usage/extensions/intersphinx.html#configuration

# Example configuration for intersphinx: refer to the Python standard library.
#intersphinx_mapping = {
#    'python': ('https://docs.python.org/3', None),
#    'numpy': ('https://numpy.org/doc/stable/', None),
#    'pandas': ('https://pandas.pydata.org/pandas-docs/stable/', None),
#}


# -- Options for todo extension ----------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/extensions/todo.html#configuration

# If true, `todo` and `todoList` produce output, else they produce nothing.
todo_include_todos = True


# -- Options for copybutton extension ----------------------------------------
# https://sphinx-copybutton.readthedocs.io/en/latest/use.html

copybutton_prompt_text = r">>> |\.\.\. |\$ "
copybutton_prompt_is_regexp = True


# -- Options for LaTeX output ------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-latex-output

#latex_elements = {
    # 'papersize': 'letterpaper',
    # 'pointsize': '10pt',
    # 'preamble': '',
#}
latex_engine = 'xelatex'
latex_elements = {
    'preamble': r'''
\usepackage{xeCJK}
\setCJKmainfont{Noto Sans CJK SC}
''',
}

# Grouping the document tree into LaTeX files. List of tuples
# (source start file, target name, title, author, documentclass [howto, manual, or own class]).
latex_documents = [
    (master_doc, 'gal3d.tex', 'Gal3d Documentation',
     author, 'manual'),
]


# -- Options for manual page output ---------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-manual-page-output

# One entry per manual page. List of tuples
# (source start file, name, description, authors, manual section).
man_pages = [
    (master_doc, 'gal3d', 'Gal3d Documentation',
     [author], 1)
]


# -- Options for Texinfo output ----------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-texinfo-output



# Options for copybutton extension
copybutton_copy_empty_lines = False
copybutton_selector =  "div.highlight > pre"
copybutton_prompt_text = r">>> |\.\.\. |\$ |In \[\d*\]: | {2,5}\.{3,5}: | {5,8}: "
copybutton_prompt_is_regexp = True
copybutton_only_copy_prompt_lines = True


import matplotlib

matplotlib.rcParams['savefig.dpi'] = 200 # Set the DPI for saved figures
matplotlib.rcParams['savefig.bbox'] = 'tight' # Set the bounding box to 'tight'
matplotlib.rcParams['savefig.pad_inches'] = 0.15 # Set the padding for saved figures
matplotlib.rcParams['figure.figsize'] = (5, 4)
