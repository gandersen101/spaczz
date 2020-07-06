spaczz: Fuzzy matching and more for spaCy
=========================================

.. toctree::
   :hidden:
   :maxdepth: 1

   reference
   license

Spaczz provides fuzzy matching and multi-token regex matching
functionality to `spaCy <https://spacy.io/>`_.
Spaczz’s components have similar APIs to their spaCy counterparts and spaczz pipeline components
can integrate into spaCy pipelines where they can be saved/loaded as models.


While this website will eventually be the home for definitive spaczz documentation, for now it is kind of a placeholder.
Please visit `spaczz's GitHub page <https://github.com/gandersen101/spaczz>`_ for now to see usage documentation and more.

Installation
------------

Spaczz can be installed using pip. It is strongly recommended that the
“fast” extra is installed. This installs the optional python-Levenshtein
package which speeds up fuzzywuzzy’s fuzzy matching by 4-10x.

.. code:: Python

    # Basic Install
    pip install spaczz

    # Install with python-Levenshtein
    pip install "spaczz[fast]"

If you decide to install the optional python-Levenshtein package later
simply pip install it when desired.

.. code:: Python

    pip install python-Levenshtein
