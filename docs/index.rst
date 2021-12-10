.. sagenet documentation master file, created by
   sphinx-quickstart on Tue Dec  7 05:28:34 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.
SageNet: Spatial reconstruction of single-cell dissociated datasets using graph neural networks
=========================================================================
.. raw:: html

**SageNet** is a robust and generalizable graph neural network approach that probabilistically maps  dissociated single cells from an scRNAseq dataset to their hypothetical tissue of origin using one or more reference datasets aquired by spatially resolved transcriptomics techniques. It is compatible with both high-plex imaging (e.g., seqFISH, MERFISH, etc.) and spatial barcoding (e.g., 10X visium, Slide-seq, etc.) datasets as the spatial reference. 


.. raw:: html

    <p align="center">
        <a href="">
            <img src="https://user-images.githubusercontent.com/55977725/145551267-2611c05f-0f7f-49e5-8859-0e6f5994bdb0.png"
             width="700px" alt="sagenet logo">
        </a>
    </p>

SageNet is implemented with `pytorch <https://pytorch.org/docs/stable/index.html>`_ and `pytorch-geometric <https://pytorch-geometric.readthedocs.io/en/latest/>`_ to be modular, fast, and scalable. Also, it uses ``anndata`` to be compatible with `scanpy <https://scanpy.readthedocs.io/en/stable/>`_ and `squidpy <https://squidpy.readthedocs.io/en/stable/>`_ for pre- and post-processing steps.

Installation
-------------------------------
You can get the latest development version of our toolkit from `Github <https://github.com/MarioniLab/sagenet>`_ using the following steps:

First, clone the repository using ``git``::

    git clone https://github.com/MarioniLab/sagenet

Then, ``cd`` to the sagenet folder and run the install command::

    cd sagenet
    python setup.py install #or pip install ` 


The dependency ``torch-geometric`` should be installed separately, corresponding the system specefities, look at `this link <https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html>`_ for instructions. 


.. raw:: html

    <p align="center">
        <a href="">
            <img src="https://user-images.githubusercontent.com/55977725/144909791-7b451f94-bcf4-4f2d-9f7e-6c1a692e6ffd.gif"
             width="400px" alt="activations logo">
        </a>
    </p>


Notebooks
-------------------------------
To see some examples of our pipeline's capability, look at the `notebooks <https://github.com/MarioniLab/sagenet/notebooks>`_ directory. The notebooks are also avaialble on google colab:

#. `Intro to SageNet <https://colab.research.google.com/drive/1H4gVFfxzZgilk6nbUhzFlrFsa1vEHNTl?usp=sharing>`_
#. `Using multiple references <https://colab.research.google.com/drive/1H4gVFfxzZgilk6nbUhzFlrFsa1vEHNTl?usp=sharing>`_
		
Interactive examples
-------------------------------
See `this <https://www.dropbox.com/s/krjgp19i62p7nfx/joint_mapping-2_interactive.html?dl=0>`_ 


Support and contribute
-------------------------------
If you have a question or new architecture or a model that could be integrated into our pipeline, you can
post an `issue <https://github.com/MarioniLab/sagenet/issues/new>`__ or reach us by `email <mailto:eheidari@student.ethz.ch>`_.


Contributions
-------------------------------
This work is led by Elyas Heidari and Shila Ghazanfar as a joint effort between `MarioniLab@CRUK@EMBL-EBI <https://www.ebi.ac.uk/research-beta/marioni/>`__ and `RobinsonLab@UZH <https://robinsonlabuzh.github.io>`__.




.. toctree::
   :maxdepth: 1
   :caption: Main
   :hidden:

   about
   installation
   api.rst
   
.. toctree::
   :maxdepth: 1
   :caption: Examples
   :hidden:

   01_multiple_references
   00_hello_sagenet
   
