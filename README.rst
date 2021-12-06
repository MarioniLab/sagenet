SageNet: spatial reconstruction of dissociated single-cell data using graph neural networks
=========================================================================
.. raw:: html

**SageNet** is a robust and generalizable graph neural network approach that probabilistically maps dissociated single cells to their hypothetical tissue of origin using one or more references datasets aquired by spatially resolved transcriptomics techniques. It is compatible with both high-plex imaging (e.g., seqFISH, MERFISH, etc.) and spatial barcoding (e.g., 10X visium, Slide-seq, etc.) datasets as the spatial reference. 


.. raw:: html

    <p align="center">
        <a href="">
            <img src="https://github.com/MarioniLab/sagenet/files/7663030/final.pdf"
             width="700px" alt="sagenet logo">
        </a>
    </p>

SageNet implemented with `pytorch <https://pytorch.org/docs/stable/index.html>`_ and `pytorch-geometric <https://pytorch-geometric.readthedocs.io/en/latest/>`_ in python to be modular, fast, and scalable.

The workflow
-------------------------------

#. Training phase:
	* Input: 
		* Expression matrix associated with the (spatial) reference dataset (an `anndata` object)

		* gene-gene interaction network

		* one or more partitionings of the spatial reference into distinct connected neighborhoods of cells or spots

	* Output:
		* A set of pre-trained models (one for each partitioning)

		* A concensus scoring of spatially informativity of each gene


#. Mapping phase:
	* Input: 
		* Expression matrix associated with the (dissociated) query dataset (an `anndata` object)

	* Output:
		* The reconstructed cell-cell spatial distance matrix

		* A concensus scoring of mapability (uncertainity of mapping) of each cell to the references
    <p align="center">
        <a href="">
            <img src="https://user-images.githubusercontent.com/55977725/144909791-7b451f94-bcf4-4f2d-9f7e-6c1a692e6ffd.gif"
             width="400px" alt="activations logo">
        </a>
    </p>

Usage and installation
-------------------------------
You can get the latest development version of our toolkit from `Github <https://github.com/e-sollier/DL2020/>`_ using the following steps:
First, clone the repository using ``git``::

    git clone https://github.com/MarioniLab/sagenet

Then, ``cd`` to the scArches folder and run the install command::

    cd sagenet
    python setup.py install #or `pip install .` 


The dependency `torch-geometric` should be installed separately based on the system specefities, look at `this <https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html>`_. for the instructions. 

Support and contribute
-------------------------------
If you have a question or new architecture or a model that could be integrated into our pipeline, you can
post an `issue <https://github.com/MarioniLab/sagenet/issues/new>`__ or reach us by `email <mailto:eheidari@student.ethz.ch>`_.


