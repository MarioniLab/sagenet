pip install torch-scatter -f https://data.pyg.org/whl/torch-1.10.0+cu113.html
pip install torch-sparse -f https://data.pyg.org/whl/torch-1.10.0+cu113.html
pip install torch-geometric

from pathlib import Path

from setuptools import setup, find_packages

long_description = Path('README.rst').read_text('utf-8')

try:
    from GNNProject import __author__, __email__
except ImportError:  # Deps not yet installed
    __author__ = __maintainer__ ='Elyas Heidari'
    __email__ = ['eheidari@student.ethz.ch']

setup(name='sagenet',
      __version__ = "0.1.0",
      description='Spatial reconstruction of dissociated single-cell data',
      long_description=long_description,
      long_description_content_type="text/markdown",
      url='https://github.com/MarioniLab/sagenet',
      author=__author__,
      author_email=__email__,
      license='MIT',
      platforms=["Linux", "MacOSX"],
      packages=find_packages(),
      zip_safe=False,
    #   download_url="https://pypi.org/project/squidpy/",
    # project_urls={
    #     "Documentation": "https://squidpy.readthedocs.io/en/stable",
    #     "Source Code": "https://github.com/theislab/squidpy",
    # },
    install_requires=[l.strip() for l in Path("requirements.txt").read_text("utf-8").splitlines()],
      classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Science/Research",
        "Natural Language :: English",
        "License :: OSI Approved :: BSD License",
        "Operating System :: POSIX :: Linux",
        "Operating System :: MacOS :: MacOS X",
        "Typing :: Typed",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Environment :: Console",
        "Framework :: Jupyter",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "Topic :: Scientific/Engineering :: Visualization",
      ],
      doc=[
          'sphinx',
          'sphinx_rtd_theme',
          'sphinx_autodoc_typehints',
          'typing_extensions; python_version < "3.8"',
      ],
    keywords=sorted(
        [
            "single-cell",
            "bio-informatics",
            "spatial transcriptomics",
            "spatial data analysis",
            "single-cell data analysis",
        ]
    ),
)
