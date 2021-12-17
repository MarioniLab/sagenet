from copy import copy
from squidpy.datasets._utils import AMetadata

_MGA_scRNAseq = AMetadata(
    name="MGA_scRNAseq",
    doc_header="",
    # shape=(270876, 43),
    url="https://figshare.com/ndownloader/files/31767704",
)

_MGA_seqFISH1 = AMetadata(
    name="MGA_seqFISH1",
    doc_header="",
    # shape=(270876, 43),
    url="https://figshare.com/ndownloader/files/31716029",
)

_MGA_seqFISH2 = AMetadata(
    name="MGA_seqFISH2",
    doc_header="",
    # shape=(270876, 43),
    url="https://figshare.com/ndownloader/files/31716041",
)


_MGA_seqFISH3 = AMetadata(
    name="MGA_seqFISH3",
    doc_header="",
    # shape=(270876, 43),
    url="https://figshare.com/ndownloader/files/31716089",
)

_DHH_visium_ = AMetadata(
    name="DHH_visium_",
    doc_header="",
    # shape=(270876, 43),
    url="https://figshare.com/ndownloader/files/31796207",
)

_DHH_scRNAseq = AMetadata(
    name="DHH_scRNAseq",
    doc_header="",
    # shape=(270876, 43),
    url="https://figshare.com/ndownloader/files/31796219",
)




for name, var in copy(locals()).items():
    if isinstance(var, AMetadata):
        var._create_function(name, globals())


__all__ = [  # noqa: F822
    "MGA_scRNAseq",
    "MGA_seqFISH1",
    "MGA_seqFISH2",
    "MGA_seqFISH3",
    'DHH_visium_',
    'DHH_scRNAseq'
    
]
