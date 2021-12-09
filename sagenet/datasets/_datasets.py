from copy import copy
from squidpy.datasets._utils import AMetadata

_MGA = AMetadata(
    name="MGA",
    doc_header="",
    # shape=(270876, 43),
    url="https://figshare.com/ndownloader/files/31716146",
)

_seqFISH1 = AMetadata(
    name="seqFISH1",
    doc_header="",
    # shape=(270876, 43),
    url="https://figshare.com/ndownloader/files/31716029",
)

_seqFISH2 = AMetadata(
    name="seqFISH2",
    doc_header="",
    # shape=(270876, 43),
    url="https://figshare.com/ndownloader/files/31716041",
)


_seqFISH3 = AMetadata(
    name="seqFISH3",
    doc_header="",
    # shape=(270876, 43),
    url="https://figshare.com/ndownloader/files/31716089",
)


for name, var in copy(locals()).items():
    if isinstance(var, AMetadata):
        var._create_function(name, globals())


__all__ = [  # noqa: F822
    "MGA",
    "seqFISH1",
    "seqFISH2",
    "seqFISH3",
]