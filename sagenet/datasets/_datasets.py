from copy import copy
from squidpy.datasets._utils import AMetadata

_MGA = AMetadata(
    name="MGA",
    doc_header="",
    # shape=(270876, 43),
    url="https://figshare.com/ndownloader/files/31700690",
)

_seqFISH = AMetadata(
    name="seqFISH",
    doc_header="",
    # shape=(270876, 43),
    url="https://figshare.com/ndownloader/files/31700699",
)

for name, var in copy(locals()).items():
    if isinstance(var, AMetadata):
        var._create_function(name, globals())


__all__ = [  # noqa: F822
    "MGA",
    "seqFISH"
]