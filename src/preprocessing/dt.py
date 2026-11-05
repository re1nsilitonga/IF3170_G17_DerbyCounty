# Preprocessor untuk Decision Tree.

from .common import BasePreprocessor


# NaN dan kategori teks dibiarkan karena pohon bisa menanganinya langsung, tanpa scaling.
class TreePreprocessor(BasePreprocessor):
    pass
