"""Raw .ipynb documents carry base64 image outputs that teach nothing.

code_jupyter is 15.5% of the pretraining corpus by characters, and most of those
characters are output blobs rather than code.
"""
from mesosfer.data.dataset import strip_notebook_blobs

NOTEBOOK = '''{
 "cells": [
  {"cell_type": "code",
   "source": ["import torch\\n", "model = torch.nn.Linear(4, 2)\\n"],
   "outputs": [
    {"output_type": "display_data",
     "data": {"image/png": "%s", "text/plain": ["<Figure size 640x480>"]}}
   ]}
 ],
 "nbformat": 4
}''' % ("iVBORw0KGgoAAAANSUhEUg" * 40)


def test_base64_payload_is_replaced():
    out = strip_notebook_blobs(NOTEBOOK)

    assert "iVBORw0KGgoAAAANSUhEUg" not in out
    assert '"image/png": "<stripped>"' in out
    assert len(out) < len(NOTEBOOK) / 2


def test_code_and_structure_survive():
    out = strip_notebook_blobs(NOTEBOOK)

    assert "import torch" in out
    assert "torch.nn.Linear(4, 2)" in out
    assert '"cell_type": "code"' in out
    assert "<Figure size 640x480>" in out, "non-base64 outputs are still useful context"


def test_ordinary_documents_pass_through_untouched():
    """The substring guard must not rewrite source files that merely contain base64."""
    plain = 'key = "' + "QUJDREVGR0hJSktMTU5PUFFSU1RVVldYWVo=" * 20 + '"\nprint(key)\n'

    assert strip_notebook_blobs(plain) is plain
    assert strip_notebook_blobs("def f():\n    return 1\n") == "def f():\n    return 1\n"
