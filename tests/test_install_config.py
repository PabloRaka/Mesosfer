import tomllib
from pathlib import Path


def _pyproject():
    return tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))


def test_torch_install_extras_include_cuda_and_rocm():
    config = _pyproject()
    extras = config["project"]["optional-dependencies"]
    sources = config["tool"]["uv"]["sources"]["torch"]
    triton_sources = config["tool"]["uv"]["sources"]["pytorch-triton-rocm"]

    assert "cuda" in extras
    assert "rocm" in extras
    assert "pytorch-triton-rocm>=3.1.0" in extras["rocm"]
    assert any(source.get("extra") == "cuda" and source.get("index") == "pytorch-cu128" for source in sources)
    assert any(source.get("extra") == "rocm" and source.get("index") == "pytorch-rocm" for source in sources)
    assert any(source.get("extra") == "rocm" and source.get("index") == "pytorch-rocm" for source in triton_sources)


def test_declared_index_names_all_exist():
    """Every index referenced by [tool.uv.sources] must be declared in [[tool.uv.index]].

    A rename that updates only one side makes `uv sync` fail with an unresolved
    index name, which is easy to miss until someone installs on that backend.
    """
    config = _pyproject()
    declared = {entry["name"] for entry in config["tool"]["uv"]["index"]}
    referenced = set()
    for source_list in config["tool"]["uv"]["sources"].values():
        referenced |= {source["index"] for source in source_list if "index" in source}

    assert referenced <= declared, f"sources point at undeclared index: {referenced - declared}"


def test_torch_install_extras_conflict_with_each_other():
    conflicts = _pyproject()["tool"]["uv"]["conflicts"]

    assert [
        {"extra": "cpu"},
        {"extra": "cuda"},
        {"extra": "gpu"},
        {"extra": "rocm"},
    ] in conflicts
