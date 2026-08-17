import re
import tomllib
from pathlib import Path


def _pyproject():
    return tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))


def test_torch_install_extras_include_cuda_and_rocm():
    config = _pyproject()
    extras = config["project"]["optional-dependencies"]
    sources = config["tool"]["uv"]["sources"]["torch"]
    all_sources = config["tool"]["uv"]["sources"]

    assert "cuda" in extras
    assert "rocm" in extras
    assert any(source.get("extra") == "cuda" and source.get("index") == "pytorch-cu128" for source in sources)
    assert any(source.get("extra") == "rocm" and source.get("index") == "pytorch-rocm" for source in sources)

    # Both triton spellings must point at the ROCm index: torch <=2.10 depends on
    # pytorch-triton-rocm, torch >=2.11 on triton-rocm, and the index is explicit=true
    # so an unmapped name cannot resolve from anywhere.
    for name in ("pytorch-triton-rocm", "triton-rocm"):
        assert name in all_sources, f"{name} has no source mapping"
        assert any(source.get("index") == "pytorch-rocm" for source in all_sources[name])


def test_sources_with_extra_are_declared_in_that_extra():
    """uv rejects a source carrying `extra = X` unless the package is also listed in
    project.optional-dependencies[X]. Transitive deps (triton) must therefore be mapped
    without an extra. Getting this wrong fails at `uv sync`, not at import time.
    """
    config = _pyproject()
    extras = config["project"]["optional-dependencies"]

    for package, source_list in config["tool"]["uv"]["sources"].items():
        for source in source_list:
            extra = source.get("extra")
            if extra is None:
                continue
            listed = extras.get(extra, [])
            assert any(req.split(">")[0].split("=")[0].split("[")[0].strip() == package for req in listed), (
                f"source for {package!r} declares extra={extra!r} but {package!r} "
                f"is not in optional-dependencies[{extra!r}]={listed}"
            )


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


def test_rocm_extra_declares_triton_directly():
    """triton must be a direct dependency of the rocm extra, not left to torch.

    uv only consults [tool.uv.sources] for the project's direct dependencies. As a
    transitive dependency of torch, triton-rocm never reaches the explicit=true ROCm
    index, and `uv sync --extra rocm` fails with "no version of triton-rocm==3.6.0"
    even though the wheel exists in that index.
    """
    config = _pyproject()
    rocm = config["project"]["optional-dependencies"]["rocm"]
    names = [re.split(r"[<>=;\[ ]", req.strip())[0] for req in rocm]

    assert "triton-rocm" in names, f"rocm extra must list triton-rocm directly, got {rocm}"
    assert any(s.get("index") == "pytorch-rocm" for s in config["tool"]["uv"]["sources"]["triton-rocm"])
