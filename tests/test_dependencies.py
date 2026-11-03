import ast
from pathlib import Path
import tomllib

from packaging.requirements import Requirement

ROOT = Path(__file__).resolve().parents[1]


def test_web_requires_pydantic_v2_locally_and_on_modal():
    project = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]
    requirements = [Requirement(value) for value in project["optional-dependencies"]["web"]]
    pydantic = next((r for r in requirements if r.name == "pydantic"), None)
    assert pydantic is not None, "The app uses Pydantic v2 APIs directly"
    assert "1.10.27" not in pydantic.specifier
    assert "2.14.2" in pydantic.specifier
    assert "3.0.0" not in pydantic.specifier

    module = ast.parse((ROOT / "webapp/modal_app.py").read_text())
    deployed = [Requirement(argument.value) for node in ast.walk(module)
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and node.func.attr == "pip_install" for argument in node.args
                if isinstance(argument, ast.Constant) and isinstance(argument.value, str)]
    assert any(r.name == "pydantic" and str(r.specifier) == str(pydantic.specifier) for r in deployed)
