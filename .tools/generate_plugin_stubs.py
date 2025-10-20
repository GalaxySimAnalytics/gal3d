import inspect
import textwrap
from pathlib import Path
from typing import Type
import typing as t
import types as _types

from gal3d.plugin import PluginManager, PluginManagerRegistry

HEADER = "# Auto-generated. Do not edit by hand.\n"

def _format_type(ann: t.Any) -> str:
    """Return a stub-friendly string for a type annotation, using fully-qualified names when needed."""
    if ann is inspect._empty:
        return ""
    if ann is t.Any:
        return "Any"
    if ann is None or ann is type(None):
        return "None"

    origin = t.get_origin(ann)
    args = t.get_args(ann)

    # Union types (A | B)
    if origin in (t.Union, _types.UnionType):
        return " | ".join(_format_type(a) for a in args)

    # Literal values
    if origin is t.Literal:
        return f"Literal[{', '.join(repr(a) for a in args)}]"

    # Builtin generics
    name_map = {list: "list", dict: "dict", tuple: "tuple", set: "set", frozenset: "frozenset", type: "type"}
    if origin in name_map:
        if not args:
            return name_map[origin]
        return f"{name_map[origin]}[{', '.join(_format_type(a) for a in args)}]"

    # typing constructs (e.g., Sequence[T], Mapping[K, V])
    if origin is not None:
        mod = getattr(origin, "__module__", "")
        qual = getattr(origin, "__qualname__", str(origin))
        prefix = "" if mod == "builtins" else f"{mod}."
        if args:
            return f"{prefix}{qual}[{', '.join(_format_type(a) for a in args)}]"
        return f"{prefix}{qual}"

    # ForwardRef or classes
    if isinstance(ann, t.ForwardRef):
        return ann.__forward_arg__

    mod = getattr(ann, "__module__", "")
    qual = getattr(ann, "__qualname__", None)
    if qual:
        if mod == "builtins":
            return qual
        return f"{mod}.{qual}"

    # Fallback
    return repr(ann)

def _format_default(val: t.Any) -> str:
    try:
        return repr(val)
    except Exception:
        return "..."  # fallback for unserializable defaults

def _format_docstring(doc: str, indent: str) -> list[str]:
    if not doc:
        return []
    # Keep the doc readable in .pyi
    lines = doc.splitlines()
    out = [f'{indent}"""']
    out.extend(f"{indent}{line}" for line in lines)
    out.append(f'{indent}"""')
    return out

def _build_func_stub(name: str, obj: t.Any, indent: str = "    ") -> list[str]:
    """Build a function/method stub with decorators and docstring."""
    dec_lines: list[str] = []
    func = obj
    is_cm = isinstance(obj, classmethod)
    is_sm = isinstance(obj, staticmethod)
    if is_cm or is_sm:
        func = obj.__func__  # unwrap

    # Decorators
    if getattr(func, "__isabstractmethod__", False):
        dec_lines.append(indent + "@abstractmethod")
    if is_cm:
        dec_lines.append(indent + "@classmethod")
    if is_sm:
        dec_lines.append(indent + "@staticmethod")

    sig = inspect.signature(func)

    parts: list[str] = []
    need_kwonly_marker = True
    for p in sig.parameters.values():
        p_txt = p.name
        if p.kind is p.VAR_POSITIONAL:
            p_txt = "*" + p_txt
        elif p.kind is p.VAR_KEYWORD:
            p_txt = "**" + p_txt
        elif p.kind is p.KEYWORD_ONLY and need_kwonly_marker:
            # Insert * to start keyword-only section
            parts.append("*")
            need_kwonly_marker = False

        if p.annotation is not inspect._empty:
            p_txt += f": {_format_type(p.annotation)}"
        if p.default is not inspect._empty:
            p_txt += f" = {_format_default(p.default)}"
        parts.append(p_txt)

    params_str = ", ".join(parts)
    ret_str = ""
    if sig.return_annotation is not inspect._empty:
        ret_str = f" -> {_format_type(sig.return_annotation)}"

    header = indent + f"def {func.__name__}({params_str}){ret_str}:"

    # Docstring
    raw_doc = inspect.getdoc(func) or func.__doc__ or ""
    body: list[str] = []
    body.extend(_format_docstring(raw_doc, indent + "    "))
    body.append(indent + "    ...")

    return dec_lines + [header] + body

def _gen_base_stub(base: type) -> tuple[list[str], bool]:
    """Generate class stub lines for the base class with docstrings, return (lines, needs_abstractmethod_import)."""
    base_name = base.__name__
    lines: list[str] = []
    needs_abstract = False

    lines.append(f"class {base_name}(_{base_name}):")
    class_doc = inspect.getdoc(base) or base.__doc__ or ""
    body_lines: list[str] = []
    body_lines.extend(_format_docstring(class_doc, "    "))

    # Only methods defined on this class (not inherited)
    for name, obj in base.__dict__.items():
        if name.startswith("__") and name.endswith("__"):
            continue
        if not (callable(obj) or isinstance(obj, (classmethod, staticmethod))):
            continue
        if getattr(getattr(obj, "__func__", obj), "__isabstractmethod__", False):
            needs_abstract = True
        if body_lines and body_lines[-1] != "":
            body_lines.append("")  # blank line between methods
        body_lines.extend(_build_func_stub(name, obj, indent="    "))

    if not body_lines:
        body_lines.append("    ...")

    lines.extend(body_lines)
    lines.append("")  # trailing blank line after class
    return lines, needs_abstract

def gen_manager_pyi(manager: Type[PluginManager], out_dir: Path) -> None:
    mgr_name = manager.__name__
    module = manager.__module__

    plugins = manager.available_plugins()

    lines: list[str] = []
    lines.append(HEADER)

    # typing imports
    typing_imports = ["Any", "Literal", "overload"]
    # Will add abstractmethod if needed later
    lines.append(f"from typing import {', '.join(typing_imports)}\n")
    lines.append(f"from {module} import {mgr_name} as _{mgr_name}\n")

    base = manager._base_class
    base_name = base.__name__
    lines.append(f"from {base.__module__} import {base_name} as _{base_name}\n")

    imports: list[str] = []
    overloads_get: list[str] = []

    for pname in plugins:
        cls = manager.get_plugin(pname)
        cls_name = cls.__name__
        imports.append(f"from {cls.__module__} import {cls_name}")
        overloads_get.append(
            "    @overload\n"
            f"    @classmethod\n"
            f"    def get_plugin(cls, name: Literal[\"{pname}\"]) -> type[{cls_name}]: ...\n"
        )

    lines.extend(i + "\n" for i in sorted(set(imports)))

    # Base class stub with docs and typed method stubs
    base_stub_lines, needs_abstract = _gen_base_stub(base)
    if needs_abstract:
        lines.insert(2, "from abc import abstractmethod\n")  # after HEADER
    lines.append("\n".join(base_stub_lines))

    # Manager class with overloads
    lines.append(f"class {mgr_name}(_{mgr_name}):\n")
    lines.extend(overloads_get)

    src = inspect.getsourcefile(manager)
    assert src
    out_path = Path(src).with_suffix(".pyi")
    out_path.write_text("".join(lines), encoding="utf-8")
    print(f"Wrote {out_path}")

def main():
    PluginManagerRegistry.all_managers()
    for _, mgr in PluginManagerRegistry._managers.items():
        if mgr is PluginManager:
            continue
        gen_manager_pyi(mgr, Path("."))

if __name__ == "__main__":
    main()