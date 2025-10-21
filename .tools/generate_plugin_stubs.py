import inspect
import subprocess
import sys
from pathlib import Path
from typing import Type

from gal3d.plugin import PluginManager, PluginManagerRegistry


def _find_package_root(file_path: Path, top_pkg: str) -> Path:
    """
    Find the directory to pass to stubgen -o so that it recreates 'top_pkg/.../module.pyi'
    next to the original sources (e.g. src).
    """
    p = file_path.parent
    while p != p.parent:
        if (p / top_pkg).exists():
            return p
        p = p.parent
    # Fallback: place into the directory of the module file (no package tree rebuild)
    return file_path.parent


def _ensure_typing_imports(lines: list[str]) -> None:
    """Ensure 'from typing import Literal, overload' exists."""
    needed = "from typing import Literal, overload\n"
    if any(l.startswith("from typing import") and "Literal" in l and "overload" in l for l in lines):
        return
    # insert after the last import block
    insert_at = 0
    for i, l in enumerate(lines):
        if l.startswith(("from ", "import ")):
            insert_at = i + 1
    lines.insert(insert_at, needed)


def _ensure_plugin_imports(lines: list[str], imports: list[str]) -> None:
    """Insert plugin class imports if not already present."""
    add = [imp for imp in sorted(set(imports)) if all(imp not in l for l in lines)]
    if not add:
        return
    # insert after the last import block
    insert_at = 0
    for i, l in enumerate(lines):
        if l.startswith(("from ", "import ")):
            insert_at = i + 1
    for imp in add:
        lines.insert(insert_at, imp + "\n")
        insert_at += 1


def _class_block_range(lines: list[str], cls_name: str) -> tuple[int, int, int]:
    """
    Return (start_index, end_index, body_indent) for class cls_name in given lines.
    end_index points to the first line after the class block (insertion point).
    """
    start = -1
    for i, l in enumerate(lines):
        # Only match top-level classes
        if not l.startswith("class "):
            continue
        rest = l[len("class "):]
        # class Name(...): or class Name:
        name = rest.split("(", 1)[0].split(":", 1)[0].strip()
        if name == cls_name:
            start = i
            break
    if start < 0:
        return -1, -1, 4
    # find body indent
    j = start + 1
    body_indent = 4
    while j < len(lines) and lines[j].strip() == "":
        j += 1
    if j < len(lines):
        body_indent = len(lines[j]) - len(lines[j].lstrip(" "))
    # find end (dedent)
    k = j
    while k < len(lines):
        s = lines[k].strip()
        if s != "":
            indent = len(lines[k]) - len(lines[k].lstrip(" "))
            if indent < body_indent:
                break
        k += 1
    end = k
    return start, end, body_indent


def _inject_overloads(lines: list[str], mgr_name: str, overloads: list[str]) -> None:
    """Inject overload methods into the manager class body if not already present."""
    if not overloads:
        return
    start, end, body_indent = _class_block_range(lines, mgr_name)
    if start < 0:
        # No manager class in stub? Append a partial class stub.
        lines += [
            f"\nclass {mgr_name}:\n",
        ] + [ol for ol in overloads]
        return

    # If the class is a single-line stub: "class X(...): ...", expand it to a block
    header_line = lines[start]
    after_colon = header_line.split(":", 1)[1] if ":" in header_line else ""
    if after_colon.strip() in ("...", "pass"):
        # Turn into:
        # class X(...):
        #     ...
        prefix = header_line.split(":", 1)[0]
        lines[start] = prefix + ":\n"
        # Prefer a 4-space indent for class body
        placeholder = " " * 4 + "...\n"
        lines.insert(start + 1, placeholder)
        # Recompute class block after expansion
        start, end, body_indent = _class_block_range(lines, mgr_name)

    # Filter out overloads already present (by plugin Literal[...] marker)
    class_block = "".join(lines[start:end])
    todo = []
    for chunk in overloads:
        # Each chunk contains 'Literal["{pname}"]'
        if "Literal[" in chunk and chunk in class_block:
            continue
        if "Literal[" in chunk:
            lit = chunk.split("Literal[", 1)[1].split("]", 1)[0]
            if lit in class_block:
                continue
        todo.append(chunk)
    if not todo:
        return

    # If the class body only contains a placeholder "...", remove it before inserting methods
    body_start = start + 1
    if body_start < len(lines) and lines[body_start].strip() == "...":
        del lines[body_start]
        # Adjust end because we removed one line
        end -= 1

    # Ensure a blank line before injected methods if body not empty
    insert_lines = []
    if any(lines[i].strip() for i in range(start + 1, end)):
        insert_lines.append("\n")
    insert_lines += todo

    lines[end:end] = insert_lines


def _run_stubgen_for_module(module_name: str, out_root: Path) -> Path:
    """
    Run stubgen for a module and return the path to the generated .pyi.
    Try 'stubgen' CLI first, then in-process API, then 'python -m mypy.stubgen'.
    """
    out_root.mkdir(parents=True, exist_ok=True)

    # 1) Try console script 'stubgen'
    tried: list[list[str]] = []
    for cmd in (
        ["stubgen", "-m", module_name, "-o", str(out_root), "--include-docstrings"], #  "--include-private"
        [sys.executable, "-m", "mypy.stubgen", "-m", module_name, "-o", str(out_root), "--include-docstrings"],
    ):
        tried.append(cmd)
        try:
            subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            break
        except (FileNotFoundError, subprocess.CalledProcessError):
            continue
    else:
        # 2) Try in-process API as last robust option
        try:
            import mypy.stubgen as _stubgen  # type: ignore
            try:
                _stubgen.main(["-m", module_name, "-o", str(out_root), "--include-docstrings"])
            except SystemExit as e:
                if e.code not in (0, None):
                    raise RuntimeError(f"stubgen failed for {module_name} with exit code {e.code}")
        except Exception as e:
            cmds = "\n".join(" ".join(c) for c in tried)
            raise RuntimeError(
                f"stubgen not available or failed for {module_name}.\n"
                f"Tried commands:\n{cmds}\n"
                f"Hint: pip install -U mypy"
            ) from e

    # Resolve output file path
    parts = module_name.split(".")
    rel_dir = out_root.joinpath(*parts[:-1]) if len(parts) > 1 else out_root
    pyi_path = rel_dir / (parts[-1] + ".pyi")
    if not pyi_path.exists():
        # If stubgen emitted package stub, expect __init__.pyi
        pkg_init = rel_dir / "__init__.pyi"
        if pkg_init.exists():
            pyi_path = pkg_init
        else:
            raise FileNotFoundError(f"Generated stub not found: {pyi_path}")
    return pyi_path


def gen_manager_pyi(manager: Type[PluginManager]) -> None:
    # Discover plugins first
    plugins = manager.available_plugins()

    # Locate source module and output root
    module_name = manager.__module__
    src_file = inspect.getsourcefile(manager)
    if not src_file:
        raise RuntimeError(f"Cannot locate source for {manager}")
    src_path = Path(src_file)
    top_pkg = module_name.split(".")[0]
    out_root = _find_package_root(src_path, top_pkg)

    # Generate base .pyi via stubgen
    pyi_path = _run_stubgen_for_module(module_name, out_root)

    # Build plugin import lines and overloads
    imports: list[str] = []
    overloads_get: list[str] = []
    for pname in plugins:
        cls = manager.get_plugin(pname)
        cls_name = cls.__name__
        imports.append(f"from {cls.__module__} import {cls_name}")
        overloads_get.append(
            "    @overload\n"
            "    @classmethod\n"
            f"    def get_plugin(cls, name: Literal[\"{pname}\"]) -> type[{cls_name}]: ...\n"
        )
    overloads_get.append(
        "    @overload\n"
        "    @classmethod\n"
        f"    def get_plugin(cls, name: str) -> type[{manager._base_class.__name__}]: ...\n"
    )

    # Read, patch, write
    text = pyi_path.read_text(encoding="utf-8")
    lines = text.splitlines(keepends=True)

    # Ensure typing imports (Literal, overload)
    _ensure_typing_imports(lines)
    # Ensure plugin imports
    _ensure_plugin_imports(lines, imports)
    # Inject overloads in the manager class
    mgr_name = manager.__name__
    _inject_overloads(lines, mgr_name, overloads_get)

    pyi_path.write_text("".join(lines), encoding="utf-8")
    print(f"Patched {pyi_path}")


def main() -> None:
    PluginManagerRegistry.all_managers()
    for _, mgr in PluginManagerRegistry._managers.items():
        if mgr is PluginManager:
            continue
        gen_manager_pyi(mgr)

if __name__ == "__main__":
    main()