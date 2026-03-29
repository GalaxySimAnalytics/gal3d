import inspect
import re
import subprocess
import sys
from pathlib import Path
from typing import Type

from _log_utils import logger
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

def _ruff_fix(pyi_path: Path) -> None:
    """Run ruff check --fix and ruff format on the given .pyi file to normalize imports."""
    for cmd in (
        ["ruff", "check", "--fix", "--select", "I", str(pyi_path)],  # isort only
        ["ruff", "format", str(pyi_path)],
    ):
        try:
            subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        except (FileNotFoundError, subprocess.CalledProcessError):
            pass  # ruff not available or failed, skip silently

def _ensure_typing_imports(lines: list[str]) -> None:
    """Ensure Literal and overload are importable from typing.

    Rather than precisely merging existing typing imports, we add individual
    import lines for any missing names. ruff --fix will merge and sort them.
    """
    for name in ("Literal", "overload"):
        # Check if 'name' is already imported from typing
        if any(re.search(rf"\bimport\b.*\b{name}\b", lne) for lne in lines):
            continue
        # Append a bare import line; ruff will merge it later
        insert_at = 0
        for i, l in enumerate(lines):
            if l.startswith(("from ", "import ")):
                insert_at = i + 1
        lines.insert(insert_at, f"from typing import {name}\n")


def _ensure_plugin_imports(lines: list[str], imports: list[str]) -> None:
    """Insert plugin class imports if not already present.

    Import order does NOT matter here — ruff will sort everything after.
    """
    existing = "".join(lines)
    insert_at = 0
    for i, l in enumerate(lines):
        if l.startswith(("from ", "import ")):
            insert_at = i + 1

    for imp in imports:
        # Extract 'ClassName' from 'from x.y import ClassName'
        cls_name = imp.rsplit(" ", 1)[-1]
        if cls_name not in existing:
            lines.insert(insert_at, imp + "\n")
            insert_at += 1

_TYPING_NAMES = frozenset({
    "Union", "Optional", "Tuple", "List", "Dict", "Set", "FrozenSet",
    "Callable", "Iterator", "Generator", "Type", "ClassVar", "Final",
    "Annotated", "Concatenate",
})


def _fix_type_aliases(lines: list[str], src_path: Path) -> None:
    """Fix bare ``Name: TypeAlias`` declarations in the ``.pyi``.

    ``stubgen`` drops the RHS when all referenced types are forward-reference
    strings; this function restores those values by reading the source ``.py``.

    Only single-line ``TypeAlias`` assignments are supported.
    """
    src_text = src_path.read_text(encoding="utf-8")

    # Collect all  NAME: TypeAlias = <expr>  assignments from the source
    alias_re = re.compile(
        r'^([A-Za-z_]\w*)\s*:\s*TypeAlias\s*=\s*(.+)$',
        re.MULTILINE,
    )
    aliases: dict[str, str] = {}
    for m in alias_re.finditer(src_text):
        name, rhs = m.group(1), m.group(2).strip()
        # Strip forward-reference quotes: "ClassName" → ClassName
        rhs = re.sub(r'"([A-Za-z_]\w*)"', r'\1', rhs)
        rhs = re.sub(r"'([A-Za-z_]\w*)'", r'\1', rhs)
        aliases[name] = rhs

    if not aliases:
        return

    # Patch bare 'NAME: TypeAlias' lines (no RHS) in the .pyi
    bare_re = re.compile(r'^(\s*)([A-Za-z_]\w*)\s*:\s*TypeAlias\s*$')
    for i, line in enumerate(lines):
        m = bare_re.match(line)
        if m and m.group(2) in aliases:
            name = m.group(2)
            lines[i] = f'{m.group(1)}{name}: TypeAlias = {aliases[name]}\n'

    # Ensure any typing-module names used in the injected RHS are imported
    all_rhs = " ".join(aliases.values())
    needed = {n for n in _TYPING_NAMES if re.search(rf'\b{n}\b', all_rhs)}
    if not needed:
        return

    existing = "".join(lines)
    insert_at = 0
    for idx, lne in enumerate(lines):
        if lne.startswith(("from ", "import ")):
            insert_at = idx + 1
    for name in sorted(needed):
        if any(re.search(rf"\bimport\b.*\b{name}\b", lne) for lne in lines):
            continue
        lines.insert(insert_at, f"from typing import {name}\n")
        insert_at += 1

def _class_block_range(lines: list[str], cls_name: str) -> tuple[int, int, int]:
    """
    Return (start_index, end_index, body_indent) for class cls_name in given lines.
    end_index points to the first line after the class block (insertion point).
    """
    start = -1
    for i, l in enumerate(lines):
        if not l.startswith("class "):
            continue
        rest = l[len("class "):]
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
    return start, k, body_indent


def _inject_overloads(lines: list[str], mgr_name: str, overloads: list[str]) -> None:
    """Inject overload methods into the manager class body if not already present."""
    if not overloads:
        return
    start, end, body_indent = _class_block_range(lines, mgr_name)
    if start < 0:
        lines += [f"\nclass {mgr_name}:\n"] + overloads
        return

    # If the class is a single-line stub: "class X(...): ...", expand it
    header_line = lines[start]
    after_colon = header_line.split(":", 1)[1] if ":" in header_line else ""
    if after_colon.strip() in ("...", "pass"):
        prefix = header_line.split(":", 1)[0]
        lines[start] = prefix + ":\n"
        lines.insert(start + 1, " " * 4 + "...\n")
        start, end, body_indent = _class_block_range(lines, mgr_name)

    # Filter out overloads already present in the class block
    class_block = "".join(lines[start:end])
    todo = []
    for chunk in overloads:
        if "Literal[" in chunk:
            # Named overload: deduplicate via the Literal value
            lit = chunk.split("Literal[", 1)[1].split("]", 1)[0]
            if lit in class_block:
                continue
        else:
            # Fallback (str) overload: deduplicate via the def-line signature
            def_line = next(
                (ln.strip() for ln in chunk.splitlines() if ln.strip().startswith("def ")),
                None,
            )
            if def_line and def_line in class_block:
                continue
        todo.append(chunk)
    if not todo:
        return

    # Remove placeholder "..." body line before inserting methods
    body_start = start + 1
    if body_start < len(lines) and lines[body_start].strip() == "...":
        del lines[body_start]
        end -= 1

    # Ensure a blank line before injected methods if body not empty
    insert_lines: list[str] = []
    if any(lines[i].strip() for i in range(start + 1, end)):
        insert_lines.append("\n")
    insert_lines += todo

    lines[end:end] = insert_lines


def _run_stubgen_for_module(module_name: str, out_root: Path) -> Path:
    """
    Run stubgen for a module and return the path to the generated .pyi.
    """
    out_root.mkdir(parents=True, exist_ok=True)

    tried: list[list[str]] = []
    for cmd in (
        ["stubgen", "-m", module_name, "-o", str(out_root), "--include-docstrings"],
        [sys.executable, "-m", "mypy.stubgen", "-m", module_name, "-o", str(out_root), "--include-docstrings"],
    ):
        tried.append(cmd)
        try:
            subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            break
        except (FileNotFoundError, subprocess.CalledProcessError):
            continue
    else:
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

    parts = module_name.split(".")
    rel_dir = out_root.joinpath(*parts[:-1]) if len(parts) > 1 else out_root
    pyi_path = rel_dir / (parts[-1] + ".pyi")
    if not pyi_path.exists():
        pkg_init = rel_dir / "__init__.pyi"
        if pkg_init.exists():
            pyi_path = pkg_init
        else:
            raise FileNotFoundError(f"Generated stub not found: {pyi_path}")
    return pyi_path


def gen_manager_pyi(manager: Type[PluginManager]) -> None:
    # Discover plugins — sorted for deterministic overload order
    plugins = sorted(manager.available_plugins())
    plugin_list = ", ".join(plugins) if plugins else "(none)"
    logger.step(f"{manager.__name__}  [{len(plugins)} plugin(s): {plugin_list}]")

    module_name = manager.__module__
    src_file = inspect.getsourcefile(manager)
    if not src_file:
        raise RuntimeError(f"Cannot locate source for {manager}")
    src_path = Path(src_file)
    top_pkg = module_name.split(".")[0]
    out_root = _find_package_root(src_path, top_pkg)

    pyi_path = _run_stubgen_for_module(module_name, out_root)
    logger.item(f"Stub: {pyi_path}")

    # Build plugin import lines and overloads (sorted = deterministic)
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
    # Fallback overload always last
    overloads_get.append(
        "    @overload\n"
        "    @classmethod\n"
        f"    def get_plugin(cls, name: str) -> type[{manager._base_class.__name__}]: ...\n"
    )

    text = pyi_path.read_text(encoding="utf-8")
    lines = text.splitlines(keepends=True)

    _ensure_typing_imports(lines)
    _fix_type_aliases(lines, src_path)
    _ensure_plugin_imports(lines, imports)
    _inject_overloads(lines, manager.__name__, overloads_get)

    pyi_path.write_text("".join(lines), encoding="utf-8")

    # Delegate import sorting/formatting to ruff — this makes the output
    # identical to what pre-commit ruff would produce, so git sees no diff.
    _ruff_fix(pyi_path)

    logger.success(f"Patched {pyi_path}")


def main() -> None:
    logger.header("gal3d plugin stub generator")
    PluginManagerRegistry.all_managers()
    mgrs = [
        mgr
        for _, mgr in PluginManagerRegistry._managers.items()
        if mgr is not PluginManager
    ]
    if not mgrs:
        logger.warning("No plugin managers registered.")
        return
    logger.step(f"{len(mgrs)} plugin manager(s) to process")
    for mgr in mgrs:
        gen_manager_pyi(mgr)
    logger.summary(f"All {len(mgrs)} manager stub(s) patched")

if __name__ == "__main__":
    main()