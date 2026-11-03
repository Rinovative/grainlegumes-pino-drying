"""Resolve static intra-``src`` imports and classify them against code projections.

Responsibilities:
  - Parse a source module with ``ast`` and resolve its imports and package-alias attribute use to the module files they reach
  - Resolve package aliases through the ``from . import x as name`` lines and lazy name-to-module tables of a package ``__init__``
  - Let a runtime binding of a package name win over a TYPE_CHECKING-only one, and report contradictory runtime bindings of one name
  - Classify every import of every declared projection member as member, package ``__init__``, infrastructure, exempted, cross-task, or unclassified

Design principles:
  - Sources are parsed and never imported or executed
  - TYPE_CHECKING-only references are ignored; every other static reference counts, wherever it appears in the module
  - A package ``__init__`` that a projection only hashes with its package is not scanned as an importer; as an import target it is always classified
  - An import that reaches no file below the source root is not an intra-``src`` import

This module does NOT:
  - Follow dynamic imports whose module name is a runtime value
  - Decide whether a classification is scientifically right; projection owners declare it
"""

from __future__ import annotations

import ast
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from src import common

if TYPE_CHECKING:
    from collections.abc import Iterator

    from src.common.common_code_identity import CodeProjection

Module = tuple[str, ...]
"""Dotted module parts below ``src``; the empty tuple is the ``src`` package itself."""

_PACKAGE_INIT = "__init__.py"
_DYNAMIC_IMPORTERS = frozenset({"import_module", "__import__"})
_SOURCE_PACKAGE = "src"


@dataclass(frozen=True, slots=True)
class ImportReference:
    """One runtime static reference from a source module to another module file."""

    importer: str
    lineno: int
    target: str


@dataclass(frozen=True, slots=True)
class UnclassifiedImport:
    """A member import that is neither a member, an exemption, nor infrastructure."""

    importer: str
    lineno: int
    target: str
    projection: str

    def message(self) -> str:
        """Return the actionable failure line naming importer and target."""
        return (
            f"src/{self.importer}:{self.lineno} imports src/{self.target}, which is not classified for projection "
            f"{self.projection!r}: classify it: add it to the projection or exempt it with a reason."
        )


@dataclass(frozen=True, slots=True)
class ContradictoryAlias:
    """A package ``__init__`` that binds one name to different modules at runtime."""

    initializer: str
    lineno: int
    name: str
    targets: tuple[str, ...]

    def message(self) -> str:
        """Return the actionable failure line naming the initializer, the name and its competing modules."""
        return (
            f"src/{self.initializer}:{self.lineno} binds {self.name!r} to different modules "
            f"({', '.join(f'src/{target}' for target in self.targets)}): contradictory alias: "
            "keep exactly one runtime binding of the name."
        )


@dataclass(frozen=True, slots=True)
class PackageMaps:
    """What a package ``__init__`` exposes: module aliases, re-exports, and the names it binds contradictorily."""

    aliases: dict[str, Module]
    reexports: dict[str, Module]
    contradictions: tuple[ContradictoryAlias, ...]


@dataclass(frozen=True, slots=True)
class _Binding:
    kind: str
    target: Module
    file: str
    lineno: int


class SourceTree:
    """Module resolution below one source root; package ``__init__`` files win over same-named modules."""

    def __init__(self, root: Path) -> None:
        """Bind the tree to the directory that plays the role of the ``src`` package."""
        self.root = root
        self._files: dict[Module, Path | None] = {}
        self._maps: dict[Module, PackageMaps] = {}

    def file_of(self, module: Module) -> Path | None:
        """Return the file that defines ``module``, or ``None`` if the tree has none."""
        if module not in self._files:
            base = self.root.joinpath(*module)
            if (base / _PACKAGE_INIT).is_file():
                candidate = base / _PACKAGE_INIT
            elif module:
                candidate = base.parent / f"{base.name}.py"
            else:
                candidate = base / _PACKAGE_INIT
            self._files[module] = candidate if candidate.is_file() else None
        return self._files[module]

    def is_package(self, module: Module) -> bool:
        """Return whether ``module`` is a package directory with an ``__init__.py``."""
        return (self.root.joinpath(*module) / _PACKAGE_INIT).is_file()

    def relative(self, path: Path) -> str:
        """Return the source-relative POSIX path of a file in the tree."""
        return path.relative_to(self.root).as_posix()

    def package_maps(self, package: Module) -> PackageMaps:
        """Return the alias map, the re-export map and the contradictions of a package ``__init__`` read from its syntax tree.

        The alias map holds ``from . import x as name`` and string-valued module-level dict entries of lazy
        tables. The re-export map holds ``from .x import name``. A name bound at runtime (outside
        TYPE_CHECKING, or in a lazy table) is never overridden by a TYPE_CHECKING-only binding; a name with
        different runtime bindings is left out of both maps and reported as a contradiction.
        """
        if package not in self._maps:
            self._maps[package] = self._read_package_maps(package)
        return self._maps[package]

    def contradictions(self) -> list[ContradictoryAlias]:
        """Return the contradictory runtime bindings found in every package ``__init__`` read so far, in a stable order."""
        found = [contradiction for maps in self._maps.values() for contradiction in maps.contradictions]
        return sorted(found, key=lambda item: (item.initializer, item.lineno, item.name))

    def _read_package_maps(self, package: Module) -> PackageMaps:
        init_file = self.file_of(package)
        if init_file is None:
            return PackageMaps({}, {}, ())
        tree = _parse(init_file)
        runtime: dict[str, list[_Binding]] = defaultdict(list)
        typing_only: dict[str, list[_Binding]] = defaultdict(list)
        for node, type_checking in _walk(tree):
            if not isinstance(node, ast.ImportFrom) or node.level != 1:
                continue
            for imported in node.names:
                kind, target = ("alias", (*package, imported.name)) if node.module is None else ("reexport", (*package, *node.module.split(".")))
                file = self.file_of(target)
                if file is not None:
                    (typing_only if type_checking else runtime)[imported.asname or imported.name].append(
                        _Binding(kind, target, self.relative(file), node.lineno)
                    )
        for key, value, lineno in _string_tables(tree):
            target = (*package, value)
            file = self.file_of(target)
            if file is not None:
                runtime[key].append(_Binding("alias", target, self.relative(file), lineno))
        aliases: dict[str, Module] = {}
        reexports: dict[str, Module] = {}
        contradictions: list[ContradictoryAlias] = []
        for name in sorted({*runtime, *typing_only}):
            bindings = runtime.get(name) or typing_only[name]
            if runtime.get(name) and len({(binding.kind, binding.target) for binding in bindings}) > 1:
                targets = tuple(sorted({binding.file for binding in bindings}))
                contradictions.append(ContradictoryAlias(self.relative(init_file), max(binding.lineno for binding in bindings), name, targets))
                continue
            (aliases if bindings[0].kind == "alias" else reexports)[name] = bindings[0].target
        return PackageMaps(aliases, reexports, tuple(contradictions))


def _parse(path: Path) -> ast.Module:
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def _is_type_checking(test: ast.expr) -> bool:
    if isinstance(test, ast.Name):
        return test.id == "TYPE_CHECKING"
    return isinstance(test, ast.Attribute) and test.attr == "TYPE_CHECKING"


def _walk(node: ast.AST, *, type_checking: bool = False) -> Iterator[tuple[ast.AST, bool]]:
    """Yield every node with whether it lies inside an ``if TYPE_CHECKING:`` body."""
    yield node, type_checking
    if isinstance(node, ast.If) and _is_type_checking(node.test):
        for child in node.body:
            yield from _walk(child, type_checking=True)
        for child in node.orelse:
            yield from _walk(child, type_checking=type_checking)
        return
    for child in ast.iter_child_nodes(node):
        yield from _walk(child, type_checking=type_checking)


def _string_tables(tree: ast.Module) -> Iterator[tuple[str, str, int]]:
    """Yield key, value and line of the string pairs of module-level dict literals such as lazy ``_MODULES`` tables."""
    for node in tree.body:
        if not isinstance(node, ast.Assign) or not isinstance(node.value, ast.Dict):
            continue
        for key, value in zip(node.value.keys, node.value.values, strict=True):
            if isinstance(key, ast.Constant) and isinstance(key.value, str) and isinstance(value, ast.Constant) and isinstance(value.value, str):
                yield key.value, value.value, node.lineno


class _ModuleScan:
    """Collect the static references of one module file, with the TYPE_CHECKING flag of each."""

    def __init__(self, tree: SourceTree, relative: str) -> None:
        self.tree = tree
        self.relative = relative
        self.package: Module = tuple(Path(relative).parts[:-1])
        self.syntax = _parse(tree.root / relative)
        self.references: dict[tuple[str, int], bool] = {}
        self.bindings: dict[str, list[tuple[Module | None, bool]]] = defaultdict(list)
        self._collect_imports()
        self._collect_attribute_uses()
        self._collect_dynamic_imports()
        self._collect_lazy_table()

    def runtime_references(self) -> list[ImportReference]:
        """Return the references that are not TYPE_CHECKING-only, sorted by line."""
        found = [ImportReference(self.relative, lineno, target) for (target, lineno), type_checking in self.references.items() if not type_checking]
        return sorted(found, key=lambda reference: (reference.lineno, reference.target))

    def _reference(self, module: Module, lineno: int, *, type_checking: bool) -> None:
        file = self.tree.file_of(module)
        if file is None:
            return
        key = (self.tree.relative(file), lineno)
        self.references[key] = self.references.get(key, True) and type_checking

    def _from_base(self, node: ast.ImportFrom) -> Module | None:
        parts = tuple(node.module.split(".")) if node.module else ()
        if node.level == 0:
            return parts[1:] if parts[:1] == (_SOURCE_PACKAGE,) else None
        up = node.level - 1
        if up > len(self.package):
            return None
        return (*self.package[: len(self.package) - up], *parts)

    def _collect_imports(self) -> None:
        for node, type_checking in _walk(self.syntax):
            if isinstance(node, ast.Import):
                self._import(node, type_checking=type_checking)
            elif isinstance(node, ast.ImportFrom):
                self._import_from(node, type_checking=type_checking)

    def _import(self, node: ast.Import, *, type_checking: bool) -> None:
        for imported in node.names:
            parts = tuple(imported.name.split("."))
            if parts[0] != _SOURCE_PACKAGE:
                continue
            module = parts[1:]
            self._reference(module, node.lineno, type_checking=type_checking)
            if imported.asname:
                self.bindings[imported.asname].append((module, type_checking))
            else:
                self.bindings[_SOURCE_PACKAGE].append(((), type_checking))

    def _import_from(self, node: ast.ImportFrom, *, type_checking: bool) -> None:
        base = self._from_base(node)
        if base is None or self.tree.file_of(base) is None:
            return
        self._reference(base, node.lineno, type_checking=type_checking)
        maps = self.tree.package_maps(base) if self.tree.is_package(base) else PackageMaps({}, {}, ())
        for imported in node.names:
            name = imported.asname or imported.name
            submodule = (*base, imported.name)
            if self.tree.file_of(submodule) is not None:
                self._reference(submodule, node.lineno, type_checking=type_checking)
                self.bindings[name].append((submodule, type_checking))
            elif imported.name in maps.aliases:
                self._reference(maps.aliases[imported.name], node.lineno, type_checking=type_checking)
                self.bindings[name].append((maps.aliases[imported.name], type_checking))
            else:
                if imported.name in maps.reexports:
                    self._reference(maps.reexports[imported.name], node.lineno, type_checking=type_checking)
                self.bindings[name].append((None, type_checking))

    def _collect_attribute_uses(self) -> None:
        for node, _type_checking in _walk(self.syntax):
            if not isinstance(node, ast.Attribute):
                continue
            attributes: list[str] = []
            current: ast.expr = node
            while isinstance(current, ast.Attribute):
                attributes.append(current.attr)
                current = current.value
            if not isinstance(current, ast.Name) or current.id not in self.bindings:
                continue
            for target, bound_in_type_checking in self.bindings[current.id]:
                if target is not None:
                    self._follow(target, attributes[::-1], node.lineno, type_checking=bound_in_type_checking)

    def _follow(self, start: Module, attributes: list[str], lineno: int, *, type_checking: bool) -> None:
        """Resolve ``alias.attr...`` through package aliases to the module files it reaches."""
        current = start
        for attribute in attributes:
            if not self.tree.is_package(current):
                return
            maps = self.tree.package_maps(current)
            submodule = (*current, attribute)
            if attribute in maps.aliases:
                current = maps.aliases[attribute]
            elif self.tree.file_of(submodule) is not None:
                current = submodule
            else:
                if attribute in maps.reexports:
                    self._reference(maps.reexports[attribute], lineno, type_checking=type_checking)
                return
            self._reference(current, lineno, type_checking=type_checking)

    def _collect_dynamic_imports(self) -> None:
        """Resolve ``import_module("src....")`` calls whose module name is a string literal."""
        for node, type_checking in _walk(self.syntax):
            if not isinstance(node, ast.Call) or not node.args:
                continue
            function = node.func
            name = function.attr if isinstance(function, ast.Attribute) else getattr(function, "id", "")
            argument = node.args[0]
            if name in _DYNAMIC_IMPORTERS and isinstance(argument, ast.Constant) and isinstance(argument.value, str):
                parts = tuple(argument.value.split("."))
                if parts[:1] == (_SOURCE_PACKAGE,):
                    self._reference(parts[1:], node.lineno, type_checking=type_checking)

    def _collect_lazy_table(self) -> None:
        """Count the lazy name-to-module table of a package ``__init__`` as runtime-reachable."""
        if Path(self.relative).name != _PACKAGE_INIT:
            return
        for _key, value, lineno in _string_tables(self.syntax):
            self._reference((*self.package, value), lineno, type_checking=False)


def module_references(tree: SourceTree, relative: str) -> list[ImportReference]:
    """Return the runtime static references from one source-relative module to other ``src`` module files."""
    return _ModuleScan(tree, relative).runtime_references()


def flatten(projection: CodeProjection) -> list[CodeProjection]:
    """Return a projection followed by every projection it includes, recursively."""
    flattened = [projection]
    for included in projection.includes:
        flattened.extend(flatten(included))
    return flattened


def is_cross_task(target: str, markers: tuple[str, ...]) -> bool:
    """Return whether a module path lies in the other task's files or directories by marker."""
    return any(marker in part for part in Path(target).parts for marker in markers)


def is_infrastructure(target: str) -> bool:
    """Return whether a module is declared infrastructure, by module or by directory."""
    if target in common.code_identity.INFRASTRUCTURE_MODULES:
        return True
    directory = Path(target).parent
    return any(directory.is_relative_to(declared) for declared in common.code_identity.INFRASTRUCTURE_DIRECTORIES)


def unclassified_imports(projection: CodeProjection, root: Path) -> list[UnclassifiedImport]:
    """Return every static import of a declared member that the projection and the shared rules leave unclassified.

    A target is classified when it is a member, a package ``__init__.py`` (it only exposes aliases, which
    this resolver follows to their modules), declared infrastructure, an exemption of the projection or of
    a projection it includes, or a cross-task module whose path matches an excluded marker. The package
    initializers a projection only hashes with their members are not scanned as importers.
    """
    parts = flatten(projection)
    members = set(projection.members(root))
    exemptions = {name for part in parts for name in part.exemptions}
    markers = tuple(sorted({marker for part in parts for marker in part.excluded_markers}))
    tree = SourceTree(root)
    found: dict[tuple[str, str], UnclassifiedImport] = {}
    for member in projection.declared_members(root):
        for reference in module_references(tree, member):
            target = reference.target
            if (
                target == member
                or Path(target).name == _PACKAGE_INIT
                or target in members
                or target in exemptions
                or is_infrastructure(target)
                or is_cross_task(target, markers)
            ):
                continue
            found.setdefault((member, target), UnclassifiedImport(member, reference.lineno, target, projection.name))
    return list(found.values())


def contradictory_aliases(projection: CodeProjection, root: Path) -> list[ContradictoryAlias]:
    """Return every package ``__init__`` name with different runtime bindings that a member or a hashed initializer reaches."""
    tree = SourceTree(root)
    for member in projection.members(root):
        module_references(tree, member)
        if Path(member).name == _PACKAGE_INIT:
            tree.package_maps(Path(member).parent.parts)
    return tree.contradictions()
