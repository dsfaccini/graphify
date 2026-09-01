"""Bounded internal helpers for NetworkX node-link JSON files.

The global graph uses the node-link format but must not need to materialize the
whole persisted graph for a one-repository update.  This module deliberately
has no CLI surface: it validates one stable file identity, makes independent
header/node/link passes, and emits a replacement file through the project's
existing atomic-write primitive.
"""

from __future__ import annotations

import json
import math
from collections.abc import Callable, Iterator, Mapping, Set
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import BinaryIO, Literal, TextIO, TypeAlias

import ijson
from ijson.common import JSONError, ObjectBuilder

from graphify.paths import write_callback_atomic


JsonScalar: TypeAlias = str | int | float | bool | None
JsonValue: TypeAlias = JsonScalar | list["JsonValue"] | dict[str, "JsonValue"]
JsonObject: TypeAlias = dict[str, JsonValue]
# Tuple identifiers are emitted in node-link JSON as lists.  The tuple members
# are objects here because a node id accepts recursively nested tuples, exactly
# as NetworkX's ``_to_tuple`` helper does for node records.
NodeIdentifier: TypeAlias = JsonScalar | tuple[object, ...]
LinkKey: TypeAlias = Literal["links", "edges"]
_MAX_NONFINITE_TOKEN_BYTES = len(b"-Infinity")


class NodeLinkStreamError(ValueError):
    """Base error for malformed or unstable node-link inputs."""


class NodeLinkFormatError(NodeLinkStreamError):
    """The source is not a supported node-link JSON object."""


class NonFiniteJsonError(NodeLinkFormatError):
    """The source uses non-standard JSON NaN or Infinity tokens."""


class NodeLinkChangedError(NodeLinkStreamError):
    """The input changed between (or during) streaming passes."""


class NodeLinkEmissionError(NodeLinkStreamError):
    """A callback tried to emit records outside node-link section order."""


@dataclass(frozen=True)
class FileIdentity:
    """The stat fields used to reject a source changed during a multi-pass read."""

    device: int
    inode: int
    size: int
    mtime_ns: int

    @classmethod
    def capture(cls, path: Path) -> FileIdentity:
        status = path.stat()
        return cls(status.st_dev, status.st_ino, status.st_size, status.st_mtime_ns)


@dataclass(frozen=True)
class NodeLinkHeader:
    """The topology fields retained while nodes and links stream independently."""

    directed: bool
    multigraph: bool
    graph: JsonObject
    link_key: LinkKey


@dataclass(frozen=True)
class EdgeIdentity:
    """NetworkX-equivalent identity for a simple or keyed node-link edge."""

    endpoints: tuple[NodeIdentifier, NodeIdentifier] | frozenset[NodeIdentifier]
    key: NodeIdentifier | None


def _backend(backend_name: str | None):
    if backend_name is None:
        return ijson
    try:
        return ijson.get_backend(backend_name)
    except ImportError as exc:
        raise NodeLinkFormatError(f"ijson backend {backend_name!r} is unavailable") from exc


def _iter_events(handle: BinaryIO, backend_name: str | None) -> Iterator[tuple[str, str, object]]:
    parser = _backend(backend_name).parse(handle, use_float=False)
    for prefix, event, value in parser:
        yield prefix, event, value


def _reject_nonfinite_tokens(path: Path) -> None:
    """Reject JSON extensions that would make persisted topology non-portable.

    ``NaN`` and ``Infinity`` are accepted by some Python JSON encoders but are
    not JSON.  Detect them outside strings before selecting an ijson backend so
    every backend fails closed with the same actionable error.
    """

    in_string = False
    escaped = False
    token = bytearray()
    token_too_long = False

    def finish_token() -> None:
        nonlocal token_too_long
        if not token_too_long and bytes(token) in {b"NaN", b"Infinity", b"-Infinity"}:
            raise NonFiniteJsonError(
                f"{path} contains a non-standard JSON number (NaN or Infinity); refusing to read it"
            )
        token.clear()
        token_too_long = False

    with path.open("rb") as handle:
        while chunk := handle.read(65_536):
            for byte in chunk:
                if in_string:
                    if escaped:
                        escaped = False
                    elif byte == ord("\\"):
                        escaped = True
                    elif byte == ord('"'):
                        in_string = False
                    continue

                if byte == ord('"'):
                    finish_token()
                    in_string = True
                elif ord("A") <= byte <= ord("Z") or ord("a") <= byte <= ord("z") or byte == ord("-"):
                    # The longest token we care about is ``-Infinity``.  Keep
                    # a bounded candidate even when a malformed input contains
                    # millions of bare letters before the JSON parser rejects
                    # it on the later pass.
                    if len(token) < _MAX_NONFINITE_TOKEN_BYTES:
                        token.append(byte)
                    else:
                        token_too_long = True
                else:
                    finish_token()
    finish_token()


def _normalize_json(value: object) -> JsonValue:
    """Convert ijson values into portable JSON values without narrowing ints."""

    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, Decimal):
        try:
            normalized = float(value)
        except (OverflowError, ValueError) as exc:
            raise NonFiniteJsonError("a JSON decimal cannot be represented as a finite float") from exc
        if not math.isfinite(normalized):
            raise NonFiniteJsonError("a JSON decimal cannot be represented as a finite float")
        return normalized
    if isinstance(value, float):
        if not math.isfinite(value):
            raise NonFiniteJsonError("NaN and Infinity are not accepted in node-link JSON")
        return value
    if isinstance(value, list):
        return [_normalize_json(item) for item in value]
    if isinstance(value, dict):
        normalized_object: JsonObject = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise NodeLinkFormatError("node-link JSON object keys must be strings")
            normalized_object[key] = _normalize_json(item)
        return normalized_object
    raise NodeLinkFormatError(f"unsupported JSON value type: {type(value).__name__}")


def _json_object(value: object, context: str) -> JsonObject:
    normalized = _normalize_json(value)
    if not isinstance(normalized, dict):
        raise NodeLinkFormatError(f"{context} must be a JSON object")
    return normalized


def _validate_hashable_identifier(value: object, field: str) -> NodeIdentifier:
    """Return a supported NetworkX identifier or raise before it reaches a graph.

    NetworkX accepts scalar node-link ids and tuples created from JSON lists.
    A nested list or object can still be present inside a tuple produced from an
    endpoint, so use ``hash`` as the final validity check instead of assuming
    that a tuple is sufficient.
    """

    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if isinstance(value, tuple):
        try:
            hash(value)
        except TypeError as exc:
            raise NodeLinkFormatError(f"{field} must be a hashable node identifier") from exc
        return value
    raise NodeLinkFormatError(f"{field} must be a hashable node identifier")


def _recursive_node_tuple(value: JsonValue) -> object:
    """Match NetworkX ``_to_tuple`` for node-record ids without changing data."""

    if isinstance(value, list):
        return tuple(_recursive_node_tuple(item) for item in value)
    return value


def _node_identifier(value: JsonValue, field: str) -> NodeIdentifier:
    return _validate_hashable_identifier(_recursive_node_tuple(value), field)


def _endpoint_identifier(value: JsonValue, field: str) -> NodeIdentifier:
    """Match NetworkX endpoint conversion: tupleize one JSON-list level only.

    ``node_link_graph`` recursively tupleizes *node* ids, but it only calls
    ``tuple(...)`` for list-valued edge endpoints.  Keep that distinction and
    reject an endpoint whose nested list remains unhashable before graph
    mutation would encounter it.
    """

    normalized: object = tuple(value) if isinstance(value, list) else value
    return _validate_hashable_identifier(normalized, field)


def node_identity(node: Mapping[str, JsonValue]) -> NodeIdentifier:
    """Return the node-link ``id`` after enforcing NetworkX-hashable JSON IDs."""

    if "id" not in node:
        raise NodeLinkFormatError("node-link node is missing 'id'")
    return _node_identifier(node["id"], "node id")


def node_is_target(node: Mapping[str, JsonValue], target_repo: str) -> bool:
    """Whether a node belongs to the target repository by its authoritative attr."""

    return node.get("repo") == target_repo


def _edge_endpoints(edge: Mapping[str, JsonValue]) -> tuple[NodeIdentifier, NodeIdentifier]:
    if "source" not in edge or "target" not in edge:
        raise NodeLinkFormatError("node-link edge is missing 'source' or 'target'")
    return (
        _endpoint_identifier(edge["source"], "edge source"),
        _endpoint_identifier(edge["target"], "edge target"),
    )


def edge_identity(edge: Mapping[str, JsonValue], header: NodeLinkHeader) -> EdgeIdentity:
    """Return the identity NetworkX uses when a streamed edge is added."""

    source, target = _edge_endpoints(edge)
    endpoints: tuple[NodeIdentifier, NodeIdentifier] | frozenset[NodeIdentifier]
    endpoints = (source, target) if header.directed else frozenset((source, target))
    key: NodeIdentifier | None = None
    if header.multigraph:
        if "key" not in edge:
            raise NodeLinkFormatError("node-link multigraph edge is missing 'key'")
        key = _validate_hashable_identifier(edge["key"], "edge key")
    return EdgeIdentity(endpoints=endpoints, key=key)


def edge_touches_target(
    edge: Mapping[str, JsonValue], target_nodes: Set[NodeIdentifier]
) -> bool:
    """Whether an edge is incident to a node removed from a target repo slice."""

    source, target = _edge_endpoints(edge)
    return source in target_nodes or target in target_nodes


def merge_edge_attributes(
    existing: Mapping[str, JsonValue], incoming: Mapping[str, JsonValue], header: NodeLinkHeader
) -> JsonObject:
    """Apply ``Graph.add_edge`` attribute semantics to one equal edge identity.

    The stored endpoint spelling and multigraph key remain from the first edge;
    incoming non-identity attributes overwrite matching old attributes while
    omitted attributes survive.
    """

    if edge_identity(existing, header) != edge_identity(incoming, header):
        raise NodeLinkFormatError("cannot merge attributes for different edge identities")
    merged = dict(existing)
    identity_fields = {"source", "target"}
    if header.multigraph:
        identity_fields.add("key")
    for key, value in incoming.items():
        if key not in identity_fields:
            merged[key] = value
    return merged


def merge_node_attributes(
    existing: Mapping[str, JsonValue], incoming: Mapping[str, JsonValue]
) -> JsonObject:
    """Apply ``Graph.add_node`` attributes for one equal node identity.

    The retained record's ``id`` spelling stays authoritative, including a
    list-form tuple id. Incoming non-id attributes overwrite old values while
    attributes omitted by the incoming node remain intact.
    """

    if node_identity(existing) != node_identity(incoming):
        raise NodeLinkFormatError("cannot merge attributes for different node identities")
    merged = dict(existing)
    for key, value in incoming.items():
        if key != "id":
            merged[key] = value
    return merged


def _read_header(path: Path, backend_name: str | None) -> NodeLinkHeader:
    directed: bool | None = None
    multigraph: bool | None = None
    graph: JsonObject | None = None
    array_keys: set[str] = set()
    seen_standard_keys: set[str] = set()
    root_started = False
    root_finished = False
    expected_key: str | None = None
    graph_builder: ObjectBuilder | None = None

    try:
        with path.open("rb") as handle:
            for prefix, event, value in _iter_events(handle, backend_name):
                if graph_builder is not None:
                    if prefix == "graph" or prefix.startswith("graph."):
                        graph_builder.event(event, value)
                        if prefix == "graph" and event in {"end_map", "end_array", "null", "boolean", "number", "string"}:
                            graph = _json_object(graph_builder.value, "node-link graph metadata")
                            graph_builder = None
                            expected_key = None
                        continue

                if prefix == "":
                    if event == "start_map":
                        if root_started:
                            raise NodeLinkFormatError("node-link input must contain one top-level object")
                        root_started = True
                    elif event == "map_key":
                        if not root_started or root_finished or not isinstance(value, str):
                            raise NodeLinkFormatError("node-link input has an invalid top-level key")
                        expected_key = value
                        if value in {"directed", "multigraph", "graph", "nodes", "links", "edges"}:
                            if value in seen_standard_keys:
                                raise NodeLinkFormatError(f"node-link input repeats top-level key {value!r}")
                            seen_standard_keys.add(value)
                    elif event == "end_map":
                        root_finished = True
                    continue

                if expected_key is None or prefix != expected_key:
                    continue

                if expected_key in {"nodes", "links", "edges"}:
                    if event != "start_array":
                        raise NodeLinkFormatError(f"node-link {expected_key!r} must be an array")
                    array_keys.add(expected_key)
                    expected_key = None
                elif expected_key == "directed":
                    if not isinstance(value, bool):
                        raise NodeLinkFormatError("node-link 'directed' must be boolean")
                    directed = value
                    expected_key = None
                elif expected_key == "multigraph":
                    if not isinstance(value, bool):
                        raise NodeLinkFormatError("node-link 'multigraph' must be boolean")
                    multigraph = value
                    expected_key = None
                elif expected_key == "graph":
                    graph_builder = ObjectBuilder()
                    graph_builder.event(event, value)
                    if event in {"null", "boolean", "number", "string"}:
                        graph = _json_object(graph_builder.value, "node-link graph metadata")
                        graph_builder = None
                        expected_key = None
                else:
                    expected_key = None
    except JSONError as exc:
        raise NodeLinkFormatError(f"cannot parse node-link JSON at {path}: {exc}") from exc

    if not root_started or not root_finished or graph_builder is not None:
        raise NodeLinkFormatError(f"{path} is not a complete node-link JSON object")
    if directed is None or multigraph is None or graph is None or "nodes" not in array_keys:
        raise NodeLinkFormatError(f"{path} is missing required node-link topology fields")
    if "links" in array_keys:
        link_key: LinkKey = "links"
    elif "edges" in array_keys:
        link_key = "edges"
    else:
        raise NodeLinkFormatError(f"{path} is missing node-link 'links' (or legacy 'edges')")
    return NodeLinkHeader(directed=directed, multigraph=multigraph, graph=graph, link_key=link_key)


class NodeLinkStream:
    """A stable, bounded multi-pass view of one node-link JSON file."""

    def __init__(self, path: str | Path, *, backend: str | None = None) -> None:
        self.path = Path(path)
        self.backend = backend
        self.token = FileIdentity.capture(self.path)
        _reject_nonfinite_tokens(self.path)
        self._assert_identity()
        self._header: NodeLinkHeader | None = None

    def _assert_identity(self) -> None:
        current = FileIdentity.capture(self.path)
        if current != self.token:
            raise NodeLinkChangedError(
                f"{self.path} changed while streaming; retry from a stable source"
            )

    def header(self) -> NodeLinkHeader:
        """Return validated topology metadata without materializing node/link arrays."""

        self._assert_identity()
        if self._header is None:
            try:
                self._header = _read_header(self.path, self.backend)
            finally:
                self._assert_identity()
        return self._header

    def _records(self, array_key: str) -> Iterator[JsonObject]:
        header = self.header()
        self._assert_identity()
        try:
            with self.path.open("rb") as handle:
                for raw in _backend(self.backend).items(handle, f"{array_key}.item", use_float=False):
                    record = _json_object(raw, f"node-link {array_key} item")
                    if array_key == "nodes":
                        node_identity(record)
                    elif array_key == header.link_key:
                        edge_identity(record, header)
                    yield record
        except JSONError as exc:
            raise NodeLinkFormatError(f"cannot parse node-link JSON at {self.path}: {exc}") from exc
        finally:
            self._assert_identity()

    def nodes(self) -> Iterator[JsonObject]:
        """Iterate validated node records in a fresh, identity-checked pass."""

        return self._records("nodes")

    def links(self) -> Iterator[JsonObject]:
        """Iterate authoritative ``links`` or only-if-absent legacy ``edges``."""

        return self._records(self.header().link_key)


def _write_json_value(handle: TextIO, value: JsonValue, indent: int) -> None:
    encoded = json.dumps(value, ensure_ascii=True, indent=2, allow_nan=False)
    padding = " " * indent
    handle.write(padding)
    handle.write(encoded.replace("\n", f"\n{padding}"))


class NodeLinkEmitter:
    """Incrementally write one canonical node-link object through a callback."""

    def __init__(self, handle: TextIO, header: NodeLinkHeader) -> None:
        self._handle = handle
        self._header = header
        self._section = "nodes"
        self._node_count = 0
        self._link_count = 0
        self._handle.write("{\n  \"directed\": ")
        self._handle.write(json.dumps(header.directed))
        self._handle.write(",\n  \"multigraph\": ")
        self._handle.write(json.dumps(header.multigraph))
        self._handle.write(",\n  \"graph\": ")
        self._handle.write(json.dumps(header.graph, ensure_ascii=True, indent=2, allow_nan=False))
        self._handle.write(",\n  \"nodes\": [")

    def write_node(self, node: Mapping[str, object]) -> None:
        """Append one node before links begin."""

        if self._section != "nodes":
            raise NodeLinkEmissionError("nodes must be emitted before links")
        record = _json_object(dict(node), "emitted node")
        node_identity(record)
        if self._node_count:
            self._handle.write(",")
        self._handle.write("\n")
        _write_json_value(self._handle, record, 4)
        self._node_count += 1

    def _begin_links(self) -> None:
        if self._section == "links":
            return
        if self._section != "nodes":
            raise NodeLinkEmissionError("node-link emission is already finished")
        if self._node_count:
            self._handle.write("\n")
        self._handle.write("  ],\n  \"links\": [")
        self._section = "links"

    def write_link(self, link: Mapping[str, object]) -> None:
        """Append one link, beginning the authoritative links section if needed."""

        self._begin_links()
        record = _json_object(dict(link), "emitted link")
        edge_identity(record, self._header)
        if self._link_count:
            self._handle.write(",")
        self._handle.write("\n")
        _write_json_value(self._handle, record, 4)
        self._link_count += 1

    def finish(self) -> None:
        """Close a valid JSON object, including an empty links array when needed."""

        self._begin_links()
        if self._link_count:
            self._handle.write("\n")
        self._handle.write("  ]\n}\n")
        self._section = "finished"


def emit_node_link_atomic(
    path: str | Path,
    header: NodeLinkHeader,
    emit: Callable[[NodeLinkEmitter], None],
) -> None:
    """Emit a canonical links-keyed node-link file through an atomic callback."""

    def write(handle: TextIO) -> None:
        emitter = NodeLinkEmitter(handle, header)
        emit(emitter)
        emitter.finish()

    write_callback_atomic(path, write)


__all__ = [
    "EdgeIdentity",
    "FileIdentity",
    "NodeLinkChangedError",
    "NodeLinkEmissionError",
    "NodeLinkFormatError",
    "NodeLinkHeader",
    "NodeLinkStream",
    "NodeLinkStreamError",
    "NonFiniteJsonError",
    "edge_identity",
    "edge_touches_target",
    "emit_node_link_atomic",
    "merge_edge_attributes",
    "merge_node_attributes",
    "node_identity",
    "node_is_target",
]
