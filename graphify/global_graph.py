from __future__ import annotations

import hashlib
import json
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import TypeAlias

import networkx as nx
from filelock import FileLock
from networkx.readwrite import json_graph as _jg

from graphify.node_link_stream import (
    EdgeIdentity,
    FileIdentity,
    JsonObject,
    JsonValue,
    NodeIdentifier,
    NodeLinkChangedError,
    NodeLinkFormatError,
    NodeLinkHeader,
    NodeLinkStream,
    edge_identity,
    edge_touches_target,
    emit_node_link_atomic,
    merge_edge_attributes,
    merge_node_attributes,
    node_identity,
)

_GLOBAL_DIR = Path.home() / ".graphify"
_GLOBAL_GRAPH = _GLOBAL_DIR / "global-graph.json"
_GLOBAL_MANIFEST = _GLOBAL_DIR / "global-manifest.json"
_HASH_CHUNK_BYTES = 65_536
EndpointPair: TypeAlias = tuple[NodeIdentifier, NodeIdentifier] | frozenset[NodeIdentifier]


@dataclass
class _IncomingSource:
    """The bounded state needed to replay one source graph into a global store."""

    stream: NodeLinkStream
    header: NodeLinkHeader
    nodes: dict[NodeIdentifier, JsonObject]
    node_order: list[NodeIdentifier]
    node_ids: set[NodeIdentifier]
    links: dict[EdgeIdentity, JsonObject]
    link_order: list[EdgeIdentity]
    edge_count: int


def _load_manifest() -> dict:
    if _GLOBAL_MANIFEST.exists():
        try:
            return json.loads(_GLOBAL_MANIFEST.read_text(encoding="utf-8"))
        except Exception as exc:
            # Don't silently wipe the user's manifest on a parse error: that
            # deletes every tracked repo. Back the bad file up and surface the
            # error so the user can recover or report it.
            backup = _GLOBAL_MANIFEST.with_suffix(
                _GLOBAL_MANIFEST.suffix + f".corrupt.{int(datetime.now(timezone.utc).timestamp())}"
            )
            try:
                _GLOBAL_MANIFEST.rename(backup)
                print(
                    f"[graphify global] manifest at {_GLOBAL_MANIFEST} failed to parse ({exc}); "
                    f"moved to {backup} and starting fresh. Restore from the backup if this was "
                    f"unexpected.",
                    file=sys.stderr,
                )
            except Exception as rename_exc:
                print(
                    f"[graphify global] manifest at {_GLOBAL_MANIFEST} failed to parse ({exc}) "
                    f"and could not be backed up ({rename_exc}). Starting fresh.",
                    file=sys.stderr,
                )
    return {"version": 1, "repos": {}}


def _save_manifest(manifest: dict) -> None:
    _GLOBAL_DIR.mkdir(parents=True, exist_ok=True)
    from graphify.paths import write_json_atomic

    write_json_atomic(_GLOBAL_MANIFEST, manifest, indent=2)


def _load_global_graph() -> nx.Graph:
    """Materialize the global graph for compatibility-only callers and tests."""

    if _GLOBAL_GRAPH.exists():
        from graphify.security import check_graph_file_size_cap

        check_graph_file_size_cap(_GLOBAL_GRAPH)
        data = json.loads(_GLOBAL_GRAPH.read_text(encoding="utf-8"))
        if "links" not in data and "edges" in data:
            data = dict(data, links=data["edges"])
        try:
            return _jg.node_link_graph(data, edges="links")
        except TypeError:
            return _jg.node_link_graph(data)
    return nx.Graph()


def _save_global_graph(G: nx.Graph) -> None:
    """Materialize-and-save compatibility helper; mutations stream instead."""

    _GLOBAL_DIR.mkdir(parents=True, exist_ok=True)
    try:
        data = _jg.node_link_data(G, edges="links")
    except TypeError:
        data = _jg.node_link_data(G)
    from graphify.paths import write_json_atomic

    write_json_atomic(_GLOBAL_GRAPH, data, indent=2)


def _file_hash(path: Path) -> str:
    """Hash a graph incrementally so a source hash never needs a whole-file read."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(_HASH_CHUNK_BYTES):
            digest.update(chunk)
    return digest.hexdigest()[:16]


def _assert_stream_stable(stream: NodeLinkStream) -> None:
    if FileIdentity.capture(stream.path) != stream.token:
        raise NodeLinkChangedError(f"{stream.path} changed while streaming; retry from a stable source")


def _transaction_lock() -> FileLock:
    _GLOBAL_DIR.mkdir(parents=True, exist_ok=True)
    return FileLock(_GLOBAL_DIR / "global-graph.lock")


def _source_endpoint(link: Mapping[str, JsonValue], field: str) -> NodeIdentifier:
    return node_identity({"id": link[field]})


def _collect_incoming_source(stream: NodeLinkStream) -> _IncomingSource:
    """Collect only source identities and final duplicate-node attributes."""

    header = stream.header()
    nodes: dict[NodeIdentifier, JsonObject] = {}
    node_order: list[NodeIdentifier] = []
    node_ids: set[NodeIdentifier] = set()

    def remember_node(node_id: NodeIdentifier) -> None:
        if node_id not in node_ids:
            node_ids.add(node_id)
            node_order.append(node_id)

    for node in stream.nodes():
        node_id = node_identity(node)
        remember_node(node_id)
        if node_id in nodes:
            nodes[node_id] = merge_node_attributes(nodes[node_id], node)
        else:
            nodes[node_id] = node

    links: dict[EdgeIdentity, JsonObject] = {}
    link_order: list[EdgeIdentity] = []
    for link in stream.links():
        link_id = edge_identity(link, header)
        if link_id in links:
            links[link_id] = merge_edge_attributes(links[link_id], link, header)
        else:
            links[link_id] = link
            link_order.append(link_id)
        remember_node(_source_endpoint(link, "source"))
        remember_node(_source_endpoint(link, "target"))

    _assert_stream_stable(stream)
    return _IncomingSource(
        stream=stream,
        header=header,
        nodes=nodes,
        node_order=node_order,
        node_ids=node_ids,
        links=links,
        link_order=link_order,
        edge_count=len(links),
    )


def _prefixed_id(repo_tag: str, node_id: NodeIdentifier) -> str:
    return f"{repo_tag}::{node_id}"


def _prefixed_node(
    record: Mapping[str, JsonValue], node_id: NodeIdentifier, repo_tag: str
) -> JsonObject:
    prefixed = _prefixed_id(repo_tag, node_id)
    rewritten = dict(record)
    rewritten["id"] = prefixed
    rewritten["repo"] = repo_tag
    rewritten.setdefault("local_id", prefixed.split("::", 1)[1])
    return rewritten


def _prefixed_implicit_node(node_id: NodeIdentifier, repo_tag: str) -> JsonObject:
    prefixed = _prefixed_id(repo_tag, node_id)
    return {"id": prefixed, "repo": repo_tag, "local_id": prefixed.split("::", 1)[1]}


def _prefixed_metadata(
    metadata: Mapping[str, JsonValue], repo_tag: str, source_ids: set[NodeIdentifier]
) -> JsonObject:
    """Mirror the graph-level hyperedge rewrite in ``prefix_graph_for_global``."""

    rewritten = dict(metadata)
    hyperedges = metadata.get("hyperedges")
    if not isinstance(hyperedges, list):
        return rewritten

    rewritten_hyperedges: list[JsonValue] = []
    for hyperedge in hyperedges:
        if not isinstance(hyperedge, dict):
            rewritten_hyperedges.append(hyperedge)
            continue
        copied = dict(hyperedge)
        members = copied.get("nodes")
        if isinstance(members, list):
            rewritten_members: list[JsonValue] = []
            for member in members:
                try:
                    member_id = node_identity({"id": member})
                except NodeLinkFormatError:
                    rewritten_members.append(member)
                else:
                    rewritten_members.append(
                        _prefixed_id(repo_tag, member_id) if member_id in source_ids else member
                    )
            copied["nodes"] = rewritten_members
        if copied.get("id"):
            copied["id"] = f"{repo_tag}::{copied['id']}"
        rewritten_hyperedges.append(copied)
    rewritten["hyperedges"] = rewritten_hyperedges
    return rewritten


def _fresh_header(incoming: _IncomingSource, repo_tag: str) -> NodeLinkHeader:
    return NodeLinkHeader(
        directed=incoming.header.directed,
        multigraph=incoming.header.multigraph,
        graph=_prefixed_metadata(incoming.header.graph, repo_tag, incoming.node_ids),
        link_key="links",
    )


def _incoming_external_labels(incoming: _IncomingSource) -> set[str]:
    labels: set[str] = set()
    for node in incoming.nodes.values():
        label = node.get("label")
        if not node.get("source_file") and isinstance(label, str) and label:
            labels.add(label)
    return labels


def _retained_node_info(
    stream: NodeLinkStream | None, repo_tag: str, incoming_labels: set[str]
) -> tuple[set[NodeIdentifier], dict[str, JsonValue]]:
    target_nodes: set[NodeIdentifier] = set()
    existing_external_labels: dict[str, JsonValue] = {}
    if stream is None:
        return target_nodes, existing_external_labels

    relevant_nodes: set[NodeIdentifier] = set()
    for node in stream.nodes():
        node_id = node_identity(node)
        label = node.get("label")
        if node.get("repo") == repo_tag or (
            isinstance(label, str) and label in incoming_labels
        ):
            relevant_nodes.add(node_id)

    final_nodes: dict[NodeIdentifier, JsonObject] = {}
    for node in stream.nodes():
        node_id = node_identity(node)
        if node_id not in relevant_nodes:
            continue
        if node_id in final_nodes:
            final_nodes[node_id] = merge_node_attributes(final_nodes[node_id], node)
        else:
            final_nodes[node_id] = node

    for node_id, node in final_nodes.items():
        if node.get("repo") == repo_tag:
            target_nodes.add(node_id)
            continue
        label = node.get("label")
        if not node.get("source_file") and isinstance(label, str) and label in incoming_labels:
            existing_external_labels[label] = node["id"]
    return target_nodes, existing_external_labels


def _external_remap(
    incoming: _IncomingSource, repo_tag: str, existing_external_labels: Mapping[str, JsonValue]
) -> dict[str, JsonValue]:
    remap: dict[str, JsonValue] = {}
    for node_id, node in incoming.nodes.items():
        label = node.get("label")
        if not node.get("source_file") and isinstance(label, str) and label in existing_external_labels:
            remap[_prefixed_id(repo_tag, node_id)] = existing_external_labels[label]
    return remap


def _pair_for_target(
    source: JsonValue, target: JsonValue, header: NodeLinkHeader
) -> EndpointPair:
    source_id = node_identity({"id": source})
    target_id = node_identity({"id": target})
    return (source_id, target_id) if header.directed else frozenset((source_id, target_id))


def _retained_multigraph_keys(
    stream: NodeLinkStream | None,
    header: NodeLinkHeader,
    target_nodes: set[NodeIdentifier],
    relevant_pairs: set[EndpointPair],
) -> dict[EndpointPair, set[NodeIdentifier]]:
    keys: dict[EndpointPair, set[NodeIdentifier]] = {}
    if stream is None or not header.multigraph or not relevant_pairs:
        return keys

    for link in stream.links():
        if edge_touches_target(link, target_nodes):
            continue
        identity = edge_identity(link, header)
        if identity.endpoints in relevant_pairs and identity.key is not None:
            keys.setdefault(identity.endpoints, set()).add(identity.key)
    return keys


def _next_multigraph_key(keys: dict[EndpointPair, set[NodeIdentifier]], pair: EndpointPair) -> int:
    pair_keys = keys.setdefault(pair, set())
    key = len(pair_keys)
    while key in pair_keys:
        key += 1
    pair_keys.add(key)
    return key


def _rewrite_directional_attributes(
    link: Mapping[str, JsonValue], repo_tag: str, source_ids: set[NodeIdentifier]
) -> JsonObject:
    rewritten = dict(link)
    for field in ("_src", "_tgt"):
        value = rewritten.get(field)
        if value is None:
            continue
        try:
            node_id = node_identity({"id": value})
        except NodeLinkFormatError:
            continue
        if node_id in source_ids:
            rewritten[field] = _prefixed_id(repo_tag, node_id)
    return rewritten


def _incoming_multigraph_pairs(
    incoming: _IncomingSource,
    target_header: NodeLinkHeader,
    repo_tag: str,
    remap: Mapping[str, JsonValue],
) -> set[EndpointPair]:
    """Return only retained pairs that can affect generated simple-source keys."""

    if not target_header.multigraph or incoming.header.multigraph:
        return set()

    pairs: set[EndpointPair] = set()
    for link_id in incoming.link_order:
        link = incoming.links[link_id]
        source_id = _source_endpoint(link, "source")
        target_id = _source_endpoint(link, "target")
        source = remap.get(_prefixed_id(repo_tag, source_id), _prefixed_id(repo_tag, source_id))
        target = remap.get(_prefixed_id(repo_tag, target_id), _prefixed_id(repo_tag, target_id))
        source_after = node_identity({"id": source})
        target_after = node_identity({"id": target})
        if source_id != target_id and source_after == target_after:
            continue
        pairs.add(_pair_for_target(source, target, target_header))
    return pairs


def _prefixed_link(
    link: Mapping[str, JsonValue],
    incoming: _IncomingSource,
    target_header: NodeLinkHeader,
    repo_tag: str,
    remap: Mapping[str, JsonValue],
    keys: dict[EndpointPair, set[NodeIdentifier]],
) -> JsonObject | None:
    source_id = _source_endpoint(link, "source")
    target_id = _source_endpoint(link, "target")
    prefixed_source = _prefixed_id(repo_tag, source_id)
    prefixed_target = _prefixed_id(repo_tag, target_id)
    rewritten = _rewrite_directional_attributes(link, repo_tag, incoming.node_ids)
    rewritten["source"] = remap.get(prefixed_source, prefixed_source)
    rewritten["target"] = remap.get(prefixed_target, prefixed_target)
    for field in ("_src", "_tgt"):
        value = rewritten.get(field)
        if isinstance(value, str):
            rewritten[field] = remap.get(value, value)

    source_after = node_identity({"id": rewritten["source"]})
    target_after = node_identity({"id": rewritten["target"]})
    if source_id != target_id and source_after == target_after:
        return None

    if target_header.multigraph:
        if not incoming.header.multigraph:
            pair = _pair_for_target(rewritten["source"], rewritten["target"], target_header)
            if "key" not in rewritten or rewritten["key"] is None:
                rewritten["key"] = _next_multigraph_key(keys, pair)
            else:
                explicit_key = edge_identity(rewritten, target_header).key
                if explicit_key is not None:
                    keys.setdefault(pair, set()).add(explicit_key)
    elif incoming.header.multigraph:
        rewritten.pop("key", None)
    return rewritten


def _target_final_incoming_links(
    incoming: _IncomingSource,
    target_header: NodeLinkHeader,
    repo_tag: str,
    remap: Mapping[str, JsonValue],
    keys: dict[EndpointPair, set[NodeIdentifier]],
) -> tuple[dict[EdgeIdentity, JsonObject], list[EdgeIdentity]]:
    """Apply target schema identity and NetworkX merge order to source-final links."""

    links: dict[EdgeIdentity, JsonObject] = {}
    order: list[EdgeIdentity] = []
    for source_link_id in incoming.link_order:
        rewritten = _prefixed_link(
            incoming.links[source_link_id],
            incoming,
            target_header,
            repo_tag,
            remap,
            keys,
        )
        if rewritten is None:
            continue
        target_link_id = edge_identity(rewritten, target_header)
        if target_link_id in links:
            links[target_link_id] = merge_edge_attributes(
                links[target_link_id], rewritten, target_header
            )
        else:
            links[target_link_id] = rewritten
            order.append(target_link_id)
    return links, order


def _target_final_incoming_nodes(
    incoming: _IncomingSource, repo_tag: str, remap: Mapping[str, JsonValue]
) -> tuple[dict[str, JsonObject], list[str]]:
    """Return each source-final node once, excluding externally remapped nodes."""

    nodes: dict[str, JsonObject] = {}
    order: list[str] = []
    for node_id in incoming.node_order:
        prefixed = _prefixed_id(repo_tag, node_id)
        if prefixed in remap:
            continue
        record = incoming.nodes.get(node_id)
        nodes[prefixed] = (
            _prefixed_implicit_node(node_id, repo_tag)
            if record is None
            else _prefixed_node(record, node_id, repo_tag)
        )
        order.append(prefixed)
    return nodes, order


def _retained_collision_nodes(
    stream: NodeLinkStream | None,
    target_nodes: set[NodeIdentifier],
    incoming_nodes: Mapping[str, JsonObject],
) -> dict[str, JsonObject]:
    """Collect only retained node identities that will receive incoming attrs."""

    collisions: dict[str, JsonObject] = {}
    if stream is None:
        return collisions
    for node in stream.nodes():
        node_id = node_identity(node)
        if node_id in target_nodes or not isinstance(node_id, str) or node_id not in incoming_nodes:
            continue
        if node_id in collisions:
            collisions[node_id] = merge_node_attributes(collisions[node_id], node)
        else:
            collisions[node_id] = node
    return collisions


def _retained_collision_links(
    stream: NodeLinkStream | None,
    header: NodeLinkHeader,
    target_nodes: set[NodeIdentifier],
    incoming_links: Mapping[EdgeIdentity, JsonObject],
) -> dict[EdgeIdentity, JsonObject]:
    """Collect only retained identities that must merge with incoming output."""

    collisions: dict[EdgeIdentity, JsonObject] = {}
    if stream is None:
        return collisions
    for link in stream.links():
        if edge_touches_target(link, target_nodes):
            continue
        link_id = edge_identity(link, header)
        if link_id not in incoming_links:
            continue
        if link_id in collisions:
            collisions[link_id] = merge_edge_attributes(collisions[link_id], link, header)
        else:
            collisions[link_id] = link
    return collisions


def _emit_updated_graph(
    target_stream: NodeLinkStream | None,
    target_header: NodeLinkHeader,
    incoming: _IncomingSource,
    repo_tag: str,
    target_nodes: set[NodeIdentifier],
    remap: Mapping[str, JsonValue],
    keys: dict[EndpointPair, set[NodeIdentifier]],
) -> None:
    incoming_nodes, incoming_node_order = _target_final_incoming_nodes(incoming, repo_tag, remap)
    node_collisions = _retained_collision_nodes(target_stream, target_nodes, incoming_nodes)
    incoming_links, incoming_order = _target_final_incoming_links(
        incoming, target_header, repo_tag, remap, keys
    )
    collisions = _retained_collision_links(
        target_stream, target_header, target_nodes, incoming_links
    )

    def emit(emitter) -> None:
        if target_stream is not None:
            for node in target_stream.nodes():
                node_id = node_identity(node)
                if node_id not in target_nodes and node_id not in incoming_nodes:
                    emitter.write_node(node)

        for node_id in incoming_node_order:
            incoming_node = incoming_nodes[node_id]
            retained_node = node_collisions.get(node_id)
            if retained_node is None:
                emitter.write_node(incoming_node)
            else:
                emitter.write_node(merge_node_attributes(retained_node, incoming_node))

        if target_stream is not None:
            for link in target_stream.links():
                link_id = edge_identity(link, target_header)
                if not edge_touches_target(link, target_nodes) and link_id not in incoming_links:
                    emitter.write_link(link)

        for link_id in incoming_order:
            incoming_link = incoming_links[link_id]
            retained_link = collisions.get(link_id)
            if retained_link is None:
                emitter.write_link(incoming_link)
            else:
                emitter.write_link(merge_edge_attributes(retained_link, incoming_link, target_header))

    emit_node_link_atomic(_GLOBAL_GRAPH, target_header, emit)


def _existing_global_stream() -> tuple[NodeLinkStream | None, NodeLinkHeader | None]:
    if not _GLOBAL_GRAPH.exists():
        return None, None
    from graphify.security import check_graph_file_size_cap

    check_graph_file_size_cap(_GLOBAL_GRAPH)
    stream = NodeLinkStream(_GLOBAL_GRAPH)
    return stream, stream.header()


def global_add(source_path: Path, repo_tag: str) -> dict:
    """Add or update one project graph through a locked streaming transaction."""

    if not source_path.exists():
        raise FileNotFoundError(f"graph not found: {source_path}")

    from graphify.security import check_graph_file_size_cap

    check_graph_file_size_cap(source_path)
    source_stream = NodeLinkStream(source_path)
    src_hash = _file_hash(source_path)
    _assert_stream_stable(source_stream)

    with _transaction_lock():
        manifest = _load_manifest()
        existing = manifest["repos"].get(repo_tag, {})
        existing_path = existing.get("source_path", "")
        if existing_path and existing_path != str(source_path.resolve()):
            print(
                f"[graphify global] warning: repo tag '{repo_tag}' previously pointed to "
                f"{existing_path!r}, now updating to {str(source_path.resolve())!r}. "
                f"Use --as <tag> to give it a different name.",
                file=sys.stderr,
            )
        if existing.get("source_hash") == src_hash:
            _assert_stream_stable(source_stream)
            return {"repo_tag": repo_tag, "nodes_added": 0, "nodes_removed": 0, "skipped": True}

        incoming = _collect_incoming_source(source_stream)
        target_stream, existing_header = _existing_global_stream()
        target_header = existing_header or _fresh_header(incoming, repo_tag)
        labels = _incoming_external_labels(incoming)
        target_nodes, existing_externals = _retained_node_info(target_stream, repo_tag, labels)
        remap = _external_remap(incoming, repo_tag, existing_externals)
        relevant_pairs = _incoming_multigraph_pairs(incoming, target_header, repo_tag, remap)
        keys = _retained_multigraph_keys(
            target_stream, target_header, target_nodes, relevant_pairs
        )
        _assert_stream_stable(source_stream)
        _emit_updated_graph(
            target_stream,
            target_header,
            incoming,
            repo_tag,
            target_nodes,
            remap,
            keys,
        )

        manifest["repos"][repo_tag] = {
            "added_at": datetime.now(timezone.utc).isoformat(),
            "source_path": str(source_path.resolve()),
            "node_count": len(incoming.node_ids) - len(remap),
            "edge_count": incoming.edge_count,
            "source_hash": src_hash,
        }
        _save_manifest(manifest)
        return {
            "repo_tag": repo_tag,
            "nodes_added": len(incoming.node_ids) - len(remap),
            "nodes_removed": len(target_nodes),
            "skipped": False,
        }


def global_remove(repo_tag: str) -> int:
    """Remove one repository through a locked streaming graph+manifest transaction."""

    with _transaction_lock():
        manifest = _load_manifest()
        if repo_tag not in manifest["repos"]:
            raise KeyError(f"repo '{repo_tag}' not in global graph")

        target_stream, target_header = _existing_global_stream()
        header = target_header or NodeLinkHeader(False, False, {}, "links")
        target_nodes, _ = _retained_node_info(target_stream, repo_tag, set())

        def emit(emitter) -> None:
            if target_stream is None:
                return
            for node in target_stream.nodes():
                if node_identity(node) not in target_nodes:
                    emitter.write_node(node)
            for link in target_stream.links():
                if not edge_touches_target(link, target_nodes):
                    emitter.write_link(link)

        emit_node_link_atomic(_GLOBAL_GRAPH, header, emit)
        del manifest["repos"][repo_tag]
        _save_manifest(manifest)
        return len(target_nodes)


def global_list() -> dict:
    """Return the manifest repos dict."""

    return _load_manifest().get("repos", {})


def global_path() -> Path:
    return _GLOBAL_GRAPH
