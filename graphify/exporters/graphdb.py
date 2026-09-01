"""graphdb — moved verbatim from graphify/export.py."""
from __future__ import annotations

from collections.abc import Hashable, Iterator, Mapping
from graphify.analyze import _node_community_map
import networkx as nx
import re
from typing import TypeVar


_BATCH_SIZE = 500
_MAX_OPEN_BATCHES = 64

_BatchShape = TypeVar("_BatchShape", bound=Hashable)
_BatchRow = TypeVar("_BatchRow")


def _safe_rel(relation: object) -> str:
    sanitized = re.sub(r"[^A-Z0-9_]", "_", str(relation).upper().replace(" ", "_").replace("-", "_"))
    if sanitized and sanitized[0].isdigit():
        sanitized = f"_{sanitized}"
    return sanitized or "RELATED_TO"


def _safe_label(label: object) -> str:
    """Sanitize a graph-database node label to prevent Cypher injection."""
    sanitized = re.sub(r"[^A-Za-z0-9_]", "", str(label))
    if sanitized and sanitized[0].isdigit():
        sanitized = f"_{sanitized}"
    return sanitized if sanitized else "Entity"


def _scalar_props(data: Mapping[str, object]) -> dict[str, object]:
    return {
        key: value for key, value in data.items()
        if isinstance(value, (str, int, float, bool)) and not key.startswith("_")
    }


def _node_label(data: Mapping[str, object]) -> str:
    return _safe_label(str(data.get("file_type", "Entity")).capitalize())


def _append_to_batch(
    batches: dict[_BatchShape, list[_BatchRow]],
    shape: _BatchShape,
    row: _BatchRow,
) -> tuple[_BatchShape, list[_BatchRow]] | None:
    flushed = None
    if shape not in batches and len(batches) >= _MAX_OPEN_BATCHES:
        oldest_shape = next(iter(batches))
        flushed = oldest_shape, batches.pop(oldest_shape)

    batch = batches.setdefault(shape, [])
    batch.append(row)
    if len(batch) == _BATCH_SIZE:
        return shape, batches.pop(shape)
    return flushed


def _node_batches(
    G: nx.Graph,
    node_community: Mapping[object, object],
) -> Iterator[tuple[str, list[dict[str, object]]]]:
    batches: dict[str, list[dict[str, object]]] = {}
    for node_id, data in G.nodes(data=True):
        props = _scalar_props(data)
        props["id"] = node_id
        cid = node_community.get(node_id)
        if cid is not None:
            props["community"] = cid
        flushed = _append_to_batch(batches, _node_label(data), props)
        if flushed is not None:
            yield flushed
    yield from batches.items()


def _edge_batches(
    G: nx.Graph,
) -> Iterator[tuple[tuple[str, str, str], list[dict[str, object]]]]:
    batches: dict[tuple[str, str, str], list[dict[str, object]]] = {}
    for ordinal, (source, target, data) in enumerate(G.edges(data=True)):
        shape = (_node_label(G.nodes[source]), _node_label(G.nodes[target]), _safe_rel(data.get("relation", "RELATED_TO")))
        row = {
            "src": source,
            "tgt": target,
            "props": _scalar_props(data),
            "ordinal": ordinal,
        }
        flushed = _append_to_batch(batches, shape, row)
        if flushed is not None:
            yield flushed
    yield from batches.items()


def push_to_neo4j(
    G: nx.Graph,
    uri: str,
    user: str,
    password: str,
    communities: dict[int, list[str]] | None = None,
) -> dict[str, int]:
    """Push graph directly to a running Neo4j instance via the Python driver.

    Requires: pip install neo4j

    Uses MERGE so re-running is safe - nodes and edges are upserted, not duplicated.
    Returns a dict with counts of nodes and edges pushed.
    """
    try:
        from neo4j import GraphDatabase
    except ImportError as e:
        raise ImportError(
            "neo4j driver not installed. Run: pip install neo4j"
        ) from e

    node_community = _node_community_map(communities) if communities else {}

    driver = GraphDatabase.driver(uri, auth=(user, password))
    nodes_pushed = 0
    edges_pushed = 0

    try:
        with driver.session() as session:
            for label, rows in _node_batches(G, node_community):
                session.run(
                    f"UNWIND $rows AS row MERGE (n:{label} {{id: row.id}}) SET n += row",
                    rows=rows,
                ).consume()
                nodes_pushed += len(rows)

            for (source_label, target_label, relation), rows in _edge_batches(G):
                session.run(
                    f"UNWIND $rows AS row "
                    f"WITH row ORDER BY row.ordinal "
                    f"MATCH (a:{source_label} {{id: row.src}}), (b:{target_label} {{id: row.tgt}}) "
                    f"MERGE (a)-[r:{relation}]->(b) SET r += row.props",
                    rows=rows,
                ).consume()
                edges_pushed += len(rows)
    finally:
        driver.close()

    return {"nodes": nodes_pushed, "edges": edges_pushed}

def push_to_falkordb(
    G: nx.Graph,
    uri: str,
    user: str | None = None,
    password: str | None = None,
    communities: dict[int, list[str]] | None = None,
    graph_name: str = "graphify",
) -> dict[str, int]:
    """Push graph directly to a running FalkorDB instance via the Python SDK.

    Requires: pip install falkordb

    FalkorDB is OpenCypher-compatible, so the MERGE/SET upsert queries are
    identical to push_to_neo4j. Differences from the Neo4j path:
      - connects with FalkorDB(host, port, username, password) instead of a bolt
        driver; only the host/port are read from the URI, so the scheme is
        informational - "falkordb://localhost:6379", "redis://localhost:6379"
        and a bare "localhost:6379" are all equivalent (default port 6379).
      - a named graph is selected via db.select_graph(graph_name) (default
        "graphify"); FalkorDB keys each graph by name in the same instance.
      - queries run via graph.query(cypher, params) - there is no session object.
      - auth is optional (FalkorDB runs without credentials by default), so user
        and password may be None.
      - no APOC: the Neo4j path does not use APOC either, so nothing to port.

    Uses MERGE so re-running is safe - nodes and edges are upserted, not
    duplicated. Returns a dict with counts of nodes and edges pushed.
    """
    try:
        from falkordb import FalkorDB
    except ImportError as e:
        raise ImportError(
            "falkordb SDK not installed. Run: pip install falkordb"
        ) from e

    from urllib.parse import urlparse

    node_community = _node_community_map(communities) if communities else {}

    parsed = urlparse(uri if "://" in uri else f"redis://{uri}")
    # FalkorDB auth is optional. Only send credentials when a password is
    # provided; otherwise connect anonymously and ignore any bolt-style default
    # username (e.g. Neo4j's "neo4j"), which FalkorDB rejects as an unknown ACL
    # user. Credentials embedded in the URI take precedence over the args.
    connect_user = parsed.username or (user if password else None)
    connect_password = parsed.password or (password or None)
    db = FalkorDB(
        host=parsed.hostname or "localhost",
        port=parsed.port or 6379,
        username=connect_user,
        password=connect_password,
    )
    graph = db.select_graph(graph_name)
    nodes_pushed = 0
    edges_pushed = 0

    for label, rows in _node_batches(G, node_community):
        graph.query(
            f"UNWIND $rows AS row MERGE (n:{label} {{id: row.id}}) SET n += row",
            {"rows": rows},
        )
        nodes_pushed += len(rows)

    for (source_label, target_label, relation), rows in _edge_batches(G):
        graph.query(
            f"UNWIND $rows AS row "
            f"WITH row ORDER BY row.ordinal "
            f"MATCH (a:{source_label} {{id: row.src}}), (b:{target_label} {{id: row.tgt}}) "
            f"MERGE (a)-[r:{relation}]->(b) SET r += row.props",
            {"rows": rows},
        )
        edges_pushed += len(rows)

    return {"nodes": nodes_pushed, "edges": edges_pushed}
