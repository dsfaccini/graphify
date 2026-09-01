"""Unit coverage for batched graph-database exporters without database clients."""
from __future__ import annotations

import sys
from types import SimpleNamespace

import networkx as nx
import pytest

from graphify.exporters.graphdb import push_to_falkordb, push_to_neo4j


class _Result:
    def __init__(self, consume_error=None):
        self.consume_error = consume_error

    def consume(self):
        if self.consume_error is not None:
            raise self.consume_error
        return None


class _Neo4jSession:
    def __init__(self, consume_error=None):
        self.calls = []
        self.consume_error = consume_error

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        return False

    def run(self, cypher, **params):
        self.calls.append((cypher, params))
        return _Result(self.consume_error)


class _Neo4jDriver:
    def __init__(self, consume_error=None):
        self.session_instance = _Neo4jSession(consume_error)
        self.closed = False

    def session(self):
        return self.session_instance

    def close(self):
        self.closed = True


class _GraphDatabase:
    driver_instance = None
    consume_error = None

    @classmethod
    def driver(cls, uri, auth):
        cls.uri = uri
        cls.auth = auth
        cls.driver_instance = _Neo4jDriver(cls.consume_error)
        return cls.driver_instance


class _FalkorGraph:
    def __init__(self):
        self.calls = []

    def query(self, cypher, params):
        self.calls.append((cypher, params))


class _FalkorDB:
    instance = None

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.graph = _FalkorGraph()
        self.selected_graph = None
        type(self).instance = self

    def select_graph(self, graph_name):
        self.selected_graph = graph_name
        return self.graph


def _graph():
    graph = nx.DiGraph()
    for index in range(501):
        node_id = f"source-{index}"
        graph.add_node(
            node_id,
            file_type="source-file",
            count=index,
            tags=["not", "a", "scalar"],
            _private="never export",
        )
    graph.add_node("target", file_type="target doc", enabled=True)
    for index in range(501):
        graph.add_edge(
            f"source-{index}",
            "target",
            relation="uses-file",
            weight=0.75,
            _private="never export",
        )
    graph.add_edge("target", "source-0", relation="references this")
    return graph


def _assert_batched_rows(calls, graph):
    assert calls
    assert all(len(params["rows"]) <= 500 for _, params in calls)

    node_rows = [
        row
        for cypher, params in calls
        if "MERGE (n:" in cypher
        for row in params["rows"]
    ]
    edge_rows = [
        row
        for cypher, params in calls
        if "MERGE (a)-[r:" in cypher
        for row in params["rows"]
    ]
    assert {row["id"] for row in node_rows} == set(graph.nodes)
    assert len(node_rows) == graph.number_of_nodes()
    assert {(row["src"], row["tgt"]) for row in edge_rows} == set(graph.edges)
    assert len(edge_rows) == graph.number_of_edges()

    source = next(row for row in node_rows if row["id"] == "source-0")
    assert source["community"] == 4
    assert source["count"] == 0
    assert "tags" not in source and "_private" not in source
    edge = next(row for row in edge_rows if row["src"] == "source-0")
    assert edge["props"] == {"relation": "uses-file", "weight": 0.75}


def test_push_to_neo4j_batches_homogeneous_rows(monkeypatch):
    graph = _graph()
    monkeypatch.setitem(sys.modules, "neo4j", SimpleNamespace(GraphDatabase=_GraphDatabase))

    result = push_to_neo4j(
        graph,
        uri="bolt://example.test:7687",
        user="neo4j",
        password="secret",
        communities={4: ["source-0"]},
    )

    driver = _GraphDatabase.driver_instance
    assert result == {"nodes": graph.number_of_nodes(), "edges": graph.number_of_edges()}
    assert _GraphDatabase.uri == "bolt://example.test:7687"
    assert _GraphDatabase.auth == ("neo4j", "secret")
    assert driver.closed
    calls = driver.session_instance.calls
    _assert_batched_rows(calls, graph)
    assert any("MERGE (n:Sourcefile" in cypher for cypher, _ in calls)
    assert any("MERGE (n:Targetdoc" in cypher for cypher, _ in calls)
    assert any(
        "MATCH (a:Sourcefile {id: row.src}), (b:Targetdoc {id: row.tgt})" in cypher
        and "MERGE (a)-[r:USES_FILE]->(b)" in cypher
        for cypher, _ in calls
    )
    assert any(
        "MATCH (a:Targetdoc {id: row.src}), (b:Sourcefile {id: row.tgt})" in cypher
        and "MERGE (a)-[r:REFERENCES_THIS]->(b)" in cypher
        for cypher, _ in calls
    )


def test_push_to_falkordb_batches_homogeneous_rows(monkeypatch):
    graph = _graph()
    monkeypatch.setitem(sys.modules, "falkordb", SimpleNamespace(FalkorDB=_FalkorDB))

    result = push_to_falkordb(
        graph,
        uri="redis://uri-user:uri-password@example.test:6380",
        user="ignored-user",
        password="ignored-password",
        communities={4: ["source-0"]},
        graph_name="staging",
    )

    db = _FalkorDB.instance
    assert result == {"nodes": graph.number_of_nodes(), "edges": graph.number_of_edges()}
    assert db.kwargs == {
        "host": "example.test",
        "port": 6380,
        "username": "uri-user",
        "password": "uri-password",
    }
    assert db.selected_graph == "staging"
    calls = db.graph.calls
    _assert_batched_rows(calls, graph)
    assert any("MERGE (n:Sourcefile" in cypher for cypher, _ in calls)
    assert any("MERGE (n:Targetdoc" in cypher for cypher, _ in calls)
    assert any(
        "MATCH (a:Sourcefile {id: row.src}), (b:Targetdoc {id: row.tgt})" in cypher
        and "MERGE (a)-[r:USES_FILE]->(b)" in cypher
        for cypher, _ in calls
    )


def test_neo4j_flushes_many_sparse_shapes_without_dropping_rows(monkeypatch):
    graph = nx.Graph()
    for index in range(65):
        graph.add_node(f"node-{index}", file_type=f"kind-{index}")
    monkeypatch.setitem(sys.modules, "neo4j", SimpleNamespace(GraphDatabase=_GraphDatabase))

    result = push_to_neo4j(graph, "bolt://example.test", "neo4j", "secret")

    calls = _GraphDatabase.driver_instance.session_instance.calls
    rows = [row for _, params in calls for row in params["rows"]]
    assert result == {"nodes": 65, "edges": 0}
    assert {row["id"] for row in rows} == set(graph.nodes)
    assert all(len(params["rows"]) <= 500 for _, params in calls)


def test_empty_graphs_issue_no_queries(monkeypatch):
    graph = nx.Graph()
    monkeypatch.setitem(sys.modules, "neo4j", SimpleNamespace(GraphDatabase=_GraphDatabase))
    monkeypatch.setitem(sys.modules, "falkordb", SimpleNamespace(FalkorDB=_FalkorDB))

    neo4j_result = push_to_neo4j(graph, "bolt://example.test", "neo4j", "secret")
    falkordb_result = push_to_falkordb(graph, "example.test:6379")

    assert neo4j_result == falkordb_result == {"nodes": 0, "edges": 0}
    assert _GraphDatabase.driver_instance.session_instance.calls == []
    assert _GraphDatabase.driver_instance.closed
    assert _FalkorDB.instance.graph.calls == []
    assert _FalkorDB.instance.kwargs == {
        "host": "example.test",
        "port": 6379,
        "username": None,
        "password": None,
    }


def test_push_to_neo4j_closes_driver_when_consume_fails(monkeypatch):
    graph = nx.Graph()
    graph.add_node("node", file_type="source")
    error = RuntimeError("consume failed")
    monkeypatch.setattr(_GraphDatabase, "consume_error", error)
    monkeypatch.setitem(sys.modules, "neo4j", SimpleNamespace(GraphDatabase=_GraphDatabase))

    with pytest.raises(RuntimeError, match="consume failed"):
        push_to_neo4j(graph, "bolt://example.test", "neo4j", "secret")

    assert _GraphDatabase.driver_instance.closed


def test_exporters_prefix_identifiers_starting_with_digits(monkeypatch):
    graph = nx.DiGraph()
    graph.add_node("source", file_type="9 source")
    graph.add_node("target", file_type="2 target")
    graph.add_edge("source", "target", relation="9-rel")
    monkeypatch.setitem(sys.modules, "neo4j", SimpleNamespace(GraphDatabase=_GraphDatabase))
    monkeypatch.setitem(sys.modules, "falkordb", SimpleNamespace(FalkorDB=_FalkorDB))

    push_to_neo4j(graph, "bolt://example.test", "neo4j", "secret")
    push_to_falkordb(graph, "example.test:6379")

    neo4j_calls = _GraphDatabase.driver_instance.session_instance.calls
    falkordb_calls = _FalkorDB.instance.graph.calls
    for calls in (neo4j_calls, falkordb_calls):
        assert any("MERGE (n:_9source" in cypher for cypher, _ in calls)
        assert any("MERGE (n:_2target" in cypher for cypher, _ in calls)
        assert any(
            "MATCH (a:_9source {id: row.src}), (b:_2target {id: row.tgt})" in cypher
            and "MERGE (a)-[r:_9_REL]->(b)" in cypher
            for cypher, _ in calls
        )


def test_exporters_order_parallel_edges_before_merging(monkeypatch):
    graph = nx.MultiDiGraph()
    graph.add_node("source", file_type="source")
    graph.add_node("target", file_type="target")
    graph.add_edge("source", "target", relation="uses", winner="first")
    graph.add_edge("source", "target", relation="uses", winner="second")
    monkeypatch.setitem(sys.modules, "neo4j", SimpleNamespace(GraphDatabase=_GraphDatabase))
    monkeypatch.setitem(sys.modules, "falkordb", SimpleNamespace(FalkorDB=_FalkorDB))

    push_to_neo4j(graph, "bolt://example.test", "neo4j", "secret")
    push_to_falkordb(graph, "example.test:6379")

    neo4j_calls = _GraphDatabase.driver_instance.session_instance.calls
    falkordb_calls = _FalkorDB.instance.graph.calls
    for calls in (neo4j_calls, falkordb_calls):
        edge_query, edge_params = next(
            (cypher, params) for cypher, params in calls if "MERGE (a)-[r:USES]" in cypher
        )
        rows = edge_params["rows"]
        assert "UNWIND $rows AS row WITH row ORDER BY row.ordinal" in edge_query
        assert [row["ordinal"] for row in rows] == [0, 1]
        assert [row["props"]["winner"] for row in rows] == ["first", "second"]
        assert all("ordinal" not in row["props"] for row in rows)
