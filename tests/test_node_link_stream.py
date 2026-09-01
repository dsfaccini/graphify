"""Focused contracts for the bounded node-link JSON helper."""

from __future__ import annotations

import json
import tracemalloc

import ijson
import networkx as nx
import pytest

from graphify.node_link_stream import (
    _MAX_NONFINITE_TOKEN_BYTES,
    _reject_nonfinite_tokens,
    NodeLinkChangedError,
    NodeLinkFormatError,
    NodeLinkHeader,
    NodeLinkStream,
    NonFiniteJsonError,
    edge_identity,
    edge_touches_target,
    emit_node_link_atomic,
    merge_edge_attributes,
    merge_node_attributes,
    node_identity,
    node_is_target,
)


BACKENDS = (ijson.backend, "python")


def _write_json(path, payload) -> None:
    path.write_text(json.dumps(payload, allow_nan=False), encoding="utf-8")


@pytest.mark.parametrize("backend", BACKENDS)
def test_header_and_records_support_headers_after_arrays(tmp_path, backend):
    path = tmp_path / "graph.json"
    _write_json(
        path,
        {
            "nodes": [{"id": "a", "ratio": 1.25}],
            "links": [{"source": "a", "target": "a", "relation": "self"}],
            "directed": False,
            "multigraph": False,
            "graph": {"name": "after-arrays"},
        },
    )

    stream = NodeLinkStream(path, backend=backend)

    assert stream.header() == NodeLinkHeader(False, False, {"name": "after-arrays"}, "links")
    assert list(stream.nodes()) == [{"id": "a", "ratio": 1.25}]
    assert list(stream.links()) == [{"source": "a", "target": "a", "relation": "self"}]


@pytest.mark.parametrize("backend", BACKENDS)
def test_links_are_authoritative_with_edges_only_as_fallback(tmp_path, backend):
    both = tmp_path / "both.json"
    _write_json(
        both,
        {
            "directed": False,
            "multigraph": False,
            "graph": {},
            "nodes": [{"id": "a"}, {"id": "b"}],
            "edges": [{"source": "a", "target": "a"}],
            "links": [{"source": "a", "target": "b"}],
        },
    )
    fallback = tmp_path / "fallback.json"
    _write_json(
        fallback,
        {
            "directed": False,
            "multigraph": False,
            "graph": {},
            "nodes": [{"id": "a"}, {"id": "b"}],
            "edges": [{"source": "a", "target": "b"}],
        },
    )

    assert NodeLinkStream(both, backend=backend).header().link_key == "links"
    assert list(NodeLinkStream(both, backend=backend).links()) == [{"source": "a", "target": "b"}]
    assert NodeLinkStream(fallback, backend=backend).header().link_key == "edges"
    assert list(NodeLinkStream(fallback, backend=backend).links()) == [{"source": "a", "target": "b"}]


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize(
    ("graph_factory", "directed", "multigraph"),
    (
        (nx.Graph, False, False),
        (nx.DiGraph, True, False),
        (nx.MultiGraph, False, True),
        (nx.MultiDiGraph, True, True),
    ),
)
def test_networkx_graph_kinds_preserve_header_and_key_requirements(
    tmp_path, backend, graph_factory, directed, multigraph
):
    graph = graph_factory()
    if multigraph:
        graph.add_edge("a", "b", key="named", relation="calls")
    else:
        graph.add_edge("a", "b", relation="calls")
    path = tmp_path / f"{graph_factory.__name__}.json"
    _write_json(path, nx.node_link_data(graph, edges="links"))

    stream = NodeLinkStream(path, backend=backend)
    header = stream.header()
    link = next(stream.links())

    assert header.directed is directed
    assert header.multigraph is multigraph
    if multigraph:
        assert link["key"] == "named"
    assert edge_identity(link, header)


def test_target_helpers_and_simple_edge_attribute_merge():
    undirected = NodeLinkHeader(False, False, {}, "links")
    directed = NodeLinkHeader(True, False, {}, "links")
    target = {"id": "repoA::a", "repo": "repoA"}
    old = {"source": "a", "target": "b", "kept": "old", "changed": "old"}
    incoming = {"source": "b", "target": "a", "changed": "new", "added": "new"}

    assert node_is_target(target, "repoA") is True
    assert edge_touches_target(old, {"a"}) is True
    assert edge_identity(old, undirected) == edge_identity(incoming, undirected)
    assert edge_identity(old, directed) != edge_identity(incoming, directed)
    assert merge_edge_attributes(old, incoming, undirected) == {
        "source": "a",
        "target": "b",
        "kept": "old",
        "changed": "new",
        "added": "new",
    }


def test_keyed_multigraph_identity_and_attribute_merge():
    header = NodeLinkHeader(False, True, {}, "links")
    old = {"source": "a", "target": "b", "key": "k", "kept": "old", "changed": "old"}
    incoming = {"source": "b", "target": "a", "key": "k", "changed": "new"}
    different_key = {"source": "a", "target": "b", "key": "other"}

    assert edge_identity(old, header) == edge_identity(incoming, header)
    assert edge_identity(old, header) != edge_identity(different_key, header)
    assert merge_edge_attributes(old, incoming, header) == {
        "source": "a",
        "target": "b",
        "key": "k",
        "kept": "old",
        "changed": "new",
    }


def test_node_attribute_merge_keeps_retained_list_id_and_merges_networkx_style():
    existing = {"id": ["repo", ["path", 1]], "kept": "old", "changed": "old"}
    incoming = {"id": ["repo", ["path", 1]], "changed": "new", "added": "new"}

    assert node_identity(existing) == ("repo", ("path", 1))
    assert merge_node_attributes(existing, incoming) == {
        "id": ["repo", ["path", 1]],
        "kept": "old",
        "changed": "new",
        "added": "new",
    }
    with pytest.raises(NodeLinkFormatError, match="different node identities"):
        merge_node_attributes(existing, {"id": ["other"], "changed": "new"})


def test_networkx_recursive_node_tuple_ids_preserve_record_spelling():
    node = {"id": ["repo", ["path", 1]]}
    payload = {
        "directed": False,
        "multigraph": False,
        "graph": {},
        "nodes": [node],
        "links": [],
    }

    assert node_identity(node) == ("repo", ("path", 1))
    assert node["id"] == ["repo", ["path", 1]]
    assert ("repo", ("path", 1)) in nx.node_link_graph(payload, edges="links")


def test_networkx_flat_endpoint_tuple_conversion_rejects_nested_lists_early():
    header = NodeLinkHeader(False, False, {}, "links")
    edge = {"source": ["repo", ["path", 1]], "target": "other"}

    # NetworkX recursively converts node-record IDs but only calls tuple() on
    # edge endpoints. A nested list then remains unhashable; reject it before a
    # streaming caller can begin mutating a graph.
    with pytest.raises(NodeLinkFormatError, match="edge source must be a hashable"):
        edge_identity(edge, header)
    with pytest.raises(TypeError, match="unhashable"):
        nx.node_link_graph(
            {
                "directed": False,
                "multigraph": False,
                "graph": {},
                "nodes": [{"id": ["repo", ["path", 1]]}, {"id": "other"}],
                "links": [edge],
            },
            edges="links",
        )


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize(
    ("directed", "multigraph"),
    ((False, False), (True, False), (False, True), (True, True)),
)
def test_stream_and_emitter_support_flat_tuple_ids_for_all_graph_kinds(
    tmp_path, backend, directed, multigraph
):
    nodes = [{"id": ["repo", 1]}, {"id": ["repo", 2]}]
    link = {"source": ["repo", 1], "target": ["repo", 2], "weight": 1}
    if multigraph:
        link["key"] = "edge-key"
    payload = {
        "directed": directed,
        "multigraph": multigraph,
        "graph": {"name": "tuple ids"},
        "nodes": nodes,
        "links": [link],
    }
    path = tmp_path / "tuple-ids.json"
    _write_json(path, payload)

    stream = NodeLinkStream(path, backend=backend)
    streamed_nodes = list(stream.nodes())
    streamed_links = list(stream.links())
    assert streamed_nodes == nodes
    assert streamed_links == [link]
    assert node_identity(streamed_nodes[0]) == ("repo", 1)
    assert edge_identity(streamed_links[0], stream.header())
    assert ("repo", 1) in nx.node_link_graph(payload, edges="links")

    output = tmp_path / "emitted.json"

    def emit(emitter) -> None:
        for node in streamed_nodes:
            emitter.write_node(node)
        for streamed_link in streamed_links:
            emitter.write_link(streamed_link)

    emit_node_link_atomic(output, stream.header(), emit)
    assert json.loads(output.read_text(encoding="utf-8")) == payload


@pytest.mark.parametrize("backend", BACKENDS)
def test_decimal_normalization_preserves_large_integer_precision(tmp_path, backend):
    path = tmp_path / "numeric.json"
    path.write_text(
        """{
          "directed": false,
          "multigraph": false,
          "graph": {"weight": 1.25},
          "nodes": [{"id": 9007199254740993, "ratio": 0.125}],
          "links": []
        }""",
        encoding="utf-8",
    )

    stream = NodeLinkStream(path, backend=backend)
    node = next(stream.nodes())

    assert stream.header().graph["weight"] == 1.25
    assert isinstance(node["id"], int)
    assert node["id"] == 9007199254740993
    assert isinstance(node["ratio"], float)
    assert node["ratio"] == 0.125


@pytest.mark.parametrize("token", ("NaN", "Infinity", "-Infinity"))
def test_nonfinite_json_is_rejected_before_backend_selection(tmp_path, token):
    path = tmp_path / "nonfinite.json"
    path.write_text(
        "{" + f'"directed":false,"multigraph":false,"graph":{{}},"nodes":[{{"id":"a","x":{token}}}],"links":[]' + "}",
        encoding="utf-8",
    )

    with pytest.raises(NonFiniteJsonError, match="non-standard JSON number"):
        NodeLinkStream(path, backend="python")


def test_nonfinite_words_inside_strings_are_allowed(tmp_path):
    path = tmp_path / "string.json"
    _write_json(
        path,
        {
            "directed": False,
            "multigraph": False,
            "graph": {"note": "NaN Infinity"},
            "nodes": [{"id": "a"}],
            "links": [],
        },
    )

    assert NodeLinkStream(path).header().graph == {"note": "NaN Infinity"}


def test_nonfinite_scanner_bounds_a_very_long_malformed_bare_token(tmp_path):
    path = tmp_path / "long-token.json"
    path.write_bytes(b"x" * (_MAX_NONFINITE_TOKEN_BYTES * 100_000))

    tracemalloc.start()
    try:
        _reject_nonfinite_tokens(path)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()

    # The scanner reads fixed-size chunks and retains at most ``-Infinity``.
    # This threshold leaves room for one input buffer while ruling out a token
    # sized to the malformed input.
    assert peak < 256 * 1024


@pytest.mark.parametrize("backend", BACKENDS)
def test_corrupt_or_truncated_input_fails_closed(tmp_path, backend):
    path = tmp_path / "truncated.json"
    path.write_text(
        '{"directed":false,"multigraph":false,"graph":{},"nodes":[{"id":"a"}],"links":[',
        encoding="utf-8",
    )

    with pytest.raises(NodeLinkFormatError, match="cannot parse"):
        NodeLinkStream(path, backend=backend).header()


def test_identity_token_rejects_source_changed_between_passes(tmp_path):
    path = tmp_path / "changed.json"
    _write_json(
        path,
        {"directed": False, "multigraph": False, "graph": {}, "nodes": [{"id": "a"}], "links": []},
    )
    stream = NodeLinkStream(path)
    assert stream.header().directed is False
    path.write_text(path.read_text(encoding="utf-8") + "\n", encoding="utf-8")

    with pytest.raises(NodeLinkChangedError, match="changed while streaming"):
        list(stream.nodes())


@pytest.mark.parametrize(
    ("payload", "read_records", "message"),
    (
        (
            {"directed": False, "multigraph": False, "graph": {}, "nodes": [{}], "links": []},
            "nodes",
            "missing 'id'",
        ),
        (
            {
                "directed": False,
                "multigraph": False,
                "graph": {},
                "nodes": [],
                "links": [{"target": "b"}],
            },
            "links",
            "missing 'source' or 'target'",
        ),
        (
            {
                "directed": True,
                "multigraph": True,
                "graph": {},
                "nodes": [],
                "links": [{"source": "a", "target": "b"}],
            },
            "links",
            "missing 'key'",
        ),
    ),
)
def test_stream_records_validate_required_identity_fields(tmp_path, payload, read_records, message):
    path = tmp_path / "malformed-record.json"
    _write_json(path, payload)
    stream = NodeLinkStream(path)

    with pytest.raises(NodeLinkFormatError, match=message):
        list(getattr(stream, read_records)())


def test_atomic_emission_uses_callback_and_removes_failed_temporary_file(tmp_path):
    output = tmp_path / "global-graph.json"
    output.write_text("old graph", encoding="utf-8")
    header = NodeLinkHeader(False, False, {"name": "global"}, "links")

    def fail_after_one_node(emitter) -> None:
        emitter.write_node({"id": "a"})
        raise RuntimeError("simulated failure")

    with pytest.raises(RuntimeError, match="simulated failure"):
        emit_node_link_atomic(output, header, fail_after_one_node)

    assert output.read_text(encoding="utf-8") == "old graph"
    assert not list(tmp_path.glob(".global-graph.json.*.tmp"))


def test_atomic_emission_writes_authoritative_links_schema(tmp_path):
    output = tmp_path / "global-graph.json"
    header = NodeLinkHeader(True, True, {"name": "global"}, "edges")

    def emit(emitter) -> None:
        emitter.write_node({"id": "a"})
        emitter.write_node({"id": "b"})
        emitter.write_link({"source": "a", "target": "b", "key": "k", "relation": "calls"})

    emit_node_link_atomic(output, header, emit)
    payload = json.loads(output.read_text(encoding="utf-8"))

    assert payload == {
        "directed": True,
        "multigraph": True,
        "graph": {"name": "global"},
        "nodes": [{"id": "a"}, {"id": "b"}],
        "links": [{"source": "a", "target": "b", "key": "k", "relation": "calls"}],
    }
