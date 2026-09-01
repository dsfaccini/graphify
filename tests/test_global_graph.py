"""Tests for the global graph infrastructure (graphify/global_graph.py),
prefix/prune helpers in graphify/build.py, and the cross-repo guard in
graphify/dedup.py."""
from __future__ import annotations

from contextlib import contextmanager
import json
from pathlib import Path
import pytest
import networkx as nx
from unittest.mock import patch


# ── helpers ──────────────────────────────────────────────────────────────────

def _make_graph(nodes, edges=None):
    """Build a simple nx.Graph from node dicts."""
    G = nx.Graph()
    for n in nodes:
        nid = n["id"]
        G.add_node(nid, **{k: v for k, v in n.items() if k != "id"})
    for e in (edges or []):
        G.add_edge(
            e["source"],
            e["target"],
            **{k: v for k, v in e.items() if k not in ("source", "target")},
        )
    return G


def _graph_to_json(G, path):
    from networkx.readwrite import json_graph as jg
    try:
        data = jg.node_link_data(G, edges="links")
    except TypeError:
        data = jg.node_link_data(G)
    path.write_text(json.dumps(data), encoding="utf-8")


@contextmanager
def _global_store(tmp_path):
    global_dir = tmp_path / ".graphify"
    with patch("graphify.global_graph._GLOBAL_DIR", global_dir), \
         patch("graphify.global_graph._GLOBAL_GRAPH", global_dir / "global-graph.json"), \
         patch("graphify.global_graph._GLOBAL_MANIFEST", global_dir / "global-manifest.json"):
        yield global_dir


# ── build.py helpers ──────────────────────────────────────────────────────────

def test_prefix_graph_preserves_label():
    from graphify.build import prefix_graph_for_global
    G = _make_graph([{"id": "userservice", "label": "UserService", "source_file": "src/user.py"}])
    H = prefix_graph_for_global(G, "repoA")
    assert "repoA::userservice" in H.nodes
    assert "userservice" not in H.nodes
    assert H.nodes["repoA::userservice"]["label"] == "UserService"


def test_prefix_graph_sets_repo_and_local_id():
    from graphify.build import prefix_graph_for_global
    G = _make_graph([{"id": "userservice", "label": "UserService"}])
    H = prefix_graph_for_global(G, "repoA")
    data = H.nodes["repoA::userservice"]
    assert data["repo"] == "repoA"
    assert data["local_id"] == "userservice"


def test_prefix_graph_rewrites_edges():
    from graphify.build import prefix_graph_for_global
    G = _make_graph(
        [{"id": "a", "label": "A"}, {"id": "b", "label": "B"}],
        [{"source": "a", "target": "b"}],
    )
    H = prefix_graph_for_global(G, "repo1")
    assert H.has_edge("repo1::a", "repo1::b")
    assert not H.has_edge("a", "b")


def test_prefix_graph_rewrites_edge_directional_attributes():
    """prefix_graph_for_global must update directional edge attributes (_src/_tgt)
    so they stay aligned with the prefixed node IDs (#2261)."""
    from graphify.build import prefix_graph_for_global
    G = _make_graph(
        [{"id": "rota", "label": "rota.js"}, {"id": "collections", "label": "collections.js"}],
        [{"source": "rota", "target": "collections", "relation": "imports_from", "_src": "rota", "_tgt": "collections"}],
    )
    H = prefix_graph_for_global(G, "repoA")
    assert H.has_edge("repoA::rota", "repoA::collections")
    data = H.get_edge_data("repoA::rota", "repoA::collections")
    assert data["_src"] == "repoA::rota"
    assert data["_tgt"] == "repoA::collections"


def test_prefix_graph_offsets_community_ids():
    """#3014: every input graph numbers its communities from 0, so a merge
    carrying ids across unchanged fuses community 0 of one repo with community
    0 of another into a single meta-node in the aggregated view. An offset must
    shift integer ids into a shared id space and keep the per-repo id in
    local_community."""
    from graphify.build import prefix_graph_for_global
    G = _make_graph(
        [{"id": "a", "community": 0}, {"id": "b", "community": 1}], [],
    )
    H = prefix_graph_for_global(G, "repoA", community_offset=5)
    assert H.nodes["repoA::a"]["community"] == 5
    assert H.nodes["repoA::a"]["local_community"] == 0
    assert H.nodes["repoA::b"]["community"] == 6
    assert H.nodes["repoA::b"]["local_community"] == 1


def test_prefix_graph_zero_offset_leaves_communities_untouched():
    """The default offset must be a no-op — no community rewrite, no
    local_community noise — so single-repo callers (global store, tests)
    keep their ids exactly as stored."""
    from graphify.build import prefix_graph_for_global
    G = _make_graph(
        [{"id": "a", "community": 0}, {"id": "b", "community": 1}], [],
    )
    H = prefix_graph_for_global(G, "repoA")
    assert H.nodes["repoA::a"]["community"] == 0
    assert H.nodes["repoA::b"]["community"] == 1
    assert "local_community" not in H.nodes["repoA::a"]
    assert "local_community" not in H.nodes["repoA::b"]



def test_prune_repo_removes_correct_nodes():
    from graphify.build import prune_repo_from_graph
    G = nx.Graph()
    G.add_node("repoA::userservice", repo="repoA", label="UserService")
    G.add_node("repoB::userservice", repo="repoB", label="UserService")
    G.add_node("repoA::auth", repo="repoA", label="Auth")
    removed = prune_repo_from_graph(G, "repoA")
    assert removed == 2
    assert "repoB::userservice" in G.nodes
    assert "repoA::userservice" not in G.nodes
    assert "repoA::auth" not in G.nodes


def test_prune_repo_returns_zero_if_not_present():
    from graphify.build import prune_repo_from_graph
    G = nx.Graph()
    G.add_node("repoA::x", repo="repoA")
    removed = prune_repo_from_graph(G, "repoB")
    assert removed == 0
    assert G.number_of_nodes() == 1


# ── global_graph.py ───────────────────────────────────────────────────────────

def test_global_add_creates_global_graph(tmp_path):
    src_graph = tmp_path / "graph.json"
    G = _make_graph([{"id": "userservice", "label": "UserService", "source_file": "src/user.py"}])
    _graph_to_json(G, src_graph)

    global_dir = tmp_path / ".graphify"
    with patch("graphify.global_graph._GLOBAL_DIR", global_dir), \
         patch("graphify.global_graph._GLOBAL_GRAPH", global_dir / "global-graph.json"), \
         patch("graphify.global_graph._GLOBAL_MANIFEST", global_dir / "global-manifest.json"):
        from graphify.global_graph import global_add
        result = global_add(src_graph, "repoA")

    assert result["skipped"] is False
    assert result["nodes_added"] > 0
    manifest_path = global_dir / "global-manifest.json"
    assert manifest_path.exists()
    manifest = json.loads(manifest_path.read_text())
    assert "repoA" in manifest["repos"]


def test_global_add_skip_on_unchanged_hash(tmp_path):
    src_graph = tmp_path / "graph.json"
    G = _make_graph([{"id": "userservice", "label": "UserService", "source_file": "src/user.py"}])
    _graph_to_json(G, src_graph)

    global_dir = tmp_path / ".graphify"
    with patch("graphify.global_graph._GLOBAL_DIR", global_dir), \
         patch("graphify.global_graph._GLOBAL_GRAPH", global_dir / "global-graph.json"), \
         patch("graphify.global_graph._GLOBAL_MANIFEST", global_dir / "global-manifest.json"):
        from graphify.global_graph import global_add
        global_add(src_graph, "repoA")
        result2 = global_add(src_graph, "repoA")

    assert result2["skipped"] is True


def test_global_add_two_repos_no_collision(tmp_path):
    g1 = tmp_path / "graph1.json"
    g2 = tmp_path / "graph2.json"
    G1 = _make_graph([{"id": "userservice", "label": "UserService", "source_file": "src/user.py"}])
    G2 = _make_graph([{"id": "userservice", "label": "UserService", "source_file": "src/user.py"}])
    _graph_to_json(G1, g1)
    _graph_to_json(G2, g2)

    global_dir = tmp_path / ".graphify"
    global_graph_path = global_dir / "global-graph.json"
    global_manifest_path = global_dir / "global-manifest.json"
    with patch("graphify.global_graph._GLOBAL_DIR", global_dir), \
         patch("graphify.global_graph._GLOBAL_GRAPH", global_graph_path), \
         patch("graphify.global_graph._GLOBAL_MANIFEST", global_manifest_path):
        from graphify.global_graph import global_add, _load_global_graph
        global_add(g1, "repoA")
        global_add(g2, "repoB")
        G = _load_global_graph()

    assert "repoA::userservice" in G.nodes
    assert "repoB::userservice" in G.nodes
    assert G.number_of_nodes() == 2  # no silent merge


def test_global_remove(tmp_path):
    src_graph = tmp_path / "graph.json"
    G = _make_graph([{"id": "userservice", "label": "UserService", "source_file": "src/user.py"}])
    _graph_to_json(G, src_graph)

    global_dir = tmp_path / ".graphify"
    with patch("graphify.global_graph._GLOBAL_DIR", global_dir), \
         patch("graphify.global_graph._GLOBAL_GRAPH", global_dir / "global-graph.json"), \
         patch("graphify.global_graph._GLOBAL_MANIFEST", global_dir / "global-manifest.json"):
        from graphify.global_graph import global_add, global_remove
        global_add(src_graph, "repoA")
        removed = global_remove("repoA")

    assert removed > 0
    # manifest should no longer list repoA - need to re-patch for list call
    global_dir2 = global_dir  # same dir
    with patch("graphify.global_graph._GLOBAL_DIR", global_dir2), \
         patch("graphify.global_graph._GLOBAL_GRAPH", global_dir2 / "global-graph.json"), \
         patch("graphify.global_graph._GLOBAL_MANIFEST", global_dir2 / "global-manifest.json"):
        from graphify.global_graph import global_list
        repos = global_list()
    assert "repoA" not in repos


def test_global_remove_unknown_tag_raises(tmp_path):
    global_dir = tmp_path / ".graphify"
    with patch("graphify.global_graph._GLOBAL_DIR", global_dir), \
         patch("graphify.global_graph._GLOBAL_GRAPH", global_dir / "global-graph.json"), \
         patch("graphify.global_graph._GLOBAL_MANIFEST", global_dir / "global-manifest.json"):
        from graphify.global_graph import global_remove
        with pytest.raises(KeyError):
            global_remove("nonexistent")


def test_global_add_collision_warning(tmp_path, capsys):
    g1 = tmp_path / "graph1.json"
    g2 = tmp_path / "graph2.json"
    G = _make_graph([{"id": "x", "label": "X", "source_file": "x.py"}])
    _graph_to_json(G, g1)
    _graph_to_json(G, g2)

    global_dir = tmp_path / ".graphify"
    with patch("graphify.global_graph._GLOBAL_DIR", global_dir), \
         patch("graphify.global_graph._GLOBAL_GRAPH", global_dir / "global-graph.json"), \
         patch("graphify.global_graph._GLOBAL_MANIFEST", global_dir / "global-manifest.json"):
        from graphify.global_graph import global_add
        global_add(g1, "myrepo")
        global_add(g2, "myrepo")  # different source path, same tag

    captured = capsys.readouterr()
    assert "warning" in captured.err.lower() or "warning" in captured.out.lower()


# ── dedup guard ───────────────────────────────────────────────────────────────

def test_dedup_raises_on_cross_repo_nodes():
    from graphify.dedup import deduplicate_entities
    nodes = [
        {"id": "repoA::userservice", "label": "UserService", "repo": "repoA"},
        {"id": "repoB::userservice", "label": "UserService", "repo": "repoB"},
    ]
    with pytest.raises(ValueError, match="multiple repos"):
        deduplicate_entities(nodes, [], communities={})


def test_dedup_ok_with_single_repo():
    from graphify.dedup import deduplicate_entities
    nodes = [
        {"id": "repoA::userservice", "label": "UserService", "repo": "repoA"},
        {"id": "repoA::auth", "label": "Auth", "repo": "repoA"},
    ]
    result_nodes, result_edges = deduplicate_entities(nodes, [], communities={})
    assert len(result_nodes) == 2  # no false merge


def test_dedup_ok_with_no_repo_attr():
    from graphify.dedup import deduplicate_entities
    nodes = [
        {"id": "userservice", "label": "UserService"},
        {"id": "auth", "label": "Auth"},
    ]
    result_nodes, result_edges = deduplicate_entities(nodes, [], communities={})
    assert len(result_nodes) == 2


# ── merge-graphs prefix ───────────────────────────────────────────────────────

def test_merge_graphs_prefixes_ids(tmp_path):
    """merge-graphs should prefix node IDs with repo name to avoid silent collision."""
    from graphify.build import prefix_graph_for_global
    from networkx.readwrite import json_graph as jg

    # Two graphs with same node ID
    G1 = _make_graph([{"id": "userservice", "label": "UserService", "source_file": "src/user.py"}])
    G2 = _make_graph([{"id": "userservice", "label": "UserService", "source_file": "src/user.py"}])

    repo1 = tmp_path / "repo1" / "graphify-out"
    repo2 = tmp_path / "repo2" / "graphify-out"
    repo1.mkdir(parents=True)
    repo2.mkdir(parents=True)

    g1_path = repo1 / "graph.json"
    g2_path = repo2 / "graph.json"
    _graph_to_json(G1, g1_path)
    _graph_to_json(G2, g2_path)

    # Simulate what merge-graphs now does (prefix before compose)
    graphs = []
    graph_paths = [g1_path, g2_path]
    for gp in graph_paths:
        data = json.loads(gp.read_text())
        if "links" not in data and "edges" in data:
            data = dict(data, links=data["edges"])
        try:
            G = jg.node_link_graph(data, edges="links")
        except TypeError:
            G = jg.node_link_graph(data)
        repo_tag = gp.parent.parent.name
        graphs.append(prefix_graph_for_global(G, repo_tag))

    merged = nx.Graph()
    for G in graphs:
        merged = nx.compose(merged, G)

    assert "repo1::userservice" in merged.nodes
    assert "repo2::userservice" in merged.nodes
    assert merged.number_of_nodes() == 2  # no silent collapse


def test_global_add_rewires_edges_to_deduplicated_externals(tmp_path):
    """Edges incident to an external node that gets deduplicated against an
    already-present external must be rewired to the existing node, not dropped."""
    g1 = tmp_path / "graph1.json"
    g2 = tmp_path / "graph2.json"
    GA = _make_graph(
        [
            {"id": "moda", "label": "ModA", "source_file": "src/a.py"},
            {"id": "requests", "label": "requests"},
        ],
        [{"source": "moda", "target": "requests", "relation": "imports"}],
    )
    GB = _make_graph(
        [
            {"id": "modb", "label": "ModB", "source_file": "src/b.py"},
            {"id": "requests", "label": "requests"},
        ],
        [{"source": "modb", "target": "requests", "relation": "imports"}],
    )
    _graph_to_json(GA, g1)
    _graph_to_json(GB, g2)

    global_dir = tmp_path / ".graphify"
    with patch("graphify.global_graph._GLOBAL_DIR", global_dir), \
         patch("graphify.global_graph._GLOBAL_GRAPH", global_dir / "global-graph.json"), \
         patch("graphify.global_graph._GLOBAL_MANIFEST", global_dir / "global-manifest.json"):
        from graphify.global_graph import global_add, _load_global_graph
        global_add(g1, "repoA")
        global_add(g2, "repoB")
        G = _load_global_graph()

    # repoB's external "requests" was deduplicated against repoA's
    assert "repoA::requests" in G.nodes
    assert "repoB::requests" not in G.nodes
    # repoA's edge is untouched
    assert G.has_edge("repoA::moda", "repoA::requests")
    # repoB's edge must be rewired to the existing external node, not dropped
    assert G.has_edge("repoB::modb", "repoA::requests")
    assert G.edges["repoB::modb", "repoA::requests"]["relation"] == "imports"


def test_global_add_rejects_oversized_source_graph(monkeypatch, tmp_path):
    """#F4: global_add must refuse to read a source graph.json that
    exceeds the size cap, rather than json.loads-ing it into memory."""
    import pytest

    src_graph = tmp_path / "graph.json"
    G = _make_graph([{"id": "x", "label": "X", "source_file": "src/x.py"}])
    _graph_to_json(G, src_graph)

    global_dir = tmp_path / ".graphify"
    monkeypatch.setattr("graphify.security._MAX_GRAPH_FILE_BYTES", 8)
    with patch("graphify.global_graph._GLOBAL_DIR", global_dir), \
         patch("graphify.global_graph._GLOBAL_GRAPH", global_dir / "global-graph.json"), \
         patch("graphify.global_graph._GLOBAL_MANIFEST", global_dir / "global-manifest.json"):
        from graphify.global_graph import global_add
        with pytest.raises(ValueError, match="exceeds"):
            global_add(src_graph, "repoA")


def test_streaming_add_update_remove_matches_in_memory_reference(tmp_path):
    from graphify.build import prefix_graph_for_global, prune_repo_from_graph
    from graphify.global_graph import _load_global_graph, global_add, global_remove

    first_a = _make_graph(
        [{"id": "a", "label": "A", "source_file": "a.py"}, {"id": "x", "label": "X"}],
        [{"source": "a", "target": "x", "relation": "uses"}],
    )
    second_a = _make_graph(
        [{"id": "a", "label": "A2", "source_file": "a2.py"}], [],
    )
    graph_b = _make_graph([{"id": "b", "label": "B", "source_file": "b.py"}], [])
    first_a_path = tmp_path / "first-a.json"
    second_a_path = tmp_path / "second-a.json"
    graph_b_path = tmp_path / "b.json"
    _graph_to_json(first_a, first_a_path)
    _graph_to_json(second_a, second_a_path)
    _graph_to_json(graph_b, graph_b_path)

    expected = nx.Graph()
    for graph, tag in ((first_a, "repoA"), (graph_b, "repoB"), (second_a, "repoA")):
        prune_repo_from_graph(expected, tag)
        prefixed = prefix_graph_for_global(graph, tag)
        expected.add_nodes_from(prefixed.nodes(data=True))
        expected.add_edges_from(prefixed.edges(data=True))
    prune_repo_from_graph(expected, "repoB")

    with _global_store(tmp_path):
        global_add(first_a_path, "repoA")
        global_add(graph_b_path, "repoB")
        global_add(second_a_path, "repoA")
        global_remove("repoB")
        actual = _load_global_graph()

    assert dict(actual.nodes(data=True)) == dict(expected.nodes(data=True))
    assert dict(actual.edges(data=True)) == dict(expected.edges(data=True))


def test_global_mutations_stream_under_one_graph_manifest_lock(monkeypatch, tmp_path):
    from graphify import global_graph

    source = tmp_path / "source.json"
    _graph_to_json(_make_graph([{"id": "a", "source_file": "a.py"}]), source)
    events: list[str] = []

    class RecordingLock:
        def __init__(self, path):
            self.path = path

        def __enter__(self):
            events.append("enter")
            return self

        def __exit__(self, exc_type, exc_value, traceback):
            events.append("exit")

    real_emit = global_graph.emit_node_link_atomic
    real_save_manifest = global_graph._save_manifest

    def record_emit(*args):
        events.append("graph")
        return real_emit(*args)

    def record_manifest(*args):
        events.append("manifest")
        return real_save_manifest(*args)

    monkeypatch.setattr(global_graph, "FileLock", RecordingLock)
    monkeypatch.setattr(global_graph, "emit_node_link_atomic", record_emit)
    monkeypatch.setattr(global_graph, "_save_manifest", record_manifest)
    monkeypatch.setattr(
        global_graph,
        "_load_global_graph",
        lambda: pytest.fail("mutation materialized the persisted global graph"),
    )

    with _global_store(tmp_path):
        global_graph.global_add(source, "repoA")
        global_graph.global_remove("repoA")

    assert events == ["enter", "graph", "manifest", "exit", "enter", "graph", "manifest", "exit"]


@pytest.mark.parametrize(
    ("graph_factory", "directed", "multigraph"),
    ((nx.DiGraph, True, False), (nx.MultiGraph, False, True), (nx.MultiDiGraph, True, True)),
)
def test_fresh_global_store_adopts_source_schema_and_transformed_metadata(
    tmp_path, graph_factory, directed, multigraph
):
    from graphify.global_graph import global_add

    source_graph = graph_factory()
    source_graph.graph.update(
        {"name": graph_factory.__name__, "hyperedges": [{"id": "h", "nodes": ["a"]}]}
    )
    source_graph.add_node("a", source_file="a.py")
    source_graph.add_node("b", source_file="b.py")
    if multigraph:
        source_graph.add_edge("a", "b", key="source-key", relation="calls")
    else:
        source_graph.add_edge("a", "b", relation="calls")
    source = tmp_path / f"{graph_factory.__name__}.json"
    _graph_to_json(source_graph, source)

    with _global_store(tmp_path) as global_dir:
        global_add(source, "repoA")

    data = json.loads((global_dir / "global-graph.json").read_text(encoding="utf-8"))
    assert data["directed"] is directed
    assert data["multigraph"] is multigraph
    assert data["graph"] == {
        "name": graph_factory.__name__,
        "hyperedges": [{"id": "repoA::h", "nodes": ["repoA::a"]}],
    }
    assert "links" in data and "edges" not in data


def test_existing_global_schema_and_graph_metadata_win_over_source(tmp_path):
    from graphify.global_graph import global_add

    existing = nx.DiGraph(name="retained metadata")
    existing.add_node("retained", repo="other", source_file="retained.py")
    existing_graph = tmp_path / "existing.json"
    _graph_to_json(existing, existing_graph)
    source = nx.MultiDiGraph(name="source metadata")
    source.add_edge("a", "b", key="source-key")
    source_graph = tmp_path / "source.json"
    _graph_to_json(source, source_graph)

    with _global_store(tmp_path) as global_dir:
        global_dir.mkdir(parents=True)
        (global_dir / "global-graph.json").write_text(existing_graph.read_text(), encoding="utf-8")
        global_add(source_graph, "repoA")

    data = json.loads((global_dir / "global-graph.json").read_text(encoding="utf-8"))
    assert data["directed"] is True
    assert data["multigraph"] is False
    assert data["graph"] == {"name": "retained metadata"}


def test_external_dedup_uses_last_retained_label_and_remaps_direction_markers(tmp_path):
    from graphify.global_graph import _load_global_graph, global_add

    existing = nx.Graph()
    existing.add_node("external-first", label="requests")
    existing.add_node("external-last", label="requests")
    existing_graph = tmp_path / "existing.json"
    _graph_to_json(existing, existing_graph)
    source = _make_graph(
        [{"id": "module", "source_file": "module.py"}, {"id": "dependency", "label": "requests"}],
        [
            {
                "source": "module",
                "target": "dependency",
                "_src": "module",
                "_tgt": "dependency",
            }
        ],
    )
    source_graph = tmp_path / "source.json"
    _graph_to_json(source, source_graph)

    with _global_store(tmp_path) as global_dir:
        global_dir.mkdir(parents=True)
        (global_dir / "global-graph.json").write_text(existing_graph.read_text(), encoding="utf-8")
        global_add(source_graph, "repoA")
        graph = _load_global_graph()

    assert "repoA::dependency" not in graph
    edge = graph.edges["repoA::module", "external-last"]
    assert edge["_src"] == "repoA::module"
    assert edge["_tgt"] == "external-last"


def test_external_remap_drops_only_the_self_loop_it_creates(tmp_path):
    from graphify.global_graph import _load_global_graph, global_add

    existing = nx.Graph()
    existing.add_node("external", label="dependency")
    existing_graph = tmp_path / "existing.json"
    _graph_to_json(existing, existing_graph)
    source = _make_graph(
        [{"id": "first", "label": "dependency"}, {"id": "second", "label": "dependency"}],
        [{"source": "first", "target": "second"}],
    )
    source_graph = tmp_path / "source.json"
    _graph_to_json(source, source_graph)

    with _global_store(tmp_path) as global_dir:
        global_dir.mkdir(parents=True)
        (global_dir / "global-graph.json").write_text(existing_graph.read_text(), encoding="utf-8")
        global_add(source_graph, "repoA")
        graph = _load_global_graph()

    assert list(graph.edges) == []


@pytest.mark.parametrize(
    ("candidate_records", "remapped"),
    (
        (
            [{"id": "candidate", "label": "dependency"}, {"id": "candidate", "source_file": "internal.py"}],
            False,
        ),
        (
            [
                {"id": "candidate", "label": "dependency", "source_file": "internal.py"},
                {"id": "candidate", "label": "dependency"},
            ],
            False,
        ),
        (
            [{"id": "candidate", "label": "dependency"}, {"id": "candidate", "label": "renamed"}],
            False,
        ),
        (
            [
                {"id": "candidate", "label": "dependency", "source_file": "internal.py"},
                {"id": "candidate", "source_file": ""},
            ],
            True,
        ),
    ),
)
def test_duplicate_retained_external_classification_uses_merged_final_attrs(
    tmp_path, candidate_records, remapped
):
    from graphify.global_graph import _load_global_graph, global_add

    existing_data = {
        "directed": False,
        "multigraph": False,
        "graph": {},
        "nodes": candidate_records,
        "links": [],
    }
    source = _make_graph(
        [{"id": "module", "source_file": "module.py"}, {"id": "dependency", "label": "dependency"}],
        [{"source": "module", "target": "dependency"}],
    )
    source_graph = tmp_path / "source.json"
    _graph_to_json(source, source_graph)

    with _global_store(tmp_path) as global_dir:
        global_dir.mkdir(parents=True)
        (global_dir / "global-graph.json").write_text(json.dumps(existing_data), encoding="utf-8")
        global_add(source_graph, "repoA")
        graph = _load_global_graph()

    endpoint = "candidate" if remapped else "repoA::dependency"
    assert graph.has_edge("repoA::module", endpoint)
    assert ("repoA::dependency" not in graph) is remapped


def test_serialized_node_and_edge_collisions_merge_before_output(tmp_path):
    from graphify.global_graph import _load_global_graph, global_add

    source = tmp_path / "source.json"
    source.write_text(
        json.dumps(
            {
                "directed": False,
                "multigraph": False,
                "graph": {},
                "nodes": [
                    {"id": "a", "incoming_kept": "first", "changed": "old"},
                    {"id": "a", "changed": "new"},
                    {"id": "b"},
                ],
                "links": [
                    {"source": "a", "target": "b", "edge_kept": "first", "changed": "old"},
                    {"source": "b", "target": "a", "changed": "new"},
                ],
            }
        ),
        encoding="utf-8",
    )
    existing = nx.Graph()
    existing.add_node("repoA::a", retained="node")
    existing.add_node("repoA::b", retained="other node")
    existing.add_edge("repoA::a", "repoA::b", retained="edge", changed="retained")
    existing_graph = tmp_path / "existing.json"
    _graph_to_json(existing, existing_graph)

    with _global_store(tmp_path) as global_dir:
        global_dir.mkdir(parents=True)
        (global_dir / "global-graph.json").write_text(existing_graph.read_text(), encoding="utf-8")
        global_add(source, "repoA")
        serialized = json.loads((global_dir / "global-graph.json").read_text(encoding="utf-8"))
        graph = _load_global_graph()

    assert len([node for node in serialized["nodes"] if node["id"] == "repoA::a"]) == 1
    assert len(serialized["links"]) == 1
    assert graph.nodes["repoA::a"] == {
        "retained": "node",
        "incoming_kept": "first",
        "changed": "new",
        "repo": "repoA",
        "local_id": "a",
    }
    assert graph.edges["repoA::a", "repoA::b"] == {
        "retained": "edge",
        "edge_kept": "first",
        "changed": "new",
    }


def test_simple_source_allocates_networkx_compatible_keys_for_multigraph_target(tmp_path):
    from graphify.global_graph import _load_global_graph, global_add

    existing = nx.MultiGraph()
    existing.add_node("repoA::a")
    existing.add_node("repoA::b")
    existing.add_edge("repoA::a", "repoA::b", key=0, retained="zero")
    existing.add_edge("repoA::a", "repoA::b", key=2, retained="two")
    existing_graph = tmp_path / "existing.json"
    _graph_to_json(existing, existing_graph)
    source = _make_graph([{"id": "a"}, {"id": "b"}], [{"source": "a", "target": "b"}])
    source_graph = tmp_path / "source.json"
    _graph_to_json(source, source_graph)

    with _global_store(tmp_path) as global_dir:
        global_dir.mkdir(parents=True)
        (global_dir / "global-graph.json").write_text(existing_graph.read_text(), encoding="utf-8")
        global_add(source_graph, "repoA")
        graph = _load_global_graph()

    assert sorted(graph["repoA::a"]["repoA::b"]) == [0, 2, 3]


@pytest.mark.parametrize("edge_key", (7, "named"))
def test_simple_source_preserves_explicit_key_for_multigraph_target(tmp_path, edge_key):
    from graphify.global_graph import _load_global_graph, global_add

    existing = nx.MultiGraph()
    existing.add_node("repoA::a")
    existing.add_node("repoA::b")
    existing_graph = tmp_path / "existing.json"
    _graph_to_json(existing, existing_graph)
    source = _make_graph(
        [{"id": "a"}, {"id": "b"}],
        [{"source": "a", "target": "b", "key": edge_key}],
    )
    source_graph = tmp_path / "source.json"
    _graph_to_json(source, source_graph)

    with _global_store(tmp_path) as global_dir:
        global_dir.mkdir(parents=True)
        (global_dir / "global-graph.json").write_text(existing_graph.read_text(), encoding="utf-8")
        global_add(source_graph, "repoA")
        graph = _load_global_graph()

    assert set(graph["repoA::a"]["repoA::b"]) == {edge_key}


def test_explicit_simple_source_key_collision_precedes_generated_key_after_remap(tmp_path):
    from graphify.global_graph import _load_global_graph, global_add

    existing = nx.MultiGraph()
    existing.add_node("external-left", label="left")
    existing.add_node("external-right", label="right")
    for key in (0, 2, "explicit"):
        existing.add_edge("external-left", "external-right", key=key, retained=str(key))
    existing_graph = tmp_path / "existing.json"
    _graph_to_json(existing, existing_graph)
    source = _make_graph(
        [
            {"id": "left-one", "label": "left"},
            {"id": "right-one", "label": "right"},
            {"id": "left-two", "label": "left"},
            {"id": "right-two", "label": "right"},
        ],
        [
            {"source": "left-one", "target": "right-one", "key": "explicit"},
            {"source": "left-two", "target": "right-two"},
        ],
    )
    source_graph = tmp_path / "source.json"
    _graph_to_json(source, source_graph)

    with _global_store(tmp_path) as global_dir:
        global_dir.mkdir(parents=True)
        (global_dir / "global-graph.json").write_text(existing_graph.read_text(), encoding="utf-8")
        global_add(source_graph, "repoA")
        graph = _load_global_graph()

    assert set(graph["external-left"]["external-right"]) == {0, 2, 3, "explicit"}
    assert graph["external-left"]["external-right"]["explicit"]["retained"] == "explicit"


def test_retained_multigraph_key_index_ignores_5000_unrelated_pairs(tmp_path):
    from graphify.global_graph import _retained_multigraph_keys
    from graphify.node_link_stream import NodeLinkStream

    graph = nx.MultiGraph()
    for index in range(5_000):
        graph.add_edge(f"unrelated-left-{index}", f"unrelated-right-{index}", key=index)
    graph.add_edge("repoA::a", "repoA::b", key=0)
    graph.add_edge("repoA::a", "repoA::b", key=2)
    path = tmp_path / "global.json"
    _graph_to_json(graph, path)

    stream = NodeLinkStream(path)
    pair = frozenset(("repoA::a", "repoA::b"))
    keys = _retained_multigraph_keys(stream, stream.header(), set(), {pair})

    assert keys == {pair: {0, 2}}


def test_remapped_simple_source_uses_retained_multigraph_key_sequence(tmp_path):
    from graphify.global_graph import _load_global_graph, global_add

    existing = nx.MultiGraph()
    existing.add_node("repoA::module", source_file="retained.py")
    existing.add_node("external", label="dependency")
    existing.add_edge("repoA::module", "external", key=0, retained="zero")
    existing.add_edge("repoA::module", "external", key=2, retained="two")
    existing_graph = tmp_path / "existing.json"
    _graph_to_json(existing, existing_graph)
    source = _make_graph(
        [{"id": "module", "source_file": "module.py"}, {"id": "dependency", "label": "dependency"}],
        [{"source": "module", "target": "dependency"}],
    )
    source_graph = tmp_path / "source.json"
    _graph_to_json(source, source_graph)

    with _global_store(tmp_path) as global_dir:
        global_dir.mkdir(parents=True)
        (global_dir / "global-graph.json").write_text(existing_graph.read_text(), encoding="utf-8")
        global_add(source_graph, "repoA")
        graph = _load_global_graph()

    assert sorted(graph["repoA::module"]["external"]) == [0, 2, 3]


def test_multigraph_source_collapses_by_simple_target_identity_in_input_order(tmp_path):
    from graphify.global_graph import _load_global_graph, global_add

    existing = nx.Graph()
    existing.add_node("repoA::a")
    existing.add_node("repoA::b")
    existing.add_edge("repoA::a", "repoA::b", retained="yes", changed="retained")
    existing_graph = tmp_path / "existing.json"
    _graph_to_json(existing, existing_graph)
    source = nx.MultiGraph()
    source.add_edge("a", "b", key="first", edge_kept="first", changed="first")
    source.add_edge("a", "b", key="last", changed="last")
    source_graph = tmp_path / "source.json"
    _graph_to_json(source, source_graph)

    with _global_store(tmp_path) as global_dir:
        global_dir.mkdir(parents=True)
        (global_dir / "global-graph.json").write_text(existing_graph.read_text(), encoding="utf-8")
        global_add(source_graph, "repoA")
        serialized = json.loads((global_dir / "global-graph.json").read_text(encoding="utf-8"))
        graph = _load_global_graph()

    assert len(serialized["links"]) == 1
    assert graph.edges["repoA::a", "repoA::b"] == {
        "retained": "yes",
        "edge_kept": "first",
        "changed": "last",
    }


def test_unmatched_same_label_incoming_externals_remain_distinct(tmp_path):
    from graphify.global_graph import _load_global_graph, global_add

    existing = nx.Graph()
    existing.add_node("other", label="different")
    existing_graph = tmp_path / "existing.json"
    _graph_to_json(existing, existing_graph)
    source = _make_graph([{"id": "one", "label": "shared"}, {"id": "two", "label": "shared"}])
    source_graph = tmp_path / "source.json"
    _graph_to_json(source, source_graph)

    with _global_store(tmp_path) as global_dir:
        global_dir.mkdir(parents=True)
        (global_dir / "global-graph.json").write_text(existing_graph.read_text(), encoding="utf-8")
        global_add(source_graph, "repoA")
        graph = _load_global_graph()

    assert {"repoA::one", "repoA::two"}.issubset(graph.nodes)


def test_legacy_edges_input_is_streamed_and_output_as_authoritative_links(tmp_path):
    from graphify.global_graph import global_add

    source = tmp_path / "legacy.json"
    source.write_text(
        json.dumps(
            {
                "directed": False,
                "multigraph": False,
                "graph": {},
                "nodes": [{"id": "a"}, {"id": "b"}],
                "edges": [{"source": "a", "target": "b"}],
            }
        ),
        encoding="utf-8",
    )

    with _global_store(tmp_path) as global_dir:
        global_add(source, "repoA")

    data = json.loads((global_dir / "global-graph.json").read_text(encoding="utf-8"))
    assert data["links"] == [{"source": "repoA::a", "target": "repoA::b"}]
    assert "edges" not in data


def test_links_win_over_legacy_edges_when_both_are_present(tmp_path):
    from graphify.global_graph import global_add

    source = tmp_path / "both.json"
    source.write_text(
        json.dumps(
            {
                "directed": False,
                "multigraph": False,
                "graph": {},
                "nodes": [{"id": "a"}, {"id": "b"}],
                "edges": [{"source": "a", "target": "a"}],
                "links": [{"source": "a", "target": "b"}],
            }
        ),
        encoding="utf-8",
    )

    with _global_store(tmp_path) as global_dir:
        global_add(source, "repoA")

    data = json.loads((global_dir / "global-graph.json").read_text(encoding="utf-8"))
    assert data["links"] == [{"source": "repoA::a", "target": "repoA::b"}]


def test_corrupt_existing_graph_fails_closed_without_clobbering_it(tmp_path):
    from graphify.global_graph import global_add
    from graphify.node_link_stream import NodeLinkFormatError

    source = tmp_path / "source.json"
    _graph_to_json(_make_graph([{"id": "a"}]), source)
    corrupt = b'{"directed":false,"nodes":['

    with _global_store(tmp_path) as global_dir:
        global_dir.mkdir(parents=True)
        destination = global_dir / "global-graph.json"
        destination.write_bytes(corrupt)
        with pytest.raises(NodeLinkFormatError):
            global_add(source, "repoA")

    assert destination.read_bytes() == corrupt


def test_changed_source_across_stream_passes_fails_closed_without_clobber(monkeypatch, tmp_path):
    from graphify import global_graph
    from graphify.node_link_stream import NodeLinkChangedError

    source = tmp_path / "source.json"
    _graph_to_json(_make_graph([{"id": "a"}]), source)
    existing = _make_graph([{"id": "retained", "repo": "other"}])
    existing_graph = tmp_path / "existing.json"
    _graph_to_json(existing, existing_graph)
    real_links = global_graph.NodeLinkStream.links

    def mutate_after_links(stream):
        iterator = real_links(stream)
        for link in iterator:
            yield link
        if stream.path == source:
            source.write_text(source.read_text(encoding="utf-8") + " ", encoding="utf-8")

    monkeypatch.setattr(global_graph.NodeLinkStream, "links", mutate_after_links)
    with _global_store(tmp_path) as global_dir:
        global_dir.mkdir(parents=True)
        destination = global_dir / "global-graph.json"
        destination.write_text(existing_graph.read_text(), encoding="utf-8")
        before = destination.read_bytes()
        with pytest.raises(NodeLinkChangedError):
            global_graph.global_add(source, "repoA")

    assert destination.read_bytes() == before


def test_file_hash_reads_incrementally_without_path_read_bytes(monkeypatch, tmp_path):
    from graphify.global_graph import _file_hash

    source = tmp_path / "source.json"
    source.write_bytes(b"x" * 200_000)

    def fail_read_bytes(path):
        raise AssertionError("whole-file read must not be used for global source hashes")

    monkeypatch.setattr(Path, "read_bytes", fail_read_bytes)
    assert len(_file_hash(source)) == 16


def test_manifest_counts_source_final_nodes_edges_and_skips_unchanged_source(tmp_path):
    from graphify.global_graph import global_add

    source = tmp_path / "source.json"
    source.write_text(
        json.dumps(
            {
                "directed": False,
                "multigraph": False,
                "graph": {},
                "nodes": [{"id": "a"}, {"id": "a", "changed": "new"}, {"id": "b"}],
                "links": [{"source": "a", "target": "b"}, {"source": "b", "target": "a"}],
            }
        ),
        encoding="utf-8",
    )

    with _global_store(tmp_path) as global_dir:
        first = global_add(source, "repoA")
        skipped = global_add(source, "repoA")

    manifest = json.loads((global_dir / "global-manifest.json").read_text(encoding="utf-8"))
    assert first == {"repo_tag": "repoA", "nodes_added": 2, "nodes_removed": 0, "skipped": False}
    assert manifest["repos"]["repoA"]["node_count"] == 2
    assert manifest["repos"]["repoA"]["edge_count"] == 1
    assert skipped == {"repo_tag": "repoA", "nodes_added": 0, "nodes_removed": 0, "skipped": True}


@pytest.mark.parametrize(
    ("node_records", "expected_removed", "expected_edge"),
    (
        (
            [{"id": "duplicate", "repo": "repoA"}, {"id": "duplicate", "repo": "other"}],
            0,
            True,
        ),
        (
            [{"id": "duplicate", "repo": "other"}, {"id": "duplicate", "repo": "repoA"}],
            1,
            False,
        ),
    ),
)
def test_duplicate_retained_node_repo_membership_is_last_record_wins(
    tmp_path, node_records, expected_removed, expected_edge
):
    from graphify.global_graph import _load_global_graph, global_remove

    graph_data = {
        "directed": False,
        "multigraph": False,
        "graph": {},
        "nodes": [*node_records, {"id": "other", "repo": "other"}],
        "links": [{"source": "duplicate", "target": "other"}],
    }
    with _global_store(tmp_path) as global_dir:
        global_dir.mkdir(parents=True)
        (global_dir / "global-graph.json").write_text(json.dumps(graph_data), encoding="utf-8")
        (global_dir / "global-manifest.json").write_text(
            json.dumps({"version": 1, "repos": {"repoA": {}}}), encoding="utf-8"
        )
        removed = global_remove("repoA")
        graph = _load_global_graph()

    assert removed == expected_removed
    assert graph.has_edge("duplicate", "other") is expected_edge
    if expected_edge:
        assert graph.nodes["duplicate"]["repo"] == "other"
    else:
        assert "duplicate" not in graph
