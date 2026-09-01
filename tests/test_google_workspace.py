from pathlib import Path
import json

import pytest

import graphify.__main__ as mainmod
import graphify.google_workspace as gw


def test_read_google_shortcut_doc_id(tmp_path):
    shortcut = tmp_path / "Planning.gdoc"
    shortcut.write_text(
        '{"url":"https://docs.google.com/document/d/doc-123/edit","doc_id":"doc-123","email":"me@example.com"}',
        encoding="utf-8",
    )

    metadata = gw.read_google_shortcut(shortcut)

    assert metadata["file_id"] == "doc-123"
    assert metadata["account"] == "me@example.com"


def test_read_google_shortcut_extracts_id_from_url(tmp_path):
    shortcut = tmp_path / "Budget.gsheet"
    shortcut.write_text(
        '{"url":"https://docs.google.com/spreadsheets/d/sheet-456/edit?resourcekey=key-1"}',
        encoding="utf-8",
    )

    metadata = gw.read_google_shortcut(shortcut)

    assert metadata["file_id"] == "sheet-456"
    assert metadata["resource_key"] == "key-1"


def test_convert_gdoc_to_markdown_sidecar(tmp_path, monkeypatch):
    shortcut = tmp_path / "Planning.gdoc"
    shortcut.write_text(
        '{"url":"https://docs.google.com/document/d/doc-123/edit","doc_id":"doc-123"}',
        encoding="utf-8",
    )

    def fake_export(file_id, mime_type, output, resource_key=None):
        assert file_id == "doc-123"
        assert mime_type == "text/markdown"
        output.write_text("# Planning\n\nExported doc text.", encoding="utf-8")

    monkeypatch.setattr(gw, "_run_gws_export", fake_export)

    out = gw.convert_google_workspace_file(
        shortcut, tmp_path / "converted", allow_export=True,
    )

    assert out is not None
    assert out.suffix == ".md"
    content = out.read_text(encoding="utf-8")
    assert 'source_type: "google_workspace"' in content
    assert "# Planning" in content


def test_convert_gsheet_uses_xlsx_markdown_callback(tmp_path, monkeypatch):
    shortcut = tmp_path / "Budget.gsheet"
    shortcut.write_text('{"doc_id":"sheet-456"}', encoding="utf-8")

    def fake_export(file_id, mime_type, output, resource_key=None):
        assert file_id == "sheet-456"
        assert mime_type == "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        output.write_bytes(b"xlsx")

    monkeypatch.setattr(gw, "_run_gws_export", fake_export)

    out = gw.convert_google_workspace_file(
        shortcut,
        tmp_path / "converted",
        allow_export=True,
        xlsx_to_markdown=lambda path: "## Sheet: Main\n\n| A |\n| --- |\n| 1 |",
    )

    assert out is not None
    assert "## Sheet: Main" in out.read_text(encoding="utf-8")


def test_run_gws_export_uses_output_directory_as_cwd(tmp_path, monkeypatch):
    output = tmp_path / "converted" / "doc.md"
    calls = []

    class Result:
        returncode = 0
        stdout = ""
        stderr = ""

    def fake_run(cmd, **kwargs):
        calls.append((cmd, kwargs))
        return Result()

    monkeypatch.setattr(gw.shutil, "which", lambda name: "/usr/local/bin/gws")
    monkeypatch.setattr(gw.subprocess, "run", fake_run)

    gw._run_gws_export("doc-123", "text/markdown", output)

    assert output.parent.exists()
    cmd, kwargs = calls[0]
    assert kwargs["cwd"] == output.parent.resolve()
    assert cmd[:4] == ["/usr/local/bin/gws", "drive", "files", "export"]
    assert cmd[-2:] == ["-o", "doc.md"]


def test_run_gws_export_does_not_send_resource_key_as_query_param(tmp_path, monkeypatch):
    output = tmp_path / "converted" / "doc.md"
    calls = []

    class Result:
        returncode = 0
        stdout = ""
        stderr = ""

    def fake_run(cmd, **kwargs):
        calls.append(cmd)
        return Result()

    monkeypatch.setattr(gw.shutil, "which", lambda name: "/usr/local/bin/gws")
    monkeypatch.setattr(gw.subprocess, "run", fake_run)

    gw._run_gws_export("doc-123", "text/markdown", output, resource_key="rk-1")

    params = json.loads(calls[0][calls[0].index("--params") + 1])
    assert params == {"fileId": "doc-123", "mimeType": "text/markdown"}


def test_google_workspace_enabled_env(monkeypatch):
    monkeypatch.setenv("GRAPHIFY_GOOGLE_WORKSPACE", "yes")
    assert gw.google_workspace_enabled()

    monkeypatch.setenv("GRAPHIFY_GOOGLE_WORKSPACE", "0")
    assert not gw.google_workspace_enabled()


def test_convert_google_workspace_file_requires_explicit_opt_in(tmp_path, monkeypatch):
    shortcut = tmp_path / "Planning.gdoc"
    shortcut.write_text('{"doc_id":"doc-123"}', encoding="utf-8")

    def fail_if_read(path):
        raise AssertionError(f"unapproved shortcut was read: {path}")

    monkeypatch.setattr(gw, "read_google_shortcut", fail_if_read)

    with pytest.raises(PermissionError, match="explicit opt-in"):
        gw.convert_google_workspace_file(shortcut, tmp_path / "converted")


@pytest.mark.parametrize("opt_in", ("flag", "env"))
def test_extract_cli_grants_google_workspace_capability(tmp_path, monkeypatch, opt_in):
    project = tmp_path / "project"
    project.mkdir()
    (project / "app.py").write_text("def app():\n    return 1\n", encoding="utf-8")
    (project / "Planning.gdoc").write_text('{"doc_id":"doc-123"}', encoding="utf-8")
    calls: list[bool] = []

    def fake_convert(path, out_dir, *, allow_export=False, xlsx_to_markdown=None, root=None):
        calls.append(allow_export)
        out_dir.mkdir(parents=True, exist_ok=True)
        converted = out_dir / "planning.md"
        converted.write_text("# Planning\n", encoding="utf-8")
        return converted

    monkeypatch.setattr("graphify.detect.convert_google_workspace_file", fake_convert)
    monkeypatch.setattr(mainmod, "_check_skill_version", lambda _: None)
    monkeypatch.delenv("GRAPHIFY_GOOGLE_WORKSPACE", raising=False)
    argv = ["graphify", "extract", str(project), "--code-only", "--no-cluster"]
    if opt_in == "flag":
        argv.append("--google-workspace")
    else:
        monkeypatch.setenv("GRAPHIFY_GOOGLE_WORKSPACE", "1")
    monkeypatch.setattr(mainmod.sys, "argv", argv)

    try:
        mainmod.main()
    except SystemExit as exc:
        assert exc.code in (None, 0)

    assert calls == [True]
