"""Tests for graphify.transcribe — video/audio transcription support."""
from __future__ import annotations

import errno
import hashlib
import os
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

import graphify.transcribe as transcribe_module
from graphify.transcribe import (
    VIDEO_EXTENSIONS,
    build_whisper_prompt,
    download_audio,
    transcribe,
    transcribe_all,
)


# ---------------------------------------------------------------------------
# VIDEO_EXTENSIONS
# ---------------------------------------------------------------------------

def test_video_extensions_set():
    assert ".mp4" in VIDEO_EXTENSIONS
    assert ".mp3" in VIDEO_EXTENSIONS
    assert ".wav" in VIDEO_EXTENSIONS
    assert ".mov" in VIDEO_EXTENSIONS
    assert ".py" not in VIDEO_EXTENSIONS


# ---------------------------------------------------------------------------
# build_whisper_prompt
# ---------------------------------------------------------------------------

def test_build_whisper_prompt_no_nodes():
    """Empty god_nodes returns fallback prompt."""
    prompt = build_whisper_prompt([])
    assert "punctuation" in prompt.lower() or len(prompt) > 0


def test_build_whisper_prompt_env_override(monkeypatch):
    """GRAPHIFY_WHISPER_PROMPT env var short-circuits LLM call."""
    monkeypatch.setenv("GRAPHIFY_WHISPER_PROMPT", "Custom domain hint.")
    prompt = build_whisper_prompt([{"label": "Python"}, {"label": "FastAPI"}])
    assert prompt == "Custom domain hint."


def test_build_whisper_prompt_returns_topic_string():
    """Returns a topic-based prompt from god node labels — no LLM call."""
    god_nodes = [{"label": "neural networks"}, {"label": "transformers"}, {"label": "attention"}]
    with patch.dict(os.environ, {}, clear=False):
        os.environ.pop("GRAPHIFY_WHISPER_PROMPT", None)
        prompt = build_whisper_prompt(god_nodes)
    assert "neural networks" in prompt.lower() or "transformers" in prompt.lower()
    assert "punctuation" in prompt.lower()


def test_build_whisper_prompt_nodes_without_labels():
    """Nodes missing 'label' keys are safely skipped."""
    god_nodes = [{"id": "1"}, {"id": "2", "label": ""}]
    prompt = build_whisper_prompt(god_nodes)
    assert len(prompt) > 0


# ---------------------------------------------------------------------------
# download_audio
# ---------------------------------------------------------------------------

class _FakeYDL:
    options: dict[str, object] = {}
    output_writer: Callable[[Path], None] | None = None
    output_extension = "m4a"

    def __init__(self, options: dict[str, object]) -> None:
        type(self).options = options

    def __enter__(self):
        return self

    def __exit__(self, *args: object) -> bool:
        return False

    def extract_info(self, url: str, *, download: bool) -> dict[str, str]:
        assert url == "https://example.com/video"
        assert download is True
        output_template = self.options["outtmpl"]
        assert isinstance(output_template, str)
        output = Path(output_template.replace("%(ext)s", type(self).output_extension))
        writer = type(self).output_writer
        assert writer is not None
        writer(output)
        return {"ext": "m4a"}


def _url_hash(url: str = "https://example.com/video") -> str:
    return hashlib.sha1(url.encode(), usedforsecurity=False).hexdigest()[:12]


def _prepare_url_download(
    monkeypatch: pytest.MonkeyPatch,
    writer: Callable[[Path], None],
    *,
    max_bytes: int = 12345,
    output_extension: str = "m4a",
) -> None:
    monkeypatch.setenv("GRAPHIFY_ALLOW_UNSANDBOXED_URL_DOWNLOADS", "1")
    monkeypatch.setenv("GRAPHIFY_YTDLP_MAX_FILESIZE", str(max_bytes))
    monkeypatch.setattr("graphify.security.validate_url", lambda url: url)
    _FakeYDL.output_writer = writer
    _FakeYDL.output_extension = output_extension
    monkeypatch.setattr(
        transcribe_module,
        "_get_yt_dlp",
        lambda: SimpleNamespace(YoutubeDL=_FakeYDL),
    )


def test_download_audio_requires_capability_before_url_validation_or_ytdlp(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.delenv("GRAPHIFY_ALLOW_UNSANDBOXED_URL_DOWNLOADS", raising=False)

    def unexpected_url_validation(url: str) -> str:
        raise AssertionError(f"validate_url called for {url}")

    def unexpected_ytdlp_import() -> object:
        raise AssertionError("yt-dlp import attempted")

    monkeypatch.setattr("graphify.security.validate_url", unexpected_url_validation)
    monkeypatch.setattr(transcribe_module, "_get_yt_dlp", unexpected_ytdlp_import)

    with pytest.raises(PermissionError, match="yt-dlp is not an SSRF sandbox") as exc_info:
        download_audio("https://example.com/video", tmp_path)

    assert "Download the media locally" in str(exc_info.value)


def test_ytdlp_max_filesize_requires_a_positive_decimal(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("GRAPHIFY_YTDLP_MAX_FILESIZE", "not-a-number")

    with pytest.raises(ValueError, match="positive byte count"):
        transcribe_module._ytdlp_max_filesize()


def test_ytdlp_max_filesize_defaults_to_one_gib(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("GRAPHIFY_YTDLP_MAX_FILESIZE", raising=False)

    assert transcribe_module._ytdlp_max_filesize() == 1024 * 1024 * 1024


def test_download_audio_uses_bounded_opted_in_options(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    def write_audio(output: Path) -> None:
        output.write_bytes(b"audio")

    _prepare_url_download(monkeypatch, write_audio)

    result = download_audio("https://example.com/video", tmp_path)

    assert result == tmp_path / f"yt_{_url_hash()}.m4a"
    assert result.read_bytes() == b"audio"
    assert list(tmp_path.iterdir()) == [result]
    options = _FakeYDL.options
    assert options["max_filesize"] == 12345
    assert options["buffersize"] == 64 * 1024
    assert options["noresizebuffer"] is True
    assert options["continuedl"] is False
    assert options["retries"] == 0
    assert options["fragment_retries"] == 0
    assert options["extractor_retries"] == 0
    assert options["file_access_retries"] == 0
    assert options["socket_timeout"] == 30
    assert options["skip_unavailable_fragments"] is False
    assert options["noplaylist"] is True

    progress_hooks = options["progress_hooks"]
    assert isinstance(progress_hooks, list)
    hook = progress_hooks[0]
    assert callable(hook)
    hook({"downloaded_bytes": 12345})
    with pytest.raises(OSError, match="download limit"):
        hook({"downloaded_bytes": 12346})


@pytest.mark.parametrize("extension", ["mkv", "mp4", "aac", "flac"])
def test_download_audio_accepts_legitimate_ytdlp_audio_extensions(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    extension: str,
):
    def write_audio(output: Path) -> None:
        output.write_bytes(b"audio")

    _prepare_url_download(monkeypatch, write_audio, output_extension=extension)

    result = download_audio("https://example.com/video", tmp_path)

    assert result.suffix == f".{extension}"
    assert result.read_bytes() == b"audio"


@pytest.mark.parametrize("extension", ["mkv", "mp4", "aac", "flac"])
def test_download_audio_uses_safe_cached_ytdlp_audio_extensions(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    extension: str,
):
    monkeypatch.setenv("GRAPHIFY_ALLOW_UNSANDBOXED_URL_DOWNLOADS", "1")
    monkeypatch.setenv("GRAPHIFY_YTDLP_MAX_FILESIZE", "5")
    monkeypatch.setattr("graphify.security.validate_url", lambda url: url)
    cached = tmp_path / f"yt_{_url_hash()}.{extension}"
    cached.write_bytes(b"audio")

    def unexpected_ytdlp_import() -> object:
        raise AssertionError("yt-dlp import attempted")

    monkeypatch.setattr(transcribe_module, "_get_yt_dlp", unexpected_ytdlp_import)

    assert download_audio("https://example.com/video", tmp_path) == cached


def _raise_unsupported_link(source: Path, target: Path, *, follow_symlinks: bool) -> None:
    raise OSError(errno.EOPNOTSUPP, "hard links are unsupported")


def test_download_audio_uses_copy_fallback_when_hard_links_are_unsupported(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    def write_audio(output: Path) -> None:
        output.write_bytes(b"audio")

    _prepare_url_download(monkeypatch, write_audio)
    monkeypatch.setattr(transcribe_module.os, "link", _raise_unsupported_link)

    result = download_audio("https://example.com/video", tmp_path)

    assert result.read_bytes() == b"audio"
    assert not transcribe_module._publishing_marker(result).exists()


def test_copy_fallback_bounds_the_copy_and_cleans_up(tmp_path: Path):
    staged = tmp_path / "staged.m4a"
    destination = tmp_path / "published.m4a"
    staged.write_bytes(b"audio")

    with pytest.raises(OSError, match="download limit"):
        transcribe_module._copy_staged_audio_exclusively(staged, destination, max_bytes=4)

    assert not destination.exists()


def test_copy_fallback_never_overwrites_an_existing_destination(tmp_path: Path):
    staged = tmp_path / "staged.m4a"
    destination = tmp_path / "published.m4a"
    staged.write_bytes(b"audio")
    destination.write_bytes(b"existing audio")

    with pytest.raises(FileExistsError):
        transcribe_module._publish_with_copy_fallback(staged, destination, max_bytes=12345)

    assert destination.read_bytes() == b"existing audio"
    assert not transcribe_module._publishing_marker(destination).exists()


def test_copy_fallback_never_overwrites_a_destination_created_during_publish(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    def write_audio(output: Path) -> None:
        output.write_bytes(b"audio")

    _prepare_url_download(monkeypatch, write_audio)
    monkeypatch.setattr(transcribe_module.os, "link", _raise_unsupported_link)
    destination = tmp_path / f"yt_{_url_hash()}.m4a"
    original_open = transcribe_module.os.open

    def open_after_creating_destination(
        path: str | Path,
        flags: int,
        mode: int = 0o777,
        **kwargs: object,
    ) -> int:
        if path == destination:
            destination.write_bytes(b"existing audio")
        return original_open(path, flags, mode, **kwargs)

    monkeypatch.setattr(transcribe_module.os, "open", open_after_creating_destination)

    with pytest.raises(FileExistsError):
        download_audio("https://example.com/video", tmp_path)

    assert destination.read_bytes() == b"existing audio"
    assert not transcribe_module._publishing_marker(destination).exists()


def test_cache_rejects_an_audio_file_under_a_fallback_publication_marker(tmp_path: Path):
    candidate = tmp_path / f"yt_{_url_hash()}.m4a"
    marker = transcribe_module._publishing_marker(candidate)
    marker.write_bytes(b"")

    with pytest.raises(OSError, match="still in progress"):
        transcribe_module._cached_audio_path(tmp_path, _url_hash(), max_bytes=12345)


def test_cache_rechecks_marker_before_returning_an_interleaved_partial_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    extension = transcribe_module._AUDIO_EXTENSIONS[0]
    candidate = tmp_path / f"yt_{_url_hash()}{extension}"
    marker = transcribe_module._publishing_marker(candidate)
    original_path_exists = transcribe_module._path_exists
    publication_started = False

    def path_exists_after_publication_starts(path: Path) -> bool:
        nonlocal publication_started
        if path == candidate and not publication_started:
            publication_started = True
            candidate.write_bytes(b"partial")
            marker.write_bytes(b"")
        return original_path_exists(path)

    monkeypatch.setattr(transcribe_module, "_path_exists", path_exists_after_publication_starts)

    with pytest.raises(OSError, match="still in progress"):
        transcribe_module._cached_audio_path(tmp_path, _url_hash(), max_bytes=12345)

    assert candidate.read_bytes() == b"partial"


def test_download_audio_removes_a_failed_published_destination(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    def write_audio(output: Path) -> None:
        output.write_bytes(b"audio")

    _prepare_url_download(monkeypatch, write_audio)
    original_is_safe_audio_file = transcribe_module._is_safe_audio_file

    def reject_published_audio(path: Path, max_bytes: int) -> bool:
        if path.parent == tmp_path:
            return False
        return original_is_safe_audio_file(path, max_bytes)

    monkeypatch.setattr(transcribe_module, "_is_safe_audio_file", reject_published_audio)

    with pytest.raises(OSError, match="unsafe published audio"):
        download_audio("https://example.com/video", tmp_path)

    assert list(tmp_path.iterdir()) == []


def test_copy_fallback_removes_a_failed_published_destination(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    def write_audio(output: Path) -> None:
        output.write_bytes(b"audio")

    _prepare_url_download(monkeypatch, write_audio)
    monkeypatch.setattr(transcribe_module.os, "link", _raise_unsupported_link)
    original_is_safe_audio_file = transcribe_module._is_safe_audio_file

    def reject_published_audio(path: Path, max_bytes: int) -> bool:
        if path.parent == tmp_path:
            return False
        return original_is_safe_audio_file(path, max_bytes)

    monkeypatch.setattr(transcribe_module, "_is_safe_audio_file", reject_published_audio)

    with pytest.raises(OSError, match="unsafe published audio"):
        download_audio("https://example.com/video", tmp_path)

    assert list(tmp_path.iterdir()) == []


def test_download_audio_never_overwrites_destination_created_during_publish(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    def write_audio(output: Path) -> None:
        output.write_bytes(b"audio")

    _prepare_url_download(monkeypatch, write_audio)
    destination = tmp_path / f"yt_{_url_hash()}.m4a"
    original_link = transcribe_module.os.link

    def link_after_creating_destination(source: Path, target: Path, *, follow_symlinks: bool) -> None:
        target.write_bytes(b"existing audio")
        original_link(source, target, follow_symlinks=follow_symlinks)

    monkeypatch.setattr(transcribe_module.os, "link", link_after_creating_destination)

    with pytest.raises(OSError, match="Refusing to replace existing audio file"):
        download_audio("https://example.com/video", tmp_path)

    assert destination.read_bytes() == b"existing audio"
    assert list(tmp_path.iterdir()) == [destination]


@pytest.mark.parametrize("kind", ["symlink", "directory", "oversize"])
def test_download_audio_rejects_unsafe_cached_audio_before_ytdlp_import(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    kind: str,
):
    monkeypatch.setenv("GRAPHIFY_ALLOW_UNSANDBOXED_URL_DOWNLOADS", "1")
    monkeypatch.setenv("GRAPHIFY_YTDLP_MAX_FILESIZE", "4")
    monkeypatch.setattr("graphify.security.validate_url", lambda url: url)
    candidate = tmp_path / f"yt_{_url_hash()}.m4a"
    if kind == "symlink":
        target = tmp_path.parent / "audio.m4a"
        target.write_bytes(b"audio")
        candidate.symlink_to(target)
    elif kind == "directory":
        candidate.mkdir()
    else:
        candidate.write_bytes(b"oversize")

    def unexpected_ytdlp_import() -> object:
        raise AssertionError("yt-dlp import attempted")

    monkeypatch.setattr(transcribe_module, "_get_yt_dlp", unexpected_ytdlp_import)

    with pytest.raises(OSError, match="unsafe cached audio"):
        download_audio("https://example.com/video", tmp_path)


@pytest.mark.parametrize("kind", ["symlink", "directory", "oversize", "unexpected"])
def test_download_audio_rejects_unsafe_staged_output_and_cleans_up(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    kind: str,
):
    def write_unsafe_audio(output: Path) -> None:
        if kind == "symlink":
            target = tmp_path.parent / "audio.m4a"
            target.write_bytes(b"audio")
            output.symlink_to(target)
        elif kind == "directory":
            output.mkdir()
        elif kind == "unexpected":
            output.with_suffix(".part").write_bytes(b"audio")
        else:
            output.write_bytes(b"oversize")

    _prepare_url_download(monkeypatch, write_unsafe_audio, max_bytes=4)

    with pytest.raises(OSError, match="unsafe downloaded audio|unexpected downloaded file"):
        download_audio("https://example.com/video", tmp_path)

    assert list(tmp_path.iterdir()) == []


# ---------------------------------------------------------------------------
# transcribe
# ---------------------------------------------------------------------------

def test_transcribe_uses_cache(tmp_path):
    """If transcript already exists, transcribe() returns cached path without running Whisper."""
    video = tmp_path / "lecture.mp4"
    video.write_bytes(b"fake")
    out_dir = tmp_path / "transcripts"
    out_dir.mkdir()
    cached = out_dir / "lecture.txt"
    cached.write_text("Cached transcript content.")

    result = transcribe(video, output_dir=out_dir)
    assert result == cached


def test_transcribe_force_reruns(tmp_path):
    """force=True re-transcribes even when cache exists."""
    video = tmp_path / "talk.mp4"
    video.write_bytes(b"fake")
    out_dir = tmp_path / "transcripts"
    out_dir.mkdir()
    (out_dir / "talk.txt").write_text("Old transcript.")

    fake_segment = MagicMock()
    fake_segment.text = "New transcript segment."
    fake_info = MagicMock()
    fake_info.language = "en"

    fake_model = MagicMock()
    fake_model.transcribe.return_value = ([fake_segment], fake_info)

    with patch("graphify.transcribe._get_whisper", return_value=lambda *a, **kw: fake_model):
        result = transcribe(video, output_dir=out_dir, force=True)

    assert result.read_text() == "New transcript segment."


def test_transcribe_missing_faster_whisper(tmp_path):
    """ImportError propagates when faster_whisper is not installed."""
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"fake")

    with patch("graphify.transcribe._get_whisper", side_effect=ImportError("faster-whisper not installed")):
        with pytest.raises(ImportError):
            transcribe(video, output_dir=tmp_path / "out")


# ---------------------------------------------------------------------------
# transcribe_all
# ---------------------------------------------------------------------------

def test_transcribe_all_empty():
    """Empty input returns empty list without error."""
    assert transcribe_all([]) == []


def test_transcribe_all_uses_cache(tmp_path):
    """transcribe_all() returns cached paths for already-transcribed files."""
    video = tmp_path / "lecture.mp4"
    video.write_bytes(b"fake")
    out_dir = tmp_path / "transcripts"
    out_dir.mkdir()
    cached = out_dir / "lecture.txt"
    cached.write_text("Cached.")

    results = transcribe_all([str(video)], output_dir=out_dir)
    assert len(results) == 1
    assert str(cached) in results[0]


def test_transcribe_all_skips_failed(tmp_path):
    """transcribe_all() warns and skips files that fail to transcribe."""
    video = tmp_path / "broken.mp4"
    video.write_bytes(b"fake")

    def raise_import(*args, **kwargs):
        raise ImportError("faster_whisper not installed")

    with patch("graphify.transcribe.transcribe", side_effect=RuntimeError("boom")):
        results = transcribe_all([str(video)], output_dir=tmp_path / "out")

    assert results == []
