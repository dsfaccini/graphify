# Video transcription using faster-whisper
# Converts video/audio files to text transcripts for graph extraction
from __future__ import annotations

import errno
import hashlib
import os
import stat
import tempfile
from pathlib import Path

from graphify.paths import out_path as _out_path


VIDEO_EXTENSIONS = {'.mp4', '.mov', '.webm', '.mkv', '.avi', '.m4v', '.mp3', '.wav', '.m4a', '.ogg'}
URL_PREFIXES = ('http://', 'https://', 'www.')

_DEFAULT_MODEL = "base"
_TRANSCRIPTS_DIR = str(_out_path("transcripts"))
_FALLBACK_PROMPT = "Use proper punctuation and paragraph breaks."
_ALLOW_UNSANDBOXED_URL_DOWNLOADS = "GRAPHIFY_ALLOW_UNSANDBOXED_URL_DOWNLOADS"
_YTDLP_MAX_FILESIZE = "GRAPHIFY_YTDLP_MAX_FILESIZE"
_DEFAULT_YTDLP_MAX_BYTES = 1024 * 1024 * 1024
_YTDLP_BUFFER_BYTES = 64 * 1024
_YTDLP_SOCKET_TIMEOUT_SECONDS = 30
_AUDIO_EXTENSIONS = tuple(sorted(VIDEO_EXTENSIONS | {'.opus', '.aac', '.flac'}))
_UNSUPPORTED_LINK_ERRNOS = frozenset({errno.EOPNOTSUPP, errno.ENOSYS, errno.EPERM, errno.EXDEV})


class _DownloadSizeLimitExceeded(OSError):
    """Raised when yt-dlp reports a cumulative download larger than the budget."""


def _model_name() -> str:
    return os.environ.get("GRAPHIFY_WHISPER_MODEL", _DEFAULT_MODEL)


def _get_whisper():
    try:
        from faster_whisper import WhisperModel
        return WhisperModel
    except ImportError as exc:
        raise ImportError(
            "Video transcription requires faster-whisper. "
            "Run: pip install 'graphifyy[video]'"
        ) from exc


def _get_yt_dlp():
    try:
        import yt_dlp
        return yt_dlp
    except ImportError as exc:
        raise ImportError(
            "YouTube/URL download requires yt-dlp. "
            "Run: pip install 'graphifyy[video]'"
        ) from exc


def _ytdlp_max_filesize() -> int:
    """Return the configured positive byte ceiling for an opted-in yt-dlp download."""
    raw = os.environ.get(_YTDLP_MAX_FILESIZE)
    if raw is None or not raw.strip():
        return _DEFAULT_YTDLP_MAX_BYTES

    value = raw.strip()
    if not value.isascii() or not value.isdecimal() or int(value) <= 0:
        raise ValueError(
            f"{_YTDLP_MAX_FILESIZE} must be a positive byte count, got {raw!r}."
        )
    return int(value)


def _require_url_download_capability() -> None:
    """Reject remote downloads unless the operator explicitly enables the capability."""
    if os.environ.get(_ALLOW_UNSANDBOXED_URL_DOWNLOADS) == "1":
        return
    raise PermissionError(
        "URL downloads are disabled because yt-dlp is not an SSRF sandbox. "
        "Download the media locally and transcribe the local file instead. "
        f"Only enable {_ALLOW_UNSANDBOXED_URL_DOWNLOADS}=1 in a trusted environment."
    )


def _download_progress_limit(max_bytes: int):
    """Return a yt-dlp hook that aborts a stream exceeding its cumulative budget."""
    def enforce(status: dict[str, object]) -> None:
        downloaded = status.get("downloaded_bytes")
        if isinstance(downloaded, int) and downloaded > max_bytes:
            raise _DownloadSizeLimitExceeded(
                f"yt-dlp reported {downloaded} bytes, exceeding the {max_bytes}-byte download limit."
            )

    return enforce


def _is_safe_audio_file(path: Path, max_bytes: int) -> bool:
    """Return whether *path* is a regular, non-symlink audio file within the budget."""
    try:
        metadata = path.lstat()
    except OSError:
        return False
    return stat.S_ISREG(metadata.st_mode) and metadata.st_size <= max_bytes


def _path_exists(path: Path) -> bool:
    """Return whether *path* exists, including a dangling symlink."""
    return path.exists() or path.is_symlink()


def _cached_audio_path(output_dir: Path, url_hash: str, max_bytes: int) -> Path | None:
    """Return a safe cached audio file or reject an unsafe cache entry."""
    for extension in _AUDIO_EXTENSIONS:
        candidate = output_dir / f"yt_{url_hash}{extension}"
        if _path_exists(_publishing_marker(candidate)):
            raise OSError(f"Audio publication is still in progress: {candidate}")
        if not _path_exists(candidate):
            continue
        if not _is_safe_audio_file(candidate, max_bytes):
            raise OSError(f"Refusing unsafe cached audio file: {candidate}")
        if _path_exists(_publishing_marker(candidate)):
            raise OSError(f"Audio publication is still in progress: {candidate}")
        return candidate
    return None


def _staged_audio_path(staging_dir: Path, url_hash: str, max_bytes: int) -> Path:
    """Return the one bounded regular file produced in the request staging directory."""
    candidates = sorted(staging_dir.glob(f"yt_{url_hash}.*"))
    if len(candidates) != 1:
        raise OSError(
            f"Expected one downloaded audio file in {staging_dir}, found {len(candidates)}."
        )
    candidate = candidates[0]
    if candidate.suffix not in _AUDIO_EXTENSIONS:
        raise OSError(f"Refusing unexpected downloaded file: {candidate}")
    if not _is_safe_audio_file(candidate, max_bytes):
        raise OSError(f"Refusing unsafe downloaded audio file: {candidate}")
    return candidate


def _remove_owned_file(path: Path, device: int, inode: int) -> None:
    """Remove *path* only when it still names the file created by this call."""
    try:
        metadata = path.lstat()
    except OSError:
        return
    if metadata.st_dev != device or metadata.st_ino != inode:
        return
    try:
        path.unlink()
    except OSError:
        return


def _publishing_marker(destination: Path) -> Path:
    """Return the marker that keeps Graphify cache readers off a fallback copy."""
    return destination.with_name(f".{destination.name}.partial")


def _exclusive_file(path: Path) -> tuple[int, int]:
    """Create *path* without replacement and return its device and inode."""
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        metadata = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    return metadata.st_dev, metadata.st_ino


def _copy_staged_audio_exclusively(staged: Path, destination: Path, max_bytes: int) -> tuple[int, int]:
    """Copy *staged* to a newly created destination in bounded chunks."""
    descriptor: int | None = None
    device: int | None = None
    inode: int | None = None
    copied = 0
    try:
        descriptor = os.open(destination, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        metadata = os.fstat(descriptor)
        device, inode = metadata.st_dev, metadata.st_ino
        with staged.open("rb") as source, os.fdopen(descriptor, "wb") as published:
            descriptor = None
            while chunk := source.read(_YTDLP_BUFFER_BYTES):
                copied += len(chunk)
                if copied > max_bytes:
                    raise OSError(
                        f"Copied {copied} bytes, exceeding the {max_bytes}-byte download limit."
                    )
                published.write(chunk)
    except BaseException:
        if device is not None and inode is not None:
            _remove_owned_file(destination, device, inode)
        raise
    finally:
        if descriptor is not None:
            os.close(descriptor)
    if device is None or inode is None:
        raise OSError(f"Unable to publish audio file: {destination}")
    return device, inode


def _publish_with_copy_fallback(staged: Path, destination: Path, max_bytes: int) -> Path:
    """Publish without hard links while keeping Graphify cache readers off the partial copy.

    The marker is a portable coordination signal for Graphify processes. Other processes
    can still observe the exclusive destination before the copy completes.
    """
    marker = _publishing_marker(destination)
    try:
        marker_device, marker_inode = _exclusive_file(marker)
    except FileExistsError as exc:
        raise OSError(f"Audio publication is already in progress: {destination}") from exc

    try:
        device, inode = _copy_staged_audio_exclusively(staged, destination, max_bytes)
        if not _is_safe_audio_file(destination, max_bytes):
            _remove_owned_file(destination, device, inode)
            raise OSError(f"Refusing unsafe published audio file: {destination}")
    finally:
        _remove_owned_file(marker, marker_device, marker_inode)

    staged.unlink()
    return destination


def _publish_staged_audio(staged: Path, destination: Path, max_bytes: int) -> Path:
    """Publish *staged* without replacing an existing destination."""
    if _path_exists(destination):
        raise OSError(f"Refusing to replace existing audio file: {destination}")

    metadata = staged.lstat()
    try:
        os.link(staged, destination, follow_symlinks=False)
    except FileExistsError as exc:
        raise OSError(f"Refusing to replace existing audio file: {destination}") from exc
    except NotImplementedError:
        return _publish_with_copy_fallback(staged, destination, max_bytes)
    except OSError as exc:
        if exc.errno in _UNSUPPORTED_LINK_ERRNOS:
            return _publish_with_copy_fallback(staged, destination, max_bytes)
        raise

    if not _is_safe_audio_file(destination, max_bytes):
        _remove_owned_file(destination, metadata.st_dev, metadata.st_ino)
        raise OSError(f"Refusing unsafe published audio file: {destination}")

    staged.unlink()
    return destination


def is_url(path: str) -> bool:
    """Return True if the string looks like a URL rather than a file path."""
    return any(path.startswith(p) for p in URL_PREFIXES)


def download_audio(url: str, output_dir: Path) -> Path:
    """Download audio-only stream from a URL using yt-dlp.

    Returns the path to the downloaded media file.
    URL downloads require an explicit trusted-environment capability.
    """
    _require_url_download_capability()

    from graphify.security import validate_url

    validate_url(url)  # blocks private IPs, bad schemes before yt-dlp runs
    max_bytes = _ytdlp_max_filesize()
    output_dir.mkdir(parents=True, exist_ok=True)

    # yt-dlp uses %(title)s which can be long/weird — use a stable name based on URL hash
    url_hash = hashlib.sha1(url.encode(), usedforsecurity=False).hexdigest()[:12]

    cached = _cached_audio_path(output_dir, url_hash, max_bytes)
    if cached is not None:
        print(f"  cached audio: {cached.name}")
        return cached

    yt_dlp = _get_yt_dlp()
    ydl_opts = {
        'format': 'bestaudio[ext=m4a]/bestaudio/best',
        'quiet': True,
        'no_warnings': True,
        'noplaylist': True,
        'max_filesize': max_bytes,
        'buffersize': _YTDLP_BUFFER_BYTES,
        'noresizebuffer': True,
        'continuedl': False,
        'retries': 0,
        'fragment_retries': 0,
        'extractor_retries': 0,
        'file_access_retries': 0,
        'socket_timeout': _YTDLP_SOCKET_TIMEOUT_SECONDS,
        'skip_unavailable_fragments': False,
        'progress_hooks': [_download_progress_limit(max_bytes)],
        'postprocessors': [],  # no ffmpeg needed — use native audio
    }

    print(f"  downloading audio: {url[:80]} ...", flush=True)
    with tempfile.TemporaryDirectory(prefix=f".yt_{url_hash}-", dir=output_dir) as staging:
        staging_dir = Path(staging)
        ydl_opts['outtmpl'] = str(staging_dir / f"yt_{url_hash}.%(ext)s")
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.extract_info(url, download=True)

        staged = _staged_audio_path(staging_dir, url_hash, max_bytes)
        destination = output_dir / staged.name
        return _publish_staged_audio(staged, destination, max_bytes)


def build_whisper_prompt(god_nodes: list[dict]) -> str:
    """Build a domain hint for Whisper from god nodes extracted from the corpus.

    Formats the top god node labels into a topic string for Whisper.
    The coding agent (Claude Code, Codex, etc.) generates the actual one-sentence
    domain hint from these labels and passes it via GRAPHIFY_WHISPER_PROMPT or
    as initial_prompt — no separate API call needed here.
    """
    if not god_nodes:
        return _FALLBACK_PROMPT

    override = os.environ.get("GRAPHIFY_WHISPER_PROMPT")
    if override:
        return override

    labels = [n.get("label", "") for n in god_nodes[:10] if n.get("label")]
    if not labels:
        return _FALLBACK_PROMPT

    topics = ", ".join(labels[:5])
    return f"Technical discussion about {topics}. Use proper punctuation and paragraph breaks."


def transcribe(
    video_path: Path | str,
    output_dir: Path | None = None,
    initial_prompt: str | None = None,
    force: bool = False,
) -> Path:
    """Transcribe a video/audio file or URL to a .txt transcript.

    If video_path is a URL, audio is downloaded first via yt-dlp.
    Returns the path to the saved transcript file.
    Uses cached transcript if it exists unless force=True.

    initial_prompt: domain hint for Whisper (built from corpus god nodes).
    force: re-transcribe even if transcript already exists.
    """
    out_dir = Path(output_dir) if output_dir else Path(_TRANSCRIPTS_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    if is_url(str(video_path)):
        audio_path = download_audio(str(video_path), out_dir / "downloads")
    else:
        audio_path = Path(video_path)

    transcript_path = _transcript_cache_path(video_path, audio_path, out_dir)
    if transcript_path.exists() and not force:
        return transcript_path

    WhisperModel = _get_whisper()
    model_name = _model_name()
    prompt = initial_prompt or _FALLBACK_PROMPT

    print(f"  transcribing {audio_path.name} (model={model_name}) ...", flush=True)
    model = WhisperModel(model_name, device="cpu", compute_type="int8")
    segments, info = model.transcribe(
        str(audio_path),
        beam_size=5,
        initial_prompt=prompt,
    )

    lines = [segment.text.strip() for segment in segments if segment.text.strip()]
    transcript = "\n".join(lines)

    transcript_path.write_text(transcript, encoding="utf-8")
    lang = info.language if hasattr(info, "language") else "unknown"
    print(f"  transcript saved -> {transcript_path} (lang={lang}, {len(lines)} segments)")
    return transcript_path


def _transcript_cache_path(
    video_path: Path | str,
    audio_path: Path,
    output_dir: Path,
) -> Path:
    """Return a collision-safe transcript path for the original media source.

    Local files key by canonical path; URLs key by their original text. A legacy
    stem-only transcript is intentionally not considered because two sources can
    share a stem and its provenance cannot be recovered safely.
    """
    source = str(video_path)
    if is_url(source):
        identity = source
    else:
        try:
            identity = str(Path(source).expanduser().resolve())
        except OSError:
            identity = str(Path(source).expanduser().absolute())
    source_hash = hashlib.sha1(
        identity.encode("utf-8"), usedforsecurity=False,
    ).hexdigest()[:12]
    return output_dir / f"{audio_path.stem}-{source_hash}.txt"


def transcribe_all(
    video_files: list[str],
    output_dir: Path | None = None,
    initial_prompt: str | None = None,
    force: bool = False,
) -> list[str]:
    """Transcribe a list of video/audio files or URLs, return paths to transcript .txt files.

    Already-transcribed files are returned from cache unless force is true.
    initial_prompt is shared across all files — built once from corpus god nodes.
    """
    if not video_files:
        return []

    transcript_paths = []
    for vf in video_files:
        try:
            t = transcribe(vf, output_dir, initial_prompt=initial_prompt, force=force)
            transcript_paths.append(str(t))
        except Exception as exc:
            print(f"  warning: could not transcribe {vf}: {exc}")
    return transcript_paths
