"""Write generated frames and audio to an MP4 file (H.264 video, AAC audio) with PyAV."""

from pathlib import Path

import numpy as np

AAC_FRAME = 1024  # samples per AAC frame


def to_uint8(frames: np.ndarray) -> np.ndarray:
    """Frames in [0, 1] (float) to uint8, rounding like diffusers' `numpy_to_pil` (`(x * 255).round()`)."""
    if frames.dtype == np.uint8:
        return frames
    return (np.clip(frames, 0, 1) * 255).round().astype(np.uint8)


def write_mp4(
    path: str | Path,
    frames: np.ndarray,
    frame_rate: float = 24.0,
    audio: np.ndarray | None = None,
    sample_rate: int = 44_100,
    crf: int = 16,
) -> Path:
    """`frames`: `(F, H, W, 3)` in [0, 1] or uint8; `audio`: mono `(samples,)` in [-1, 1]."""
    import av

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    frames = to_uint8(frames)
    with av.open(str(path), "w") as container:
        video = container.add_stream("libx264", rate=round(frame_rate))
        video.height, video.width = frames.shape[1:3]
        video.pix_fmt = "yuv420p"
        video.options = {"crf": str(crf)}
        stream = None
        if audio is not None:
            stream = container.add_stream("aac", rate=sample_rate, layout="mono")
        for frame in frames:
            container.mux(video.encode(av.VideoFrame.from_ndarray(frame, format="rgb24")))
        container.mux(video.encode())
        if stream is not None:
            samples = np.clip(np.asarray(audio, np.float32).reshape(-1), -1, 1)
            for start in range(0, samples.size, AAC_FRAME):
                chunk = samples[start : start + AAC_FRAME][None]
                frame = av.AudioFrame.from_ndarray(np.ascontiguousarray(chunk), format="fltp", layout="mono")
                frame.sample_rate = sample_rate
                frame.pts = start
                container.mux(stream.encode(frame))
            container.mux(stream.encode())
    return path
