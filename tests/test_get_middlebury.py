"""Middlebury fetching and parsing.

The download itself is exercised over ``file://`` URLs. The parsing -- PFM,
calib.txt, scene assembly -- is tested against fixtures written here, so the
code path that turns a downloaded scene into a StereoScene is covered even
though this environment cannot reach vision.middlebury.edu.
"""

from __future__ import annotations

import struct
from pathlib import Path

import cv2
import numpy as np
import pytest

from tools.get_middlebury import (
    FILES,
    SCENES,
    Calibration,
    MiddleburyError,
    download_file,
    load_local_scenes,
    load_scene,
    main,
    parse_calib,
    read_pfm,
    scene_url,
)


def write_pfm(path: Path, image: np.ndarray, little_endian: bool = True) -> None:
    """Write a single-channel PFM the way Middlebury does: rows bottom-to-top."""
    height, width = image.shape
    scale = -1.0 if little_endian else 1.0
    dtype = "<f4" if little_endian else ">f4"
    with path.open("wb") as handle:
        handle.write(b"Pf\n")
        handle.write(f"{width} {height}\n".encode())
        handle.write(f"{scale}\n".encode())
        handle.write(np.flipud(image).astype(dtype).tobytes())


CALIB_TEXT = """cam0=[1758.23 0 872.36; 0 1758.23 552.32; 0 0 1]
cam1=[1758.23 0 1044.36; 0 1758.23 552.32; 0 0 1]
doffs=172.251
baseline=111.53
width=1920
height=1080
ndisp=290
isint=0
vmin=55
vmax=142
"""


# --------------------------------------------------------------------------
# PFM
# --------------------------------------------------------------------------


def test_pfm_round_trips(tmp_path: Path) -> None:
    image = np.arange(12, dtype=np.float32).reshape(3, 4)
    path = tmp_path / "d.pfm"
    write_pfm(path, image)
    assert np.array_equal(read_pfm(path), image)


def test_pfm_row_order_is_corrected(tmp_path: Path) -> None:
    # PFM stores bottom-to-top; a reader that forgets to flip produces a
    # vertically mirrored disparity map, which is subtly wrong rather than
    # obviously broken.
    image = np.array([[1.0, 1.0], [9.0, 9.0]], dtype=np.float32)
    path = tmp_path / "d.pfm"
    write_pfm(path, image)
    assert read_pfm(path)[0, 0] == 1.0
    assert read_pfm(path)[1, 0] == 9.0


def test_pfm_big_endian(tmp_path: Path) -> None:
    image = np.array([[1.5, 2.5]], dtype=np.float32)
    path = tmp_path / "d.pfm"
    write_pfm(path, image, little_endian=False)
    assert np.array_equal(read_pfm(path), image)


def test_pfm_infinities_become_nan(tmp_path: Path) -> None:
    # Middlebury marks unknown disparity as inf. Turning it into NaN means one
    # isfinite test covers every no-data case in the codebase.
    image = np.array([[np.inf, 3.0, -np.inf]], dtype=np.float32)
    path = tmp_path / "d.pfm"
    write_pfm(path, image)
    out = read_pfm(path)
    assert np.isnan(out[0, 0]) and np.isnan(out[0, 2])
    assert out[0, 1] == 3.0


def test_pfm_rejects_a_bad_header(tmp_path: Path) -> None:
    path = tmp_path / "bad.pfm"
    path.write_bytes(b"XX\n2 2\n-1.0\n" + b"\x00" * 16)
    with pytest.raises(MiddleburyError, match="not a PFM"):
        read_pfm(path)


def test_pfm_rejects_bad_dimensions(tmp_path: Path) -> None:
    path = tmp_path / "bad.pfm"
    path.write_bytes(b"Pf\nnot numbers\n-1.0\n")
    with pytest.raises(MiddleburyError, match="dimensions"):
        read_pfm(path)


def test_pfm_rejects_a_truncated_payload(tmp_path: Path) -> None:
    path = tmp_path / "short.pfm"
    path.write_bytes(b"Pf\n4 4\n-1.0\n" + struct.pack("<f", 1.0))
    with pytest.raises(MiddleburyError, match="truncated"):
        read_pfm(path)


# --------------------------------------------------------------------------
# calib.txt
# --------------------------------------------------------------------------


def test_parse_calib(tmp_path: Path) -> None:
    path = tmp_path / "calib.txt"
    path.write_text(CALIB_TEXT, encoding="utf-8")
    calibration = parse_calib(path)
    assert isinstance(calibration, Calibration)
    assert calibration.focal_px == pytest.approx(1758.23)
    assert calibration.baseline_m == pytest.approx(0.11153)  # mm -> m
    assert calibration.doffs == pytest.approx(172.251)
    assert (calibration.width, calibration.height) == (1920, 1080)
    assert calibration.ndisp == 290


def test_parse_calib_reports_a_broken_file(tmp_path: Path) -> None:
    path = tmp_path / "calib.txt"
    path.write_text("nothing=useful\n", encoding="utf-8")
    with pytest.raises(MiddleburyError, match="cannot parse"):
        parse_calib(path)


# --------------------------------------------------------------------------
# Scene assembly
# --------------------------------------------------------------------------


def _write_scene(directory: Path, width: int = 16, height: int = 8) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    image = np.zeros((height, width, 3), np.uint8)
    cv2.imwrite(str(directory / "im0.png"), image)
    cv2.imwrite(str(directory / "im1.png"), image)
    disparity = np.full((height, width), 20.0, dtype=np.float32)
    disparity[0, 0] = np.inf  # unknown
    write_pfm(directory / "disp0.pfm", disparity)
    (directory / "calib.txt").write_text(
        CALIB_TEXT.replace("width=1920", f"width={width}").replace(
            "height=1080", f"height={height}"
        ),
        encoding="utf-8",
    )
    return directory


def test_load_scene_builds_a_stereo_scene(tmp_path: Path) -> None:
    directory = _write_scene(tmp_path / "Motorcycle")
    scene = load_scene(directory)
    assert scene.name == "Motorcycle"
    assert scene.left.shape == (8, 16, 3)
    assert scene.baseline_m == pytest.approx(0.11153)
    # doffs is added back, otherwise Z = f*B/d is wrong by a constant.
    assert scene.disparity_gt[1, 1] == pytest.approx(20.0 + 172.251)
    assert np.isnan(scene.disparity_gt[0, 0])


def test_load_scene_reports_a_missing_file(tmp_path: Path) -> None:
    directory = _write_scene(tmp_path / "Piano")
    (directory / "disp0.pfm").unlink()
    with pytest.raises(MiddleburyError, match="disp0.pfm"):
        load_scene(directory)


def test_load_local_scenes_finds_complete_directories(tmp_path: Path) -> None:
    _write_scene(tmp_path / "Motorcycle")
    _write_scene(tmp_path / "Piano")
    (tmp_path / "Incomplete").mkdir()
    scenes = load_local_scenes(tmp_path)
    assert [s.name for s in scenes] == ["Motorcycle", "Piano"]


def test_load_local_scenes_tolerates_a_missing_root(tmp_path: Path) -> None:
    # No download is not an error: the committed baseline never needs one.
    assert load_local_scenes(tmp_path / "never_downloaded") == []


# --------------------------------------------------------------------------
# Download plumbing
# --------------------------------------------------------------------------


def test_scene_url_shape() -> None:
    url = scene_url("Motorcycle", "disp0.pfm")
    assert url.endswith("/Motorcycle-perfect/disp0.pfm")
    assert url.startswith("https://vision.middlebury.edu/")


def test_download_file_writes_the_payload(tmp_path: Path) -> None:
    source = tmp_path / "source.bin"
    source.write_bytes(b"payload")
    destination = tmp_path / "nested" / "out.bin"
    download_file(source.as_uri(), destination)
    assert destination.read_bytes() == b"payload"


def test_download_leaves_no_partial_file_on_failure(tmp_path: Path) -> None:
    # A half-written file that looks complete would silently poison the
    # benchmark, so the rename only happens on success.
    destination = tmp_path / "out.bin"
    missing = (tmp_path / "does_not_exist.bin").as_uri()
    with pytest.raises(MiddleburyError, match="failed to download"):
        download_file(missing, destination)
    assert not destination.exists()
    assert list(tmp_path.glob("*.part")) == []


def test_cli_list_prints_every_scene(capsys) -> None:
    assert main(["--list"]) == 0
    out = capsys.readouterr().out
    for scene in SCENES:
        assert scene in out
    for filename in FILES:
        assert filename in out


def test_cli_requires_an_action(capsys) -> None:
    with pytest.raises(SystemExit):
        main([])


def test_cli_rejects_an_unknown_scene() -> None:
    with pytest.raises(SystemExit):
        main(["--scene", "NotAScene"])
