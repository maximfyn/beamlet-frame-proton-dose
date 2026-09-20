"""The streamed 4-D MetaImage must be byte-for-byte equivalent to JoinSeries.

`submission/inference.py` writes the output stack itself rather than through
`sitk.JoinSeries`, because JoinSeries needs the whole stack resident and then
copies it — which OOM-killed the container at the scored 50-frame layout. That
buys the memory back only if the hand-written file is genuinely the same image,
so this compares the two directly: voxels, geometry, and the
``CompressedData = True`` header the evaluator checks for
(`evaluate.py:is_compressed_mha`, and an uncompressed output counts as an
implementation error against every beam in the stack).
"""

import sys
import zlib
from pathlib import Path

import numpy as np
import pytest
import SimpleITK as sitk

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "submission"))

from submission.inference import (  # noqa: E402
    StreamingStackWriter, _adler_combine, _zlib_header,
)


def reference_image(shape=(4, 5, 6)) -> sitk.Image:
    """A non-trivial grid: anisotropic spacing and a non-zero origin, so a
    writer that silently drops geometry cannot pass."""
    image = sitk.GetImageFromArray(np.zeros(shape, dtype=np.float32))
    image.SetSpacing((1.0, 1.0, 3.0))
    image.SetOrigin((-237.0, -231.0, -156.0))
    return image


@pytest.mark.parametrize("n_frames", [1, 3, 7])
def test_matches_joinseries(tmp_path, n_frames):
    shape = (4, 5, 6)
    reference = reference_image(shape)
    rng = np.random.default_rng(0)
    volumes = [rng.random(shape, dtype=np.float32) for _ in range(n_frames)]

    streamed = tmp_path / "streamed.mha"
    writer = StreamingStackWriter(streamed, reference, n_frames)
    for volume in volumes:
        writer.add(volume)
    writer.close()

    frames = []
    for volume in volumes:
        frame = sitk.GetImageFromArray(volume)
        frame.CopyInformation(reference)
        frames.append(frame)
    expected_path = tmp_path / "joined.mha"
    sitk.WriteImage(sitk.JoinSeries(frames), str(expected_path), useCompression=True)

    got = sitk.ReadImage(str(streamed))
    expected = sitk.ReadImage(str(expected_path))

    assert got.GetSize() == expected.GetSize()
    assert got.GetSpacing() == pytest.approx(expected.GetSpacing())
    assert got.GetOrigin() == pytest.approx(expected.GetOrigin())
    assert got.GetDirection() == pytest.approx(expected.GetDirection())
    np.testing.assert_array_equal(
        sitk.GetArrayFromImage(got), sitk.GetArrayFromImage(expected)
    )


def test_a_rotated_direction_reads_back_unchanged(tmp_path):
    """MetaIO stores the direction column-major. A symmetric matrix, like the
    identity every DoseRAD CT carries, hides a writer that forgets that."""
    reference = reference_image()
    reference.SetDirection((0.0, 1.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 1.0))
    path = tmp_path / "rotated.mha"
    writer = StreamingStackWriter(path, reference, 1)
    writer.add(np.zeros((4, 5, 6), dtype=np.float32))
    writer.close()
    direction = np.asarray(sitk.ReadImage(str(path)).GetDirection()).reshape(4, 4)
    np.testing.assert_allclose(direction[:3, :3],
                               np.asarray(reference.GetDirection()).reshape(3, 3))


def test_frames_land_at_their_own_index(tmp_path):
    """Ordering is the failure that would not look like a failure: a shuffled
    stack still reads back as a valid image, it just scores as noise."""
    shape = (4, 5, 6)
    reference = reference_image(shape)
    volumes = [np.full(shape, float(i), dtype=np.float32) for i in range(5)]

    path = tmp_path / "ordered.mha"
    writer = StreamingStackWriter(path, reference, len(volumes))
    for volume in volumes:
        writer.add(volume)
    writer.close()

    stacked = sitk.GetArrayFromImage(sitk.ReadImage(str(path)))
    assert stacked.shape == (5, *shape)
    for i in range(5):
        assert stacked[i].min() == i and stacked[i].max() == i


def test_header_declares_compression(tmp_path):
    reference = reference_image()
    path = tmp_path / "compressed.mha"
    writer = StreamingStackWriter(path, reference, 2)
    for _ in range(2):
        writer.add(np.zeros((4, 5, 6), dtype=np.float32))
    writer.close()

    header = path.read_bytes()[:4096]
    assert b"CompressedData = True" in header       # evaluate.py greps for this
    assert b"NDims = 4" in header
    declared = int(header.split(b"CompressedDataSize = ")[1].split(b"\n")[0])
    assert declared > 0


def test_short_stack_is_refused(tmp_path):
    """A slot missing frames must fail loudly here rather than produce a file
    whose DimSize lies about its own contents."""
    writer = StreamingStackWriter(tmp_path / "short.mha", reference_image(), 3)
    writer.add(np.zeros((4, 5, 6), dtype=np.float32))
    with pytest.raises(ValueError, match="wrote 1 of 3"):
        writer.close()


def test_wrong_grid_is_refused(tmp_path):
    writer = StreamingStackWriter(tmp_path / "bad.mha", reference_image(), 1)
    with pytest.raises(ValueError, match="does not match the input image grid"):
        writer.add(np.zeros((9, 9, 9), dtype=np.float32))


# ---------------------------------------------------------------------------
# The parallel write -- one zlib stream assembled from independent blocks
# ---------------------------------------------------------------------------
#
# The write was 65% of invoke on one core of eight, so the
# stack is now deflated in parallel blocks and concatenated. That is only worth
# anything if the file is still exactly the file ITK expects, and the two pieces
# that make it one *zlib* stream rather than raw deflate -- the header's FCHECK
# and the combined Adler-32 -- are hand-rolled. Both are pinned here against
# zlib itself, because a wrong checksum produces a file that is byte-plausible
# and rejected only by the reader.


def test_adler_combine_matches_zlib_over_random_splits():
    """`_adler_combine` must equal a sequential adler32 over the joined bytes."""
    rng = np.random.default_rng(20260815)
    for _ in range(200):
        left = rng.bytes(int(rng.integers(0, 5000)))
        right = rng.bytes(int(rng.integers(0, 5000)))
        assert _adler_combine(
            zlib.adler32(left), zlib.adler32(right), len(right)
        ) == zlib.adler32(left + right)


@pytest.mark.parametrize("level", [1, 5, 6, 9])
def test_zlib_header_passes_the_fcheck_rule(level):
    """(CMF<<8 | FLG) % 31 == 0, or a strict inflater rejects the stream."""
    header = _zlib_header(level)
    assert len(header) == 2
    assert (header[0] << 8 | header[1]) % 31 == 0
    # and it must actually be accepted as the start of a real stream
    body = zlib.compressobj(level, zlib.DEFLATED, -15)
    payload = body.compress(b"doserad" * 1000) + body.flush(zlib.Z_SYNC_FLUSH)
    stream = header + payload + b"\x03\x00" + zlib.adler32(b"doserad" * 1000).to_bytes(4, "big")
    assert zlib.decompress(stream) == b"doserad" * 1000


@pytest.mark.parametrize("workers,blocks", [(1, 1), (1, 32), (4, 32), (8, 7)])
def test_voxels_are_identical_at_every_parallelism(tmp_path, monkeypatch, workers, blocks):
    """Worker and block count are a speed knob, never a correctness one.

    Different block counts give different *bytes* -- each block restarts the
    deflate dictionary -- so this asserts on the decoded image, which is the
    only thing the evaluator ever sees.
    """
    import submission.inference as inference

    monkeypatch.setattr(inference, "ZLIB_WORKERS", workers)
    monkeypatch.setattr(inference, "ZLIB_BLOCKS", blocks)

    rng = np.random.default_rng(7)
    reference = reference_image()
    frames = [np.zeros((4, 5, 6), dtype=np.float32) for _ in range(3)]
    for frame in frames:                      # sparse, like a real prediction
        frame[1:3, 2:4, 1:5] = rng.random((2, 2, 4), dtype=np.float32) * 1.7e-3

    path = tmp_path / f"stack_{workers}_{blocks}.mha"
    writer = inference.StreamingStackWriter(path, reference, len(frames), level=1)
    for frame in frames:
        writer.add(frame)
    writer.close()

    want = sitk.GetArrayFromImage(
        sitk.JoinSeries([sitk.GetImageFromArray(f) for f in frames])
    )
    assert np.array_equal(sitk.GetArrayFromImage(sitk.ReadImage(str(path))), want)


def test_declared_compressed_size_counts_every_byte(tmp_path):
    """CompressedDataSize must cover header, blocks and trailer.

    It is parsed as an integer and used to read the payload, so counting from
    the wrong origin truncates the last frames -- silently, since the header
    still parses and the image still opens.
    """
    reference = reference_image()
    path = tmp_path / "sized.mha"
    writer = StreamingStackWriter(path, reference, 2, level=1)
    for _ in range(2):
        writer.add(np.zeros((4, 5, 6), dtype=np.float32))
    writer.close()

    blob = path.read_bytes()
    header_end = blob.index(b"ElementDataFile = LOCAL\n") + len(b"ElementDataFile = LOCAL\n")
    declared = int(blob[blob.index(b"CompressedDataSize = ") + 21:][:20])
    assert declared == len(blob) - header_end


# ---------------------------------------------------------------------------
# The all-zero block cache
# ---------------------------------------------------------------------------


def _write(path, frames, reference, deflate_attr=None, monkeypatch=None):
    """Write a stack, optionally forcing the uncached deflate path."""
    import submission.inference as inference

    writer = inference.StreamingStackWriter(path, reference, len(frames), level=1)
    if deflate_attr is not None:
        writer._deflate_cached = deflate_attr.__get__(writer)
    for frame in frames:
        writer.add(frame)
    writer.close()
    return path.read_bytes()


def _sparse_frames(rng, n=4, shape=(8, 9, 10)):
    """Frames shaped like a real prediction: one small box, zeros elsewhere."""
    frames = []
    for _ in range(n):
        frame = np.zeros(shape, dtype=np.float32)
        frame[2:4, 3:5, 1:6] = rng.random((2, 2, 5), dtype=np.float32) * 1.7e-3
        frames.append(frame)
    return frames


def test_the_zero_block_cache_writes_byte_identical_output(tmp_path):
    """The whole safety argument, asserted rather than reasoned about.

    Caching the compressed form of an all-zero block is exact only if deflate is
    deterministic for a given level and input. If it ever is not -- a zlib
    version with a different strategy, a changed level -- this fails loudly here
    rather than producing a stack the evaluator decodes differently.
    """
    import submission.inference as inference

    rng = np.random.default_rng(11)
    reference = reference_image((8, 9, 10))
    frames = _sparse_frames(rng)

    writer = inference.StreamingStackWriter(tmp_path / "cached.mha", reference,
                                            len(frames), level=1)
    for frame in frames:
        writer.add(frame)
    writer.close()
    cached = (tmp_path / "cached.mha").read_bytes()
    # Otherwise this test passes just as well when the cache is never consulted,
    # which is exactly how an optimisation gets "verified" while doing nothing.
    assert writer._zero_blocks, "no all-zero block occurred; the test proves nothing"

    plain = _write(tmp_path / "plain.mha", frames, reference,
                   deflate_attr=inference.StreamingStackWriter._deflate)

    assert cached == plain, "the zero-block cache changed the bytes on disk"


def test_the_cache_only_serves_blocks_that_are_actually_zero(tmp_path):
    """Keyed on length, so a non-zero block of a cached length must not hit it.

    The failure this prevents is the worst one available here: a frame carrying
    dose written out as the zeros of some earlier frame, same size, valid
    stream, silently empty.
    """
    reference = reference_image((8, 9, 10))
    rng = np.random.default_rng(3)
    zero = np.zeros((8, 9, 10), dtype=np.float32)
    dosed = np.zeros((8, 9, 10), dtype=np.float32)
    dosed[2:4, 3:5, 1:6] = rng.random((2, 2, 5), dtype=np.float32) * 1.7e-3

    path = tmp_path / "mixed.mha"
    # zeros first, so every block length is in the cache before dose arrives
    _write(path, [zero, dosed], reference)

    got = sitk.GetArrayFromImage(sitk.ReadImage(str(path)))
    assert not got[0].any(), "the all-zero frame did not stay zero"
    np.testing.assert_array_equal(got[1], dosed)


def test_an_all_zero_frame_still_decodes_to_zeros(tmp_path):
    """A prediction that misses the patient is all-zero by contract
    (`models/predictor.py`), so every block hits the cache."""
    reference = reference_image((8, 9, 10))
    frames = [np.zeros((8, 9, 10), dtype=np.float32) for _ in range(3)]
    path = tmp_path / "empty.mha"
    _write(path, frames, reference)
    assert not sitk.GetArrayFromImage(sitk.ReadImage(str(path))).any()
