import itertools
import pathlib
import shutil
import subprocess

import numpy as np
import pytest

import wannier90io as w90io


@pytest.mark.parametrize('example', ['example01', 'example02'])
def test_read_chk(wannier90, example):
    with open(pathlib.Path(wannier90)/f'examples/{example}/wannier.chk.fmt', 'r') as fh:
        chk = w90io.read_chk(fh)
        umn = chk['u_matrix']
        mmn_ref = chk['m_matrix']

    with open(pathlib.Path(wannier90)/f'examples/{example}/wannier.nnkp', 'r') as fh:
        nnkp = w90io.parse_nnkp_raw(fh.read())

    with open(pathlib.Path(wannier90)/f'examples/{example}/wannier.mmn', 'r') as fh:
        (mmn_test, nnkpts) = w90io.read_mmn(fh)
        nnkpts1 = np.copy(nnkpts)
        nnkpts1[:, :, 0] += 1
        nnkpts1[:, :, 1] += 1

    for (ik, ikb) in itertools.product(range(mmn_test.shape[0]), range(mmn_test.shape[1])):
        mmn_test[ik, ikb] = np.dot(np.dot(umn[ik].conj().T, mmn_test[ik, ikb]), umn[nnkpts[:, :, 1][ik, ikb]])

    assert np.allclose(np.array(nnkp['nnkpts']).reshape(nnkpts.shape), nnkpts1)
    assert np.allclose(mmn_test, mmn_ref)
    assert np.allclose(chk['real_lattice'][0], nnkp['direct_lattice']['a1'])
    assert np.allclose(chk['real_lattice'][1], nnkp['direct_lattice']['a2'])
    assert np.allclose(chk['real_lattice'][2], nnkp['direct_lattice']['a3'])
    assert np.allclose(chk['recip_lattice'][0], nnkp['reciprocal_lattice']['b1'])
    assert np.allclose(chk['recip_lattice'][1], nnkp['reciprocal_lattice']['b2'])
    assert np.allclose(chk['recip_lattice'][2], nnkp['reciprocal_lattice']['b3'])


def _assert_chk_equal(chk, chk_ref):
    assert chk.keys() == chk_ref.keys()
    for key in chk_ref:
        assert np.array_equal(chk[key], chk_ref[key]), key


# example01, 02: isolated bands; example03, 04: disentanglement
@pytest.mark.parametrize('example', ['example01', 'example02', 'example03', 'example04'])
def test_read_chk_unformatted(wannier90, example):
    path = pathlib.Path(wannier90)/f'examples/{example}'
    chk = w90io.read_chk_unformatted(path/'wannier.chk')
    with open(path/'wannier.chk.fmt', 'r') as fh:
        chk_fmt = w90io.read_chk(fh)

    # w90chk2chk.x writes reals with 17 significant digits, so the formatted values are exact
    _assert_chk_equal(chk, chk_fmt)

    (Nk, Nb, Nw, Nn) = (chk['num_kpts'], chk['num_bands'], chk['num_wann'], chk['nntot'])
    assert chk['kpt_latt'].shape == (Nk, 3)
    assert chk['u_matrix'].shape == (Nk, Nw, Nw)
    assert chk['m_matrix'].shape == (Nk, Nn, Nw, Nw)
    assert chk['wannier_centres'].shape == (Nw, 3)
    assert chk['wannier_spreads'].shape == (Nw,)
    assert chk['have_disentangled'] == (example in ['example03', 'example04'])
    for ik in range(Nk):
        assert np.allclose(chk['u_matrix'][ik].conj().T @ chk['u_matrix'][ik], np.eye(Nw))

    if chk['have_disentangled']:
        assert chk['lwindow'].shape == (Nk, Nb)
        assert np.array_equal(chk['lwindow'].sum(axis=1), chk['ndimwin'])
        # u_matrix_opt[k, :ndimwin, :] has orthonormal columns; a wrong axis order would break this
        assert chk['u_matrix_opt'].shape == (Nk, Nb, Nw)
        for ik in range(Nk):
            u_opt = chk['u_matrix_opt'][ik, :chk['ndimwin'][ik], :]
            assert np.allclose(u_opt.conj().T @ u_opt, np.eye(Nw))


def _split_records(raw):
    """Payloads of a file of single records with 4-byte little-endian markers."""
    payloads = []
    pos = 0
    while pos < len(raw):
        n = int.from_bytes(raw[pos:pos+4], 'little')
        payloads.append(raw[pos+4:pos+4+n])
        assert raw[pos+4+n:pos+8+n] == raw[pos:pos+4]
        pos += n + 8
    return payloads


def _join_records(payloads, marker_size=4, byteorder='little', max_subrecord_length=None):
    """
    Write payloads as unformatted sequential records; with `max_subrecord_length`, split them
    into subrecords as gfortran -fmax-subrecord-length does (checked in test_join_records_matches_gfortran).
    """
    def marker(n):
        return int(n).to_bytes(marker_size, byteorder, signed=True)

    out = bytearray()
    for payload in payloads:
        if max_subrecord_length is None or len(payload) <= max_subrecord_length:
            out += marker(len(payload)) + payload + marker(len(payload))
            continue
        chunks = [payload[i:i+max_subrecord_length] for i in range(0, len(payload), max_subrecord_length)]
        for (i, chunk) in enumerate(chunks):
            head = len(chunk) if i == len(chunks) - 1 else -len(chunk)
            tail = len(chunk) if i == 0 else -len(chunk)
            out += marker(head) + chunk + marker(tail)
    return bytes(out)


@pytest.mark.skipif(shutil.which('gfortran') is None, reason='gfortran not available')
def test_join_records_matches_gfortran(tmp_path):
    source = tmp_path/'write_records.f90'
    source.write_text(
        'program write_records\n'
        '  integer :: i\n'
        '  open (10, file="records.bin", form="unformatted")\n'
        '  write (10) (i, i=1, 10)\n'
        '  write (10) 7\n'
        '  write (10) (cmplx(i, -i, kind=8), i=1, 5)\n'
        '  close (10)\n'
        'end program\n'
    )
    raw = {}
    for (label, flags) in [('single', []), ('split', ['-fmax-subrecord-length=16'])]:
        subprocess.run(['gfortran', *flags, str(source), '-o', str(tmp_path/label)], check=True)
        subprocess.run([str(tmp_path/label)], cwd=tmp_path, check=True)
        raw[label] = (tmp_path/'records.bin').read_bytes()

    assert _join_records(_split_records(raw['single']), max_subrecord_length=16) == raw['split']


@pytest.fixture
def chk_raw(wannier90):
    """Bytes of a disentangled checkpoint (example03), so every record type is present."""
    return (pathlib.Path(wannier90)/'examples/example03/wannier.chk').read_bytes()


@pytest.mark.parametrize('max_subrecord_length', [64, 1000])
def test_read_chk_unformatted_subrecords(wannier90, chk_raw, tmp_path, max_subrecord_length):
    path = tmp_path/'split.chk'
    path.write_bytes(_join_records(_split_records(chk_raw), max_subrecord_length=max_subrecord_length))
    assert len(path.read_bytes()) > len(chk_raw)

    chk_ref = w90io.read_chk_unformatted(pathlib.Path(wannier90)/'examples/example03/wannier.chk')
    _assert_chk_equal(w90io.read_chk_unformatted(path), chk_ref)


def _write_modified(tmp_path, payloads, **kwargs):
    path = tmp_path/'modified.chk'
    path.write_bytes(_join_records(payloads, **kwargs))
    return path


def test_read_chk_unformatted_errors(chk_raw, tmp_path):
    payloads = _split_records(chk_raw)

    # Truncated in the last record
    path = tmp_path/'truncated.chk'
    path.write_bytes(chk_raw[:-10])
    with pytest.raises(ValueError, match="'wannier_spreads'"):
        w90io.read_chk_unformatted(path)

    # Missing last record
    with pytest.raises(ValueError, match="'wannier_spreads': unexpected end of file"):
        w90io.read_chk_unformatted(_write_modified(tmp_path, payloads[:-1]))

    # Extra data after the last record
    with pytest.raises(ValueError, match='after the last record'):
        w90io.read_chk_unformatted(_write_modified(tmp_path, payloads + [b'\0'*8]))

    # Big-endian record markers
    with pytest.raises(ValueError, match='4-byte little-endian'):
        w90io.read_chk_unformatted(_write_modified(tmp_path, payloads, byteorder='big'))

    # 8-byte record markers
    with pytest.raises(ValueError, match='4-byte little-endian'):
        w90io.read_chk_unformatted(_write_modified(tmp_path, payloads, marker_size=8))

    # 8-byte integers (-i8 build)
    num_bands_i8 = np.frombuffer(payloads[1], dtype='<i4').astype('<i8').tobytes()
    with pytest.raises(ValueError, match=r"'num_bands': expected 1 elements of int32 \(4 bytes\), got a record of 8"):
        w90io.read_chk_unformatted(_write_modified(tmp_path, [payloads[0], num_bands_i8] + payloads[2:]))

    # m_matrix with one element too few (record 18 of a disentangled checkpoint)
    m_matrix_short = payloads[18][:-16]
    with pytest.raises(ValueError, match="'m_matrix': expected"):
        w90io.read_chk_unformatted(_write_modified(tmp_path, payloads[:18] + [m_matrix_short] + payloads[19:]))

    # Inconsistent markers inside a subrecord chain: the second tail marker must be negative
    raw = bytearray(_join_records(payloads, max_subrecord_length=1000))
    offset = len(_join_records(payloads[:18], max_subrecord_length=1000))  # start of m_matrix
    head = int.from_bytes(raw[offset:offset+4], 'little', signed=True)
    assert head == -1000
    second_tail = offset + 1008 + 4 + 1000
    raw[second_tail:second_tail+4] = (1000).to_bytes(4, 'little', signed=True)
    path = tmp_path/'bad_subrecord.chk'
    path.write_bytes(bytes(raw))
    with pytest.raises(ValueError, match="'m_matrix': inconsistent subrecord markers"):
        w90io.read_chk_unformatted(path)
