from __future__ import annotations
import os
import typing

import numpy as np
import scipy.io


__all__ = ['read_chk', 'read_chk_unformatted']


def read_chk(stream: typing.TextIO) -> dict:
    """
    Read a formatted checkpoint file (seedname.chk.fmt, written by `w90chk2chk.x -export`)

    Arguments:
        stream: a file-like stream

    Returns:
        dict, with the same keys and array layouts as `read_chk_unformatted`

    """
    def read_rows(nrows: int, dtype=float) -> np.ndarray:
        # One value (or one complex pair, or one vector) per line, flattened in file order
        return np.loadtxt(stream, max_rows=nrows, dtype=dtype, ndmin=2).ravel()

    records = {}
    records['header'] = stream.readline().strip()
    records['num_bands'] = Nb = int(stream.readline())
    records['num_exclude_bands'] = int(stream.readline())
    records['exclude_bands'] = np.array(
        [int(stream.readline()) for _ in range(records['num_exclude_bands'])], dtype=int
    )
    records['real_lattice'] = np.fromstring(stream.readline(), sep=' ', dtype=float)
    records['recip_lattice'] = np.fromstring(stream.readline(), sep=' ', dtype=float)
    records['num_kpts'] = Nk = int(stream.readline())
    records['mp_grid'] = np.fromstring(stream.readline(), sep=' ', dtype=int)
    records['kpt_latt'] = read_rows(Nk)
    records['nntot'] = Nn = int(stream.readline())
    records['num_wann'] = Nw = int(stream.readline())
    records['checkpoint'] = stream.readline().strip()
    records['have_disentangled'] = bool(int(stream.readline()))
    if records['have_disentangled']:
        records['omega_invariant'] = float(stream.readline())
        records['lwindow'] = read_rows(Nk*Nb, dtype=int) != 0
        records['ndimwin'] = read_rows(Nk, dtype=int)
        records['u_matrix_opt'] = read_rows(Nk*Nw*Nb).view(complex)
    records['u_matrix'] = read_rows(Nk*Nw*Nw).view(complex)
    records['m_matrix'] = read_rows(Nk*Nn*Nw*Nw).view(complex)
    records['wannier_centres'] = read_rows(Nw)
    records['wannier_spreads'] = read_rows(Nw)

    return _chk_from_records(records)


def read_chk_unformatted(path: typing.Union[str, os.PathLike]) -> dict:
    """
    Read a binary checkpoint file (seedname.chk) written by wannier90.x

    The file must use the default gfortran/ifort unformatted sequential layout:
    4-byte little-endian record markers, 4-byte integers and logicals, 8-byte reals.
    Records longer than 2 GiB, which gfortran splits into subrecords, are supported.
    Any other layout (big-endian, 8-byte markers, `-i8` builds) raises `ValueError`.

    Reference writer: `w90_wannier90_readwrite_write_chkpt` in wannier90 `src/wannier90_readwrite.F90`
    (`param_write_chkpt` in `src/parameters.F90` for v2.0.1 to v3.1.0; the record layout is the same).

    Arguments:
        path: path to the seedname.chk file

    Returns:
        dict with keys
            header, checkpoint (str); num_bands, num_exclude_bands, num_kpts, nntot, num_wann (int);
            exclude_bands (num_exclude_bands,); real_lattice, recip_lattice (3, 3), row i = vector i;
            mp_grid (3,); kpt_latt (num_kpts, 3); have_disentangled (bool);
            only if have_disentangled: omega_invariant (float), lwindow (num_kpts, num_bands),
            ndimwin (num_kpts,), u_matrix_opt (num_kpts, num_bands, num_wann);
            u_matrix (num_kpts, num_wann, num_wann); m_matrix (num_kpts, nntot, num_wann, num_wann);
            wannier_centres (num_wann, 3); wannier_spreads (num_wann,)

    """
    with open(path, 'rb') as fh:
        # The first record is the header (`character(len=33) :: header` in the writer):
        # 4-byte leading marker, 33-byte payload, 4-byte trailing marker, both markers = 33.
        # With 8-byte markers the low half of the leading marker is also 33, so check both.
        # Bytes [0, 4): leading marker, [4, 37): header, [37, 41): trailing marker.
        first_record = fh.read(4 + 33 + 4)
        fh.seek(0)
        header_markers = (first_record[:4], first_record[37:])
        if any(len(m) != 4 or int.from_bytes(m, 'little', signed=True) != 33 for m in header_markers):
            raise ValueError(
                f'{path}: the first record is not the 33-character header with a 4-byte little-endian '
                'record marker; the file is not a standard (gfortran/ifort default) unformatted '
                'wannier90 checkpoint (big-endian, 8-byte record markers or not a .chk file?)'
            )
        fortran_file = scipy.io.FortranFile(fh, 'r', header_dtype='<u4')

        def read(name: str, dtype: str, count: int) -> np.ndarray:
            return _read_record(fh, fortran_file, f'{path}: record {name!r}', np.dtype(dtype), count)

        records = {}
        records['header'] = read('header', 'S33', 1)[0].decode('ascii').strip()
        records['num_bands'] = Nb = int(read('num_bands', '<i4', 1)[0])
        records['num_exclude_bands'] = int(read('num_exclude_bands', '<i4', 1)[0])
        records['exclude_bands'] = read('exclude_bands', '<i4', records['num_exclude_bands']).astype(int)
        records['real_lattice'] = read('real_lattice', '<f8', 9)
        records['recip_lattice'] = read('recip_lattice', '<f8', 9)
        records['num_kpts'] = Nk = int(read('num_kpts', '<i4', 1)[0])
        records['mp_grid'] = read('mp_grid', '<i4', 3).astype(int)
        records['kpt_latt'] = read('kpt_latt', '<f8', 3*Nk)
        records['nntot'] = Nn = int(read('nntot', '<i4', 1)[0])
        records['num_wann'] = Nw = int(read('num_wann', '<i4', 1)[0])
        records['checkpoint'] = read('checkpoint', 'S20', 1)[0].decode('ascii').strip()
        # Fortran logical: gfortran writes .true. as 1, ifort as -1
        records['have_disentangled'] = bool(read('have_disentangled', '<i4', 1)[0] != 0)
        if records['have_disentangled']:
            records['omega_invariant'] = float(read('omega_invariant', '<f8', 1)[0])
            records['lwindow'] = read('lwindow', '<i4', Nb*Nk) != 0
            records['ndimwin'] = read('ndimwin', '<i4', Nk).astype(int)
            records['u_matrix_opt'] = read('u_matrix_opt', '<c16', Nb*Nw*Nk)
        records['u_matrix'] = read('u_matrix', '<c16', Nw*Nw*Nk)
        records['m_matrix'] = read('m_matrix', '<c16', Nw*Nw*Nn*Nk)
        records['wannier_centres'] = read('wannier_centres', '<f8', 3*Nw)
        records['wannier_spreads'] = read('wannier_spreads', '<f8', Nw)

        if fh.read(1):
            raise ValueError(f'{path}: unexpected data after the last record (wannier_spreads)')

    return _chk_from_records(records)


def _chk_from_records(records: dict) -> dict:
    """
    Reshape the flat records of a checkpoint file into the arrays returned by `read_chk`
    and `read_chk_unformatted`.

    Both readers call this, so the two return identical layouts. Each array record is flat in
    Fortran storage order, e.g. u_matrix(i, j, k) with i fastest; the k-point index is moved
    to the first axis.
    """
    Nb = records['num_bands']
    Nk = records['num_kpts']
    Nn = records['nntot']
    Nw = records['num_wann']

    chk = dict(records)
    # real_lattice(i, j) is component j of vector i, so row i of the result is vector i
    chk['real_lattice'] = records['real_lattice'].reshape((3, 3), order='F')
    chk['recip_lattice'] = records['recip_lattice'].reshape((3, 3), order='F')
    chk['kpt_latt'] = records['kpt_latt'].reshape((3, Nk), order='F').T
    if records['have_disentangled']:
        chk['lwindow'] = records['lwindow'].reshape((Nb, Nk), order='F').T
        chk['u_matrix_opt'] = records['u_matrix_opt'].reshape((Nb, Nw, Nk), order='F').transpose((2, 0, 1))
    chk['u_matrix'] = records['u_matrix'].reshape((Nw, Nw, Nk), order='F').transpose((2, 0, 1))
    chk['m_matrix'] = records['m_matrix'].reshape((Nw, Nw, Nn, Nk), order='F').transpose((3, 2, 0, 1))
    chk['wannier_centres'] = records['wannier_centres'].reshape((3, Nw), order='F').T

    return chk


def _read_record(
    fh: typing.BinaryIO, fortran_file: scipy.io.FortranFile, name: str, dtype: np.dtype, count: int
) -> np.ndarray:
    """
    Read one unformatted sequential record of `count` elements of `dtype`.

    Called by `read_chk_unformatted` for every record of the checkpoint. A record with a
    positive leading marker is read by `scipy.io.FortranFile`; a negative leading marker
    starts a gfortran subrecord chain (records over 2 GiB, e.g. a large m_matrix), which
    scipy does not support, so `_read_subrecords` reads it.
    """
    # Peek the leading marker without consuming it
    pos = fh.tell()
    marker = fh.read(4)
    fh.seek(pos)
    if len(marker) < 4:
        raise ValueError(f'{name}: unexpected end of file')

    if int.from_bytes(marker, 'little', signed=True) >= 0:  # a single record
        try:
            data = fortran_file.read_record(dtype)
        except (scipy.io.FortranEOFError, scipy.io.FortranFormattingError, ValueError) as err:
            raise ValueError(f'{name}: cannot read record ({err})') from err
    else:  # a chain of gfortran subrecords
        data = _read_subrecords(fh, name, dtype)

    if data.size != count:
        raise ValueError(
            f'{name}: expected {count} elements of {dtype} ({count*dtype.itemsize} bytes), '
            f'got a record of {data.size*dtype.itemsize} bytes'
        )
    return data


def _read_subrecords(fh: typing.BinaryIO, name: str, dtype: np.dtype) -> np.ndarray:
    """
    Read a record that gfortran split into subrecords, starting at the current position.

    Called by `_read_record` when the leading marker is negative. gfortran writes each
    subrecord as |head| bytes framed by two 4-byte markers: head < 0 if another subrecord
    follows, tail < 0 if a subrecord precedes. `scipy.io.FortranFile` reads only single
    records, so this chain is parsed here.
    """
    def read_marker() -> int:
        b = fh.read(4)
        if len(b) < 4:
            raise ValueError(f'{name}: end of file in the middle of a subrecord chain')
        return int.from_bytes(b, 'little', signed=True)

    # First pass: validate the markers and collect (offset, length) of each subrecord
    chunks = []
    while True:
        head = read_marker()
        length = abs(head)
        offset = fh.tell()
        fh.seek(length, os.SEEK_CUR)
        tail = read_marker()
        if tail != (length if not chunks else -length):
            raise ValueError(f'{name}: inconsistent subrecord markers ({head}, {tail})')
        chunks.append((offset, length))
        if head > 0:  # last subrecord
            break
    end = fh.tell()

    total = sum(length for _, length in chunks)
    if total % dtype.itemsize != 0:
        raise ValueError(f'{name}: record of {total} bytes is not a multiple of {dtype} ({dtype.itemsize} bytes)')

    # Second pass: read the payloads straight into the output buffer
    data = np.empty(total // dtype.itemsize, dtype=dtype)
    buffer = memoryview(data.view(np.uint8))
    filled = 0
    for offset, length in chunks:
        fh.seek(offset)
        if fh.readinto(buffer[filled:filled + length]) != length:
            raise ValueError(f'{name}: end of file in the middle of a subrecord')
        filled += length
    fh.seek(end)
    return data
