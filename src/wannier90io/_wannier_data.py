from __future__ import annotations
import abc
import dataclasses
import functools
import pathlib
import typing

import numpy as np

from ._chk import read_chk, read_chk_unformatted
from ._eig import read_eig
from ._mmn import read_mmn
from ._win import parse_win_raw


__all__ = ['WannierData', 'Wannier90Data']


@dataclasses.dataclass(eq=False)
class WannierData(abc.ABC):
    """
    Wannier functions of one calculation: gauge, bands, windows and centres.

    This is the recommended entry point of the package. A subclass per Wannier code
    (`Wannier90Data`) builds it from that code's files and implements the methods that
    read further files on demand (`read_mmn`). Arrays are in the basis of the bands given
    to the Wannier code (num_bands, excluded bands removed); the `*_full` properties give
    the same arrays in the full DFT band basis.

    Attributes:
        folder: folder of the calculation files
        prefix: prefix (seedname) of the calculation files
        real_lattice: (3, 3), row i = lattice vector a_i, Angstrom
        recip_lattice: (3, 3), row i = reciprocal vector b_i, 1/Angstrom
        mp_grid: (3,) k-point grid
        kpoints: (num_kpts, 3) fractional coordinates
        exclude_bands: (num_exclude_bands,) excluded DFT bands, 1-based
        eigenvalues: (num_kpts, num_bands) band energies, eV (None if not available,
            e.g. Wannier90 without disentanglement and without prefix.eig)
        V: (num_kpts, num_bands, num_wann) gauge, Wannier function n at k is
            sum_m V[k, m, n] |psi_mk>; zero on bands outside the outer window
        lwindow: (num_kpts, num_bands) bands inside the outer window (all True without disentanglement)
        lfrozen: (num_kpts, num_bands) bands inside the frozen window (all False without disentanglement)
        dis_win_min, dis_win_max: outer window, eV (None if not set)
        dis_froz_min, dis_froz_max: frozen window, eV (None if not set)
        wannier_centres: (num_wann, 3) Cartesian, Angstrom
        wannier_spreads: (num_wann,) Angstrom^2
        omega_invariant: gauge-invariant spread, Angstrom^2 (None without disentanglement)
    """
    source: typing.ClassVar[str]

    folder: pathlib.Path
    prefix: str
    real_lattice: np.ndarray
    recip_lattice: np.ndarray
    mp_grid: np.ndarray
    kpoints: np.ndarray
    exclude_bands: np.ndarray
    eigenvalues: typing.Optional[np.ndarray]
    V: np.ndarray
    lwindow: np.ndarray
    lfrozen: np.ndarray
    dis_win_min: typing.Optional[float]
    dis_win_max: typing.Optional[float]
    dis_froz_min: typing.Optional[float]
    dis_froz_max: typing.Optional[float]
    wannier_centres: np.ndarray
    wannier_spreads: np.ndarray
    omega_invariant: typing.Optional[float]

    @property
    def num_kpts(self) -> int:
        return self.V.shape[0]

    @property
    def num_bands(self) -> int:
        return self.V.shape[1]

    @property
    def num_wann(self) -> int:
        return self.V.shape[2]

    @functools.cached_property
    def bands_in_full(self) -> np.ndarray:
        """(num_bands,) 0-based index in the full DFT band basis of each band given to the Wannier code."""
        num_bands_full = self.num_bands + len(self.exclude_bands)
        return np.setdiff1d(np.arange(num_bands_full), np.asarray(self.exclude_bands) - 1)

    @functools.cached_property
    def V_full(self) -> np.ndarray:
        """(num_kpts, num_bands_full, num_wann) `V` in the DFT band basis; zero on excluded bands."""
        return self._to_full_bands(self.V, 0)

    @functools.cached_property
    def eigenvalues_full(self) -> typing.Optional[np.ndarray]:
        """(num_kpts, num_bands_full) `eigenvalues` in the DFT band basis; NaN on excluded bands."""
        return None if self.eigenvalues is None else self._to_full_bands(self.eigenvalues, np.nan)

    @functools.cached_property
    def lwindow_full(self) -> np.ndarray:
        """(num_kpts, num_bands_full) `lwindow` in the DFT band basis; False on excluded bands."""
        return self._to_full_bands(self.lwindow, False)

    @functools.cached_property
    def lfrozen_full(self) -> np.ndarray:
        """(num_kpts, num_bands_full) `lfrozen` in the DFT band basis; False on excluded bands."""
        return self._to_full_bands(self.lfrozen, False)

    def _to_full_bands(self, array: np.ndarray, fill_value) -> np.ndarray:
        """Copy `array` (num_kpts, num_bands, ...) into the DFT band basis, `fill_value` on excluded bands."""
        shape = (self.num_kpts, self.num_bands + len(self.exclude_bands)) + array.shape[2:]
        full = np.full(shape, fill_value, dtype=array.dtype)
        full[:, self.bands_in_full] = array
        return full

    def hamiltonian_k(self) -> np.ndarray:
        """
        Hamiltonian in the Wannier gauge on the k-point grid, H(k) = V(k)^dagger diag(eigenvalues(k)) V(k).

        Returns:
            (num_kpts, num_wann, num_wann), eV
        """
        if self.eigenvalues is None:
            raise ValueError(f'{self.prefix}: no band energies were read, so H(k) cannot be built')
        return np.einsum('kmi,km,kmj->kij', self.V.conj(), self.eigenvalues, self.V)

    @abc.abstractmethod
    def read_mmn(self, gauge: str = 'bloch') -> tuple[np.ndarray, np.ndarray]:
        """
        Read the overlaps M_mn(k, b) = <u_mk|u_n,k+b> from the calculation files.

        Arguments:
            gauge: 'bloch' for the overlaps of the Bloch states (num_bands x num_bands),
                'wannier' for V(k)^dagger M(k, b) V(k+b) (num_wann x num_wann)

        Returns:
            overlaps (num_kpts, nntot, n, n) and neighbours (num_kpts, nntot, 5): 0-based k,
            0-based k+b, and the G vector bringing k+b into the first Brillouin zone
        """

    def _overlaps_in_wannier_gauge(self, mmn: np.ndarray, nnkpts: np.ndarray) -> np.ndarray:
        """V(k)^dagger M(k, b) V(k+b), shape (num_kpts, nntot, num_wann, num_wann)."""
        V_kb = self.V[nnkpts[:, :, 1]]  # (num_kpts, nntot, num_bands, num_wann)
        V_k_dagger = self.V.conj().transpose((0, 2, 1))[:, None]  # (num_kpts, 1, num_wann, num_bands)
        # Two batched matmuls; a 3-operand einsum would loop over all four band indices at once
        return V_k_dagger @ mmn @ V_kb


@dataclasses.dataclass(eq=False)
class Wannier90Data(WannierData):
    """`WannierData` of a Wannier90 calculation; build it with `Wannier90Data.from_files`."""
    source: typing.ClassVar[str] = 'wannier90'

    @classmethod
    def from_files(cls, folder: typing.Union[str, pathlib.Path], prefix: str) -> 'Wannier90Data':
        """
        Read prefix.chk (or prefix.chk.fmt), prefix.eig and prefix.win in `folder`.

        V = U_dis U is assembled from the checkpoint: rows 1..ndimwin(k) of u_matrix_opt are
        the bands with lwindow(:, k) = .true., in band order (disentangle.F90 dis_main and
        dis_windows_proj). lfrozen is not in the checkpoint; it is rebuilt from the frozen
        energy window as in disentangle.F90 dis_windows. Frozen states selected by
        projectability (dis_froz_proj) or k-space spheres (dis_spheres_num) are not
        supported and raise NotImplementedError.
        """
        folder = pathlib.Path(folder)

        # Checkpoint: binary if present, else the formatted export of w90chk2chk.x
        if (folder/f'{prefix}.chk').exists():
            chk = read_chk_unformatted(folder/f'{prefix}.chk')
        else:
            with open(folder/f'{prefix}.chk.fmt', 'r') as fh:
                chk = read_chk(fh)
        (Nk, Nb, Nw) = (chk['num_kpts'], chk['num_bands'], chk['num_wann'])

        # Band energies: Wannier90 needs prefix.eig only for disentanglement (and plotting)
        if (folder/f'{prefix}.eig').exists():
            with open(folder/f'{prefix}.eig', 'r') as fh:
                eigenvalues = read_eig(fh)
        elif chk['have_disentangled']:
            raise FileNotFoundError(f'{folder/prefix}.eig is required for a disentangled calculation')
        else:
            eigenvalues = None
        if eigenvalues is not None and eigenvalues.shape != (Nk, Nb):
            raise ValueError(f'{prefix}.eig has (num_kpts, num_bands) = {eigenvalues.shape}, the checkpoint has {(Nk, Nb)}')

        with open(folder/f'{prefix}.win', 'r') as fh:
            parameters = {k.lower(): v for (k, v) in parse_win_raw(fh.read())['parameters'].items()}
        # Logical as Wannier90 reads it (readwrite.F90 w90_readwrite_get_keyword): true if it contains 't'
        dis_froz_proj = parameters.get('dis_froz_proj', False)
        if isinstance(dis_froz_proj, str):
            dis_froz_proj = 't' in dis_froz_proj.lower()
        if dis_froz_proj:
            raise NotImplementedError(f'{prefix}.win sets dis_froz_proj; only the energy frozen window is supported')
        if parameters.get('dis_spheres_num', 0) > 0:
            raise NotImplementedError(f'{prefix}.win sets dis_spheres_num; only the energy frozen window is supported')
        windows = {key: (float(parameters[key]) if key in parameters else None)
                   for key in ['dis_win_min', 'dis_win_max', 'dis_froz_min', 'dis_froz_max']}
        if windows['dis_froz_min'] is not None and windows['dis_froz_max'] is None:
            raise ValueError(f'{prefix}.win sets dis_froz_min but not dis_froz_max')

        # Gauge V = U_dis U in the num_bands basis
        V = np.zeros((Nk, Nb, Nw), dtype=complex)
        if chk['have_disentangled']:  # u_matrix_opt rows 1..ndimwin are the lwindow bands
            lwindow = chk['lwindow']
            if not np.array_equal(lwindow.sum(axis=1), chk['ndimwin']):
                raise ValueError(f'{prefix}.chk: ndimwin differs from the number of bands in lwindow')
            for ik in range(Nk):
                ndimwin = chk['ndimwin'][ik]
                if np.any(chk['u_matrix_opt'][ik, ndimwin:] != 0):
                    raise ValueError(f'{prefix}.chk: u_matrix_opt is nonzero beyond row ndimwin at k-point {ik + 1}')
                V[ik, lwindow[ik]] = chk['u_matrix_opt'][ik, :ndimwin] @ chk['u_matrix'][ik]
        else:  # isolated bands: num_bands = num_wann, V = U
            if Nb != Nw:
                raise ValueError(f'{prefix}.chk: num_bands = {Nb} differs from num_wann = {Nw} without disentanglement')
            lwindow = np.ones((Nk, Nb), dtype=bool)
            V[:] = chk['u_matrix']

        # The outer window in the checkpoint must be the one of the .win and .eig
        win_min = -np.inf if windows['dis_win_min'] is None else windows['dis_win_min']
        win_max = np.inf if windows['dis_win_max'] is None else windows['dis_win_max']
        if chk['have_disentangled'] and not np.array_equal(lwindow, (eigenvalues >= win_min) & (eigenvalues <= win_max)):
            raise ValueError(f'{prefix}.chk: lwindow differs from the outer window of {prefix}.win and {prefix}.eig')

        # Frozen window as in dis_windows: inside the outer window and [dis_froz_min, dis_froz_max];
        # on only with disentanglement and dis_froz_max set
        if chk['have_disentangled'] and windows['dis_froz_max'] is not None:
            froz_min = -np.inf if windows['dis_froz_min'] is None else windows['dis_froz_min']
            lfrozen = lwindow & (eigenvalues >= froz_min) & (eigenvalues <= windows['dis_froz_max'])
        else:
            lfrozen = np.zeros((Nk, Nb), dtype=bool)

        return cls(
            folder=folder,
            prefix=prefix,
            real_lattice=chk['real_lattice'],
            recip_lattice=chk['recip_lattice'],
            mp_grid=chk['mp_grid'],
            kpoints=chk['kpt_latt'],
            exclude_bands=chk['exclude_bands'],
            eigenvalues=eigenvalues,
            V=V,
            lwindow=lwindow,
            lfrozen=lfrozen,
            dis_win_min=windows['dis_win_min'],
            dis_win_max=windows['dis_win_max'],
            dis_froz_min=windows['dis_froz_min'],
            dis_froz_max=windows['dis_froz_max'],
            wannier_centres=chk['wannier_centres'],
            wannier_spreads=chk['wannier_spreads'],
            omega_invariant=chk.get('omega_invariant'),
        )

    def read_mmn(self, gauge: str = 'bloch') -> tuple[np.ndarray, np.ndarray]:
        """Read prefix.mmn on each call (not cached); see `WannierData.read_mmn`."""
        if gauge not in ('bloch', 'wannier'):
            raise ValueError(f"gauge must be 'bloch' or 'wannier', got {gauge!r}")
        with open(self.folder/f'{self.prefix}.mmn', 'r') as fh:
            (mmn, nnkpts) = read_mmn(fh)
        if mmn.shape[0] != self.num_kpts or mmn.shape[2] != self.num_bands:
            raise ValueError(f'{self.prefix}.mmn has (num_kpts, num_bands) = {(mmn.shape[0], mmn.shape[2])}, '
                             f'expected {(self.num_kpts, self.num_bands)}')
        if gauge == 'bloch':
            return (mmn, nnkpts)
        else:  # rotate both band indices to the Wannier gauge
            return (self._overlaps_in_wannier_gauge(mmn, nnkpts), nnkpts)
