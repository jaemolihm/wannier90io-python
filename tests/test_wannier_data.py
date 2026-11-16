import dataclasses
import pathlib
import shutil

import numpy as np
import pytest

import wannier90io as w90io


# example01, 02: isolated bands (example01 has no .eig); example03, 04: disentanglement with a frozen window;
# example03/win_min: outer window above band 1 at some k-points, so u_matrix_opt rows are not bands 1..ndimwin
@pytest.mark.parametrize('example', ['example01', 'example02', 'example03', 'example04', 'example03/win_min'])
def test_wannier90_data(wannier90, example):
    folder = pathlib.Path(wannier90)/f'examples/{example}'
    wd = w90io.Wannier90Data.from_files(folder, 'wannier')
    chk = w90io.read_chk_unformatted(folder/'wannier.chk')
    (Nk, Nb, Nw) = (chk['num_kpts'], chk['num_bands'], chk['num_wann'])

    assert wd.source == 'wannier90'
    assert (wd.num_kpts, wd.num_bands, wd.num_wann) == (Nk, Nb, Nw)
    assert np.array_equal(wd.kpoints, chk['kpt_latt'])
    assert np.array_equal(wd.wannier_centres, chk['wannier_centres'])
    assert (wd.eigenvalues is None) == (example == 'example01')
    for ik in range(Nk):
        assert np.allclose(wd.V[ik].conj().T @ wd.V[ik], np.eye(Nw))
    assert np.all(wd.V[~wd.lwindow] == 0)

    if chk['have_disentangled']:
        assert wd.omega_invariant == chk['omega_invariant']
        assert np.array_equal(wd.lwindow, chk['lwindow'])
        if example == 'example03/win_min':
            assert not wd.lwindow[:, 0].all() and wd.lwindow[:, 0].any()
        # Frozen window of the .win (dis_froz_max only), and every frozen state lies in span V(k)
        assert np.array_equal(wd.lfrozen, wd.lwindow & (wd.eigenvalues <= wd.dis_froz_max))
        assert wd.lfrozen.sum(axis=1).min() > 0
        assert np.allclose(np.sum(np.abs(wd.V) ** 2, axis=2)[wd.lfrozen], 1)
    else:
        assert np.array_equal(wd.V, chk['u_matrix'])
        assert wd.lwindow.all() and not wd.lfrozen.any()
        assert wd.omega_invariant is None

    if wd.eigenvalues is None:
        with pytest.raises(ValueError, match='no band energies'):
            wd.hamiltonian_k()
    elif not chk['have_disentangled']:
        # Isolated bands: H(k) is a unitary rotation of diag(eigenvalues)
        assert np.allclose(np.linalg.eigvalsh(wd.hamiltonian_k()), wd.eigenvalues)


@pytest.mark.parametrize('example', ['example01', 'example02', 'example03', 'example04', 'example03/win_min'])
def test_wannier90_data_read_mmn(wannier90, example):
    folder = pathlib.Path(wannier90)/f'examples/{example}'
    wd = w90io.Wannier90Data.from_files(folder, 'wannier')
    chk = w90io.read_chk_unformatted(folder/'wannier.chk')

    (mmn, nnkpts) = wd.read_mmn()
    with open(folder/'wannier.mmn', 'r') as fh:
        (mmn_ref, nnkpts_ref) = w90io.read_mmn(fh)
    assert np.array_equal(mmn, mmn_ref) and np.array_equal(nnkpts, nnkpts_ref)

    # The checkpoint stores V^dagger M V in the neighbour order of .nnkp, which may differ from .mmn
    (mmn_wannier, nnkpts) = wd.read_mmn(gauge='wannier')
    with open(folder/'wannier.nnkp', 'r') as fh:
        nnkp = np.array(w90io.parse_nnkp_raw(fh.read())['nnkpts']).reshape(nnkpts.shape)
    for (ik, ib) in np.ndindex(nnkpts.shape[:2]):
        ib_chk = [j for j in range(nnkp.shape[1])
                  if nnkp[ik, j, 1] - 1 == nnkpts[ik, ib, 1] and np.array_equal(nnkp[ik, j, 2:], nnkpts[ik, ib, 2:])]
        assert len(ib_chk) == 1
        assert np.allclose(mmn_wannier[ik, ib], chk['m_matrix'][ik, ib_chk[0]])

    with pytest.raises(ValueError, match='gauge'):
        wd.read_mmn(gauge='wrong')


def test_wannier90_data_full_bands(wannier90):
    wd = w90io.Wannier90Data.from_files(pathlib.Path(wannier90)/'examples/example03', 'wannier')
    # Pretend DFT bands 1 and 5 were excluded: the 12 Wannier90 bands are DFT bands 2-4, 6-14
    fields = {field.name: getattr(wd, field.name) for field in dataclasses.fields(wd)}
    wd_excl = w90io.Wannier90Data(**{**fields, 'exclude_bands': np.array([1, 5])})
    bands = np.array([1, 2, 3] + list(range(5, 14)))

    assert np.array_equal(wd_excl.bands_in_full, bands)
    assert wd_excl.V_full.shape == (wd.num_kpts, 14, wd.num_wann)
    assert np.array_equal(wd_excl.V_full[:, bands], wd.V) and np.all(wd_excl.V_full[:, [0, 4]] == 0)
    assert np.array_equal(wd_excl.eigenvalues_full[:, bands], wd.eigenvalues)
    assert np.all(np.isnan(wd_excl.eigenvalues_full[:, [0, 4]]))
    assert np.array_equal(wd_excl.lwindow_full[:, bands], wd.lwindow) and not wd_excl.lwindow_full[:, [0, 4]].any()
    assert np.array_equal(wd_excl.lfrozen_full[:, bands], wd.lfrozen) and not wd_excl.lfrozen_full[:, [0, 4]].any()
    # Without excluded bands the full basis is the Wannier90 basis
    assert np.array_equal(wd.V_full, wd.V)


@pytest.fixture
def example03_copy(wannier90, tmp_path):
    """Copy of example03 (disentangled) without .mmn, for tests that edit its files."""
    for suffix in ['chk', 'eig', 'win']:
        shutil.copy(pathlib.Path(wannier90)/f'examples/example03/wannier.{suffix}', tmp_path)
    return tmp_path


@pytest.mark.parametrize('line, error, match', [
    ('dis_froz_degen = true', NotImplementedError, 'dis_froz_degen'),
    ('dis_froz_proj = .true.', NotImplementedError, 'dis_froz_proj'),
    ('dis_spheres_num = 1', NotImplementedError, 'dis_spheres_num'),
    ('dis_win_min = 0.0', ValueError, 'lwindow differs'),
])
def test_wannier90_data_win_errors(example03_copy, line, error, match):
    with open(example03_copy/'wannier.win', 'a') as fh:
        fh.write(f'\n{line}\n')
    with pytest.raises(error, match=match):
        w90io.Wannier90Data.from_files(example03_copy, 'wannier')


def test_wannier90_data_froz_min_without_max(example03_copy):
    text = (example03_copy/'wannier.win').read_text().replace('dis_froz_max', 'dis_froz_min')
    (example03_copy/'wannier.win').write_text(text)
    with pytest.raises(ValueError, match='dis_froz_min but not dis_froz_max'):
        w90io.Wannier90Data.from_files(example03_copy, 'wannier')


def test_wannier90_data_missing_eig(example03_copy):
    (example03_copy/'wannier.eig').unlink()
    with pytest.raises(FileNotFoundError, match='required for a disentangled'):
        w90io.Wannier90Data.from_files(example03_copy, 'wannier')
