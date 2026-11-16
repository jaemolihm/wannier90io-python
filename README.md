# Wannier90 I/O with Python

A Python library for reading and writing [Wannier90][w90] files.

## Quickstart

`Wannier90Data` collects the data of a Wannier90 calculation (gauge, bands, windows,
centres) from `prefix.chk`, `prefix.eig` and `prefix.win`, and reads further files on demand.
It is the recommended entry point.

<!--
```python
folder = "./tests/fixtures/wannier90-3.1.0/examples/example03"
```
-->

<!--pytest-codeblocks:cont-->
```python
import wannier90io as w90io


wd = w90io.Wannier90Data.from_files(folder, 'wannier')

V = wd.V                                # (num_kpts, num_bands, num_wann) gauge U_dis U
lwindow, lfrozen = wd.lwindow, wd.lfrozen
hk = wd.hamiltonian_k()                 # (num_kpts, num_wann, num_wann), eV
mmn, nnkpts = wd.read_mmn(gauge='wannier')
```

## File parsers

The parsers of the individual files are also available.

<!--
```python
win_fpath = "./tests/fixtures/wannier90-3.1.0/examples/example01/wannier.win"
nnkp_fpath = "./tests/fixtures/wannier90-3.1.0/examples/example01/wannier.nnkp"
```
-->

<!--pytest-codeblocks:cont-->
```python
import pprint

import wannier90io as w90io


pp = pprint.PrettyPrinter()

with open(win_fpath, 'r') as fh:
    parsed_win = w90io.parse_win_raw(fh.read())

with open(nnkp_fpath, 'r') as fh:
    parsed_nnkp = w90io.parse_nnkp_raw(fh.read())

pp.pprint(parsed_win)
pp.pprint(parsed_nnkp)
```

[w90]: http://wannier.org
