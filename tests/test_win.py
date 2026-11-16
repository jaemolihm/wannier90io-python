import pathlib

import pytest

import wannier90io as w90io


@pytest.mark.parametrize('example', [f'example{i:02d}' for i in [1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 13, 17, 18, 19, 20]])
def test_parse_win(wannier90, example):
    with open(pathlib.Path(wannier90)/f'examples/{example}/wannier.win', 'r') as fh:
        contents = fh.read()

    try:
        parsed_win = w90io.parse_win_raw(contents)
    except Exception:
        assert False

    try:
        w90io._schema.WIN.model_validate(parsed_win)
    except Exception:
        assert False


@pytest.mark.parametrize('value, expected', [
    # range vectors, Wannier90 syntax (w90_readwrite_get_range_vector)
    ('1-5, 10-12', [1, 2, 3, 4, 5, 10, 11, 12]),
    ('1-3,7', [1, 2, 3, 7]),
    ('1-3 7', [1, 2, 3, 7]),
    ('1 - 3; 7:8', [1, 2, 3, 7, 8]),
    ('1-3', [1, 2, 3]),
    ('3-1', [1, 2, 3]),
    # logicals
    ('T', True), ('.true.', True), ('false', False),
    # strings starting with t / f are not logicals
    ('three_dim', 'three_dim'), ('fermi', 'fermi'),
    # numbers and vectors are unchanged
    ('3', 3), ('6.4d0', 6.4), ('1e-5', 1e-5), ('4 4 4', [4, 4, 4]),
])
def test_convert(value, expected):
    assert w90io._core.convert(value) == expected


def test_parse_win_exclude_bands():
    parsed_win = w90io.parse_win_raw('num_wann = 4\nexclude_bands = 1-5, 10-12\n')
    assert parsed_win['parameters']['exclude_bands'] == [1, 2, 3, 4, 5, 10, 11, 12]
