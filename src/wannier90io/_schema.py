from __future__ import annotations
import typing

import pydantic


class UnitCell(pydantic.BaseModel):
    units: typing.Optional[str] = None
    a1: list
    a2: list
    a3: list


class DirectLattice(pydantic.BaseModel):
    a1: list
    a2: list
    a3: list


class ReciprocalLattice(pydantic.BaseModel):
    b1: list
    b2: list
    b3: list


class Atom(pydantic.BaseModel):
    species: str
    basis_vector: list


class Atoms(pydantic.BaseModel):
    units: typing.Optional[str] = None
    atoms: typing.List[Atom]


class Projections(pydantic.BaseModel):
    units: typing.Optional[str] = None
    projections: typing.List[str]


class Kpoints(pydantic.BaseModel):
    kpoints: typing.List[list]


class ExcludeBands(pydantic.BaseModel):
    exclude_bands: typing.Optional[typing.List[int]] = None


class WIN(pydantic.BaseModel):
    comments: typing.List[str]
    parameters: dict
    blocks: dict
    unit_cell_cart: typing.Optional[UnitCell] = None
    atoms_frac: typing.Optional[Atoms] = None
    atoms_cart: typing.Optional[Atoms] = None
    projections: Projections
    kpoints: Kpoints


class NNKP(pydantic.BaseModel):
    comments: typing.List[str]
    parameters: dict
    blocks: dict
    nnkpts: typing.List[typing.List[int]]
    direct_lattice: DirectLattice
    reciprocal_lattice: ReciprocalLattice
    kpoints: Kpoints
    exclude_bands: ExcludeBands
