# (C) British Crown Copyright 2018, Met Office
#
# This file is part of Iris.
#
# Iris is free software: you can redistribute it and/or modify it under
# the terms of the GNU Lesser General Public License as published by the
# Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# Iris is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with Iris.  If not, see <http://www.gnu.org/licenses/>.

from __future__ import (absolute_import, division, print_function)
from six.moves import (filter, input, map, range, zip)  # noqa
import six

from collections import deque, namedtuple
from copy import deepcopy

import numpy as np

import iris.exceptions
import iris.coords
from iris._lazy_data import (as_concrete_data, as_lazy_data, is_lazy_data,
                             multidim_lazy_stack)


def _all_same(a):
    a = np.asarray(a)
    return np.all(a[0] == a)


def _element_indices(a):
    """
    [10, 20, 10, 30, 20, 30]

    -> { 10: [0, 2], 20: [1, 4], 30: [3, 5]}

    [15, 25, 25, 15, 15, 25]

    -> { 15: [0, 3, 4], 25: [1, 2, 5]}

    """
    indices = {}
    for i, e in enumerate(a):
        if e in indices:
            indices[e].append(i)
        else:
            indices[e] = deque([i])
    return indices


def _cells_points(cells, dtype):
    ret = np.empty_like(cells, dtype=dtype)
    ret.flat = [cell.point for cell in cells.flat]
    return ret


def _cells_bounds(cells, dtype):
    if cells.size > 0 and cells.flat[0].bound is not None:
        ret = np.empty(cells.shape + (2,), dtype=dtype)
        ret.flat = np.array([cell.bound for cell in cells.flat])
        return ret
    else:
        return None


class _CoordMetaData(namedtuple('CoordMetaData',
                                ['points_dtype', 'bounds_dtype', 'kwargs'])):
    """
    Bespoke metadata required to build a dimension or auxiliary coordinate.

    Args:

    * points_dtype:
        The points data :class:`numpy.dtype` of an associated coordinate.
        None otherwise.

    * bounds_dtype:
        The bounds data :class:`numpy.dtype` of an associated coordinate.
        None otherwise.

    * kwargs:
        A dictionary of key/value pairs required to create a coordinate.

    """

    __slots__ = ()


class _CoordAndDims(namedtuple('CoordAndDims',
                               ['coord', 'dims'])):
    """
    Container for a coordinate and the associated data dimension/s
    spanned over a :class:`iris.cube.Cube`.

    Args:

    * coord:
        A :class:`iris.coords.DimCoord` or :class:`iris.coords.AuxCoord`
        coordinate instance.

    * dims:
        A tuple of the data dimension/s spanned by the coordinate.

    """

    __slots__ = ()


class _ScalarCoordPayload(namedtuple('ScalarCoordPayload',
                                     ['defns', 'values', 'metadata'])):
    """
    Container for all scalar coordinate data and metadata represented
    within a :class:`iris.cube.Cube`.

    All scalar coordinate related data is sorted into ascending order
    of the associated coordinate definition.

    Args:

    * defns:
        A list of scalar coordinate definitions :class:`iris.coords.CoordDefn`
        belonging to a :class:`iris.cube.Cube`.

    * values:
        A list of scalar coordinate values belonging to a
        :class:`iris.cube.Cube`.  Each scalar coordinate value is
        typically an :class:`iris.coords.Cell`.

    * metadata:
        A list of :class:`_CoordMetaData` instances belonging to a
        :class:`iris.cube.Cube`.

    """

    __slots__ = ()


class _VectorCoordPayload(namedtuple('VectorCoordPayload',
                                     ['dim_coords_and_dims',
                                      'aux_coords_and_dims'])):
    """
    Container for all vector coordinate data and metadata represented
    within a :class:`iris.cube.Cube`.

    Args:

    * dim_coords_and_dims:
        A list of :class:`_CoordAndDim` instances containing non-scalar
        (i.e. multi-valued) :class:`iris.coords.DimCoord` instances and
        the associated data dimension spanned by them for a
        :class:`iris.cube.Cube`.

    * aux_coords_and_dims:
        A list of :class:`_CoordAndDim` instances containing non-scalar
        (i.e. multi-valued) :class:`iris.coords.DimCoord` and/or
        :class:`iris.coords.AuxCoord` instances and the associated data
        dimension/s spanned by them for a :class:`iris.cube.Cube`.

    """

    __slots__ = ()


class _CoordPayload(namedtuple('CoordPayload',
                               ['scalar', 'vector', 'factory_defns'])):
    """
    Container for all the scalar and vector coordinate data and
    metadata, and auxiliary coordinate factories represented within a
    :class:`iris.cube.Cube`.

    All scalar coordinate and factory related data is sorted into
    ascending order of the associated coordinate definition.

    Args:

    * scalar:
        A :class:`_ScalarCoordPayload` instance.

    * vector:
        A :class:`_VectorCoordPayload` instance.

    * factory_defns:
        A list of :class:`_FactoryDefn` instances.

    """

    __slots__ = ()

    def as_signature(self):
        """Construct and return a :class:`_CoordSignature` from the payload."""

        return _CoordSignature(self.scalar.defns,
                               self.vector.dim_coords_and_dims,
                               self.vector.aux_coords_and_dims,
                               self.factory_defns)

    @staticmethod
    def _coords_msgs(msgs, coord_group, defns_a, defns_b):
        if defns_a != defns_b:
            # Get a new list so we can modify it
            defns_b = list(defns_b)
            diff_defns = []
            for defn_a in defns_a:
                try:
                    defns_b.remove(defn_a)
                except ValueError:
                    diff_defns.append(defn_a)
            diff_defns.extend(defns_b)
            if diff_defns:
                names = sorted(set(defn.name() for defn in diff_defns))
                msgs.append('Coordinates in {} differ: {}.'.format(
                    coord_group, ', '.join(names)))
            else:
                msgs.append('Coordinates in {} differ by dtype or class'
                            ' (i.e. DimCoord vs AuxCoord).'.format(
                                coord_group))

    def match_signature(self, signature, error_on_mismatch):
        """
        Return whether this _CoordPayload matches the corresponding
        aspects of a _CoordSignature.

        Args:

        * signature (_CoordSignature):
            The _CoordSignature to compare against.

        * error_on_mismatch (bool):
            If True, raise an Exception with detailed explanation.

        Returns:
           Boolean. True if and only if this _CoordPayload matches
           the corresponding aspects `other`.

        """
        def unzip(coords_and_dims):
            if coords_and_dims:
                coords, dims = zip(*coords_and_dims)
            else:
                coords, dims = [], []
            return coords, dims

        def dims_msgs(msgs, coord_group, dimlists_a, dimlists_b):
            if dimlists_a != dimlists_b:
                msgs.append(
                    'Coordinate-to-dimension mapping differs for {}.'.format(
                        coord_group))

        msgs = []
        self._coords_msgs(msgs, 'cube.aux_coords (scalar)', self.scalar.defns,
                          signature.scalar_defns)

        coord_group = 'cube.dim_coords'
        self_coords, self_dims = unzip(self.vector.dim_coords_and_dims)
        other_coords, other_dims = unzip(signature.vector_dim_coords_and_dims)
        self._coords_msgs(msgs, coord_group, self_coords, other_coords)
        dims_msgs(msgs, coord_group, self_dims, other_dims)

        coord_group = 'cube.aux_coords (non-scalar)'
        self_coords, self_dims = unzip(self.vector.aux_coords_and_dims)
        other_coords, other_dims = unzip(signature.vector_aux_coords_and_dims)
        self._coords_msgs(msgs, coord_group, self_coords, other_coords)
        dims_msgs(msgs, coord_group, self_dims, other_dims)

        if self.factory_defns != signature.factory_defns:
            msgs.append('cube.aux_factories() differ')

        match = not bool(msgs)
        if error_on_mismatch and not match:
            raise iris.exceptions.MergeError(msgs)
        return match


class _CoordSignature(namedtuple('CoordSignature',
                                 ['scalar_defns',
                                  'vector_dim_coords_and_dims',
                                  'vector_aux_coords_and_dims',
                                  'factory_defns'])):
    """
    Criterion for identifying a specific type of :class:`iris.cube.Cube`
    based on its scalar and vector coorinate data and metadata, and
    auxiliary coordinate factories.

    Args:

    * scalar_defns:
        A list of scalar coordinate definitions sorted into ascending order.

    * vector_dim_coords_and_dims:
        A list of :class:`_CoordAndDim` instances containing non-scalar
        (i.e. multi-valued) :class:`iris.coords.DimCoord` instances and
        the associated data dimension spanned by them for a
        :class:`iris.cube.Cube`.

    * vector_aux_coords_and_dims:
        A list of :class:`_CoordAndDim` instances containing non-scalar
        (i.e. multi-valued) :class:`iris.coords.DimCoord` and/or
        :class:`iris.coords.AuxCoord` instances and the associated data
        dimension/s spanned by them for a :class:`iris.cube.Cube`.

    * factory_defns:
        A list of :class:`_FactoryDefn` instances.

    """

    __slots__ = ()


class _CubeSignature(namedtuple('CubeSignature',
                                ['defn', 'data_shape', 'data_type',
                                 'cell_measures_and_dims'])):
    """
    Criterion for identifying a specific type of :class:`iris.cube.Cube`
    based on its metadata.

    Args:

    * defn:
        A cube definition tuple.

    * data_shape:
        The data payload shape of a :class:`iris.cube.Cube`.

    * data_type:
        The data payload :class:`numpy.dtype` of a :class:`iris.cube.Cube`.

    * cell_measures_and_dims:
        A list of cell_measures and dims for the cube.

    """

    __slots__ = ()

    def _defn_msgs(self, other_defn):
        msgs = []
        self_defn = self.defn
        if self_defn.standard_name != other_defn.standard_name:
            msgs.append('cube.standard_name differs: {!r} != {!r}'.format(
                self_defn.standard_name, other_defn.standard_name))
        if self_defn.long_name != other_defn.long_name:
            msgs.append('cube.long_name differs: {!r} != {!r}'.format(
                self_defn.long_name, other_defn.long_name))
        if self_defn.var_name != other_defn.var_name:
            msgs.append('cube.var_name differs: {!r} != {!r}'.format(
                self_defn.var_name, other_defn.var_name))
        if self_defn.units != other_defn.units:
            msgs.append('cube.units differs: {!r} != {!r}'.format(
                self_defn.units, other_defn.units))
        if self_defn.attributes != other_defn.attributes:
            diff_keys = (set(self_defn.attributes.keys()) ^
                         set(other_defn.attributes.keys()))
            if diff_keys:
                msgs.append('cube.attributes keys differ: ' +
                            ', '.join(repr(key) for key in diff_keys))
            else:
                diff_attrs = [repr(key) for key in self_defn.attributes
                              if np.all(self_defn.attributes[key] !=
                                        other_defn.attributes[key])]
                diff_attrs = ', '.join(diff_attrs)
                msgs.append(
                    'cube.attributes values differ for keys: {}'.format(
                        diff_attrs))
        if self_defn.cell_methods != other_defn.cell_methods:
            msgs.append('cube.cell_methods differ')
        return msgs

    def match(self, other, error_on_mismatch):
        """
        Return whether this _CubeSignature equals another.

        This is the first step to determine if two "cubes" (either a
        real Cube or a ProtoCube) can be merged, by considering:
            - standard_name, long_name, var_name
            - units
            - attributes
            - cell_methods
            - shape, dtype

        Args:

        * other (_CubeSignature):
            The _CubeSignature to compare against.

        * error_on_mismatch (bool):
            If True, raise a :class:`~iris.exceptions.MergeException`
            with a detailed explanation if the two do not match.

        Returns:
           Boolean. True if and only if this _CubeSignature matches `other`.

        """
        msgs = self._defn_msgs(other.defn)
        if self.data_shape != other.data_shape:
            msg = 'cube.shape differs: {} != {}'
            msgs.append(msg.format(self.data_shape, other.data_shape))
        if self.data_type != other.data_type:
            msg = 'cube data dtype differs: {} != {}'
            msgs.append(msg.format(self.data_type, other.data_type))
        if (self.cell_measures_and_dims != other.cell_measures_and_dims):
            msgs.append('cube.cell_measures differ')

        match = not bool(msgs)
        if error_on_mismatch and not match:
            raise iris.exceptions.MergeError(msgs)
        return match


class _Skeleton(namedtuple('Skeleton',
                           ['scalar_values', 'data'])):
    """
    Basis of a source-cube, containing the associated scalar coordinate values
    and data payload of a :class:`iris.cube.Cube`.

    Args:

    * scalar_values:
        A list of scalar coordinate values belonging to a
        :class:`iris.cube.Cube` sorted into ascending order of the
        associated coordinate definition. Each scalar coordinate value
        is typically an :class:`iris.coords.Cell`.

    * data:
        The data payload of a :class:`iris.cube.Cube`.

    """

    __slots__ = ()


class _FactoryDefn(namedtuple('_FactoryDefn',
                              ['class_', 'dependency_defns'])):
    """
    The information required to identify and rebuild a single AuxCoordFactory.

    Args:

    * class_:
        The class of the AuxCoordFactory.

    * dependency_defns:
        A list of pairs, where each pair contains a dependency key and its
        corresponding coordinate definition. Sorted on dependency key.

    """

    __slots__ = ()


class ProtoCube(object):
    """
    Framework for merging source-cubes into one or more higher
    dimensional cubes.

    """

    def __init__(self, cube):
        """
        Create a new ProtoCube from the given cube and record the cube
        as a source-cube.

        """

        # Default hint ordering for candidate dimension coordinates.
        self._hints = ['time', 'forecast_reference_time', 'forecast_period']

        # The proto-cube source.
        self._source = cube

        # The cube signature is metadata that defines this ProtoCube.
        self._cube_signature = self._build_signature(cube)

        # Extract the scalar and vector coordinate data and metadata
        # from the cube.
        coord_payload = self._extract_coord_payload(cube)

        # The coordinate signature defines the scalar and vector
        # coordinates of this ProtoCube.
        self._coord_signature = coord_payload.as_signature()
        self._coord_metadata = coord_payload.scalar.metadata

        # The list of stripped-down source-cubes relevant to this ProtoCube.
        self._skeletons = []
        self._add_cube(cube, coord_payload)

        # Proto-coordinates constructed from merged scalars.
        self._dim_templates = []
        self._aux_templates = []

        # During the merge this will contain the complete, merged shape
        # of a result cube.
        # E.g. Merging three (72, 96) cubes would give:
        #      self._shape = (3, 72, 96).
        self._shape = []
        # During the merge this will contain the shape of the "stack"
        # of cubes used to create a single result cube.
        # E.g. Merging three (72, 96) cubes would give:
        #      self._stack_shape = (3,)
        self._stack_shape = []

        self._nd_names = []
        self._cache_by_name = {}
        self._dim_coords_and_dims = []
        self._aux_coords_and_dims = []

        # cell measures are not merge candidates
        # they are checked and preserved through merge
        self._cell_measures_and_dims = cube._cell_measures_and_dims

    def _extract_coord_payload(self, cube):
        """
        Extract all relevant coordinate data and metadata from the cube.

        In particular, for each scalar coordinate determine its definition,
        its cell (point and bound) value and all other scalar coordinate
        metadata that allows us to fully reconstruct that scalar
        coordinate. Note that all scalar data is sorted in order of the
        scalar coordinate definition.

        The coordinate payload of the cube also includes any associated vector
        coordinates that describe that cube, and descriptions of any auxiliary
        coordinate factories.

        """
        scalar_defns = []
        scalar_values = []
        scalar_metadata = []
        vector_dim_coords_and_dims = []
        vector_aux_coords_and_dims = []

        cube_aux_coords = cube.aux_coords
        coords = cube.dim_coords + cube_aux_coords
        cube_aux_coord_ids = {id(coord) for coord in cube_aux_coords}

        # Coordinate hint ordering dictionary - from most preferred to least.
        # Copes with duplicate hint entries, where the most preferred is king.
        hint_dict = {name: i for i, name in zip(range(len(self._hints), 0, -1),
                                                self._hints[::-1])}
        # Coordinate axis ordering dictionary.
        axis_dict = {'T': 1, 'Z': 2, 'Y': 3, 'X': 4}

        # Coordinate sort function.
        # NB. This makes use of two properties which don't end up in
        # the CoordDefn used by scalar_defns: `coord.points.dtype` and
        # `type(coord)`.
        def key_func(coord):
            points_dtype = coord.dtype
            return (not np.issubdtype(points_dtype, np.number),
                    not isinstance(coord, iris.coords.DimCoord),
                    axis_dict.get(iris.util.guess_coord_axis(coord), 0),
                    hint_dict.get(coord.name(), 0),
                    coord._as_defn())

        # Order the coordinates by hints, axis, and definition.
        for coord in sorted(coords, key=key_func):
            if not cube.coord_dims(coord) and coord.shape == (1,):
                # Extract the scalar coordinate data and metadata.
                scalar_defns.append(coord._as_defn())
                # Because we know there's a single Cell in the
                # coordinate, it's quicker to roll our own than use
                # Coord.cell().
                points = coord.points
                bounds = coord.bounds
                points_dtype = points.dtype
                if bounds is not None:
                    bounds_dtype = bounds.dtype
                    bounds = bounds[0]
                else:
                    bounds_dtype = None
                scalar_values.append(iris.coords.Cell(points[0], bounds))
                kwargs = {}
                if isinstance(coord, iris.coords.DimCoord):
                    kwargs['circular'] = coord.circular
                scalar_metadata.append(_CoordMetaData(points_dtype,
                                                      bounds_dtype, kwargs))
            else:
                # Extract the vector coordinate and metadata.
                if id(coord) in cube_aux_coord_ids:
                    vector_aux_coords_and_dims.append(
                        _CoordAndDims(coord, tuple(cube.coord_dims(coord))))
                else:
                    vector_dim_coords_and_dims.append(
                        _CoordAndDims(coord, tuple(cube.coord_dims(coord))))

        factory_defns = []
        for factory in sorted(cube.aux_factories,
                              key=lambda factory: factory._as_defn()):
            dependency_defns = []
            dependencies = factory.dependencies
            for key in sorted(dependencies):
                coord = dependencies[key]
                if coord is not None:
                    dependency_defns.append((key, coord._as_defn()))
            factory_defn = _FactoryDefn(type(factory), dependency_defns)
            factory_defns.append(factory_defn)

        scalar = _ScalarCoordPayload(scalar_defns, scalar_values,
                                     scalar_metadata)
        vector = _VectorCoordPayload(vector_dim_coords_and_dims,
                                     vector_aux_coords_and_dims)

        return _CoordPayload(scalar, vector, factory_defns)

    def _report_duplicate(self, scalar_values):
        name = self._cube_signature.defn.name()
        scalars = []
        for defn, value in zip(self._coord_signature.scalar_defns,
                               scalar_values):
            scalars.append('%s=%r' % (defn.name(), value))
        msg = 'Duplicate %r cube, with scalar coordinates %s'
        msg = msg % (name, ', '.join(scalars))
        raise iris.exceptions.DuplicateDataError(msg)

    def register(self, cube, error_on_mismatch=False):
        """
        Add a compatible :class:`iris.cube.Cube` as a source-cube for
        merging under this :class:`ProtoCube`.

        A cube will be deemed compatible based on the signature of the
        cube and the signature of its scalar coordinates and vector
        coordinates being identical to that of the ProtoCube.

        Args:

        * cube:
            Candidate :class:`iris.cube.Cube` to be associated with
            this :class:`ProtoCube`.

        Kwargs:

        * error_on_mismatch:
            If True, raise an informative
            :class:`~iris.exceptions.MergeError` if registration fails.

        Returns:
            True iff the :class:`iris.cube.Cube` is compatible with
            this :class:`ProtoCube`.

        """
        cube_signature = self._cube_signature
        other = self._build_signature(cube)
        match = cube_signature.match(other, error_on_mismatch)
        if match:
            coord_payload = self._extract_coord_payload(cube)
            match = coord_payload.match_signature(self._coord_signature,
                                                  error_on_mismatch)
            if match:
                # Register the cube as a source-cube for this ProtoCube.
                self._add_cube(cube, coord_payload)
        return match

    def _order_and_reshape(self, scalar_values, data_stack, shape, coord_dims):
        if coord_dims:
            order = np.lexsort(scalar_values[coord_dims[::-1]])

            scalar_values = scalar_values[:, order]
            data_stack = data_stack[order]

        scalar_values.shape = (-1,) + shape
        data_stack.shape = shape

        return scalar_values, data_stack

    def _choose_new_dims(self, candidate_shapes, candidate_dim_indices):
        # XXX: Just choose the one with the most dimensions for now.
        print(candidate_shapes, candidate_dim_indices)
        i = np.argmax([len(cs) for cs in candidate_shapes])
        return candidate_shapes[i], candidate_dim_indices[i]

    def _build_coordinates(self, scalar_values, dim_indices):
        ndims = scalar_values.ndim - 1
        ndim_coords = len(dim_indices)
        dim_coord_dims = np.repeat(-1, len(scalar_values))
        dim_coord_dims[dim_indices] = range(ndim_coords)

        dim_coords_and_dims = []
        aux_coords_and_dims = []

        # scalar_values may be 0-d, so we add an extra axis when looping to be
        # able to treat the 0-d and N-d cases uniformally.
        for values, defn, metadata, dim_coord_dim in zip(
                scalar_values[..., None],
                self._coord_signature.scalar_defns,
                self._coord_metadata,
                dim_coord_dims):
            kwargs = defn._asdict()
            kwargs.update(metadata.kwargs)
            if dim_coord_dim != -1:
                value_slice = (0,) * dim_coord_dim + \
                              (slice(None),) + \
                              (0,) * (ndims - dim_coord_dim - 1)
                points = _cells_points(values[value_slice],
                                       metadata.points_dtype)[..., 0]
                bounds = _cells_bounds(values[value_slice],
                                       metadata.bounds_dtype)
                if bounds is not None:
                    bounds = bounds[..., 0, :]
                dim_coords_and_dims.append(
                    (iris.coords.DimCoord(points, bounds=bounds, **kwargs),
                     dim_coord_dim))
            else:
                if values.ndim == 1:
                    # No new dimensions
                    value_slice = ()
                    coord_dims = ()
                else:
                    # Find the indices spanned by the aux coord. If all the
                    # slices along a given dimension are the same, the coord
                    # does not span that dimension.
                    spans_dims = [np.any(values.take([0], axis=dim) != values)
                                  for dim in range(ndims)]
                    value_slice = tuple(slice(None) if spans_dim else 0
                                        for spans_dim in spans_dims)
                    coord_dims = tuple(np.where(spans_dims)[0])

                points = _cells_points(values[value_slice],
                                       metadata.points_dtype)[..., 0]
                bounds = _cells_bounds(values[value_slice],
                                       metadata.bounds_dtype)
                if bounds is not None:
                    bounds = bounds[..., 0, :]

                try:
                    coord = iris.coords.DimCoord(points, bounds=bounds,
                                                 **kwargs)
                except ValueError:
                    kwargs.pop('circular', None)
                    coord = iris.coords.AuxCoord(points, bounds=bounds,
                                                 **kwargs)

                aux_coords_and_dims.append((coord, coord_dims))

        return dim_coords_and_dims, aux_coords_and_dims

    def _get_cube(self, data, dim_coords_and_dims, aux_coords_and_dims):
        """
        Return a fully constructed cube for the given data, containing
        all its coordinates and metadata.

        """
        signature = self._cube_signature
        dim_coords_and_dims = [(deepcopy(coord), dim)
                               for coord, dim in dim_coords_and_dims]
        aux_coords_and_dims = [(deepcopy(coord), dims)
                               for coord, dims in aux_coords_and_dims]
        kwargs = dict(zip(iris.cube.CubeMetadata._fields, signature.defn))

        cms_and_dims = [(deepcopy(cm), dims)
                        for cm, dims in self._cell_measures_and_dims]
        cube = iris.cube.Cube(data,
                              dim_coords_and_dims=dim_coords_and_dims,
                              aux_coords_and_dims=aux_coords_and_dims,
                              cell_measures_and_dims=cms_and_dims,
                              **kwargs)

        # Add on any aux coord factories.
        for factory_defn in self._coord_signature.factory_defns:
            args = {}
            for key, defn in factory_defn.dependency_defns:
                coord = cube.coord(defn)
                args[key] = coord
            factory = factory_defn.class_(**args)
            cube.add_aux_factory(factory)

        return cube

    def get_first_duplicate(self):
        scalar_values = set()
        for skeleton in self._skeletons:
            sv = tuple(skeleton.scalar_values)
            if sv in scalar_values:
                return sv
            scalar_values.add(sv)
        return None

    def merge(self, unique=True):
        if unique:
            duplicate_scalar_values = self.get_first_duplicate()
            if duplicate_scalar_values:
                self._report_duplicate(duplicate_scalar_values)

        ncoords = len(self._coord_metadata)
        nvalues = len(self._skeletons)

        data_stack = np.empty(nvalues, 'object')
        any_lazy = False
        for i in range(nvalues):
            data_i = self._skeletons[i].data
            if is_lazy_data(data_i):
                any_lazy = True
            else:
                data_i = as_lazy_data(data_i)
            data_stack[i] = data_i

        scalar_values = np.empty((ncoords, nvalues), dtype=object)
        scalar_values.T[:] = [skeleton.scalar_values
                              for skeleton in self._skeletons]
        # Only consider numeric coordinates when searching for potential dim
        # coords.
        is_numeric = np.array([metadata.points_dtype.kind not in 'SU'
                               for metadata in self._coord_metadata],
                              dtype=bool)
        #is_monotonic = np.array([_cells_points(scalar_values)])

        numeric_scalar_values = scalar_values[is_numeric]
        row_indices = np.arange(ncoords)[is_numeric]

        candidate_shapes, candidate_dim_indices = \
            self._get_new_dims_candidates(numeric_scalar_values, row_indices)

        if candidate_shapes:
            shape, dim_indices = self._choose_new_dims(candidate_shapes,
                                                       candidate_dim_indices)
            scalar_values, data_stack = self._order_and_reshape(scalar_values,
                                                                data_stack,
                                                                shape,
                                                                dim_indices)
        else:
            shape = (nvalues,)
            dim_indices = []
            scalar_values, data_stack = self._order_and_reshape(scalar_values,
                                                                data_stack,
                                                                shape,
                                                                None)

        new_dim_coords_and_dims, new_aux_coords_and_dims = \
            self._build_coordinates(scalar_values, dim_indices)

        # Create lists combining the new vector coordinates with the existing
        # ones, offsetting the dims of the existing coordinates by the number
        # of new dimensions.
        offset = len(shape)

        dim_coords_and_dims = \
            new_dim_coords_and_dims + \
            [(coord, (dim + offset,)) for coord, (dim,) in
             self._coord_signature.vector_dim_coords_and_dims]
        aux_coords_and_dims = \
            new_aux_coords_and_dims + \
            [(coord, tuple(dim + offset for dim in dims)) for coord, dims in
             self._coord_signature.vector_aux_coords_and_dims]

        data = multidim_lazy_stack(data_stack)
        if not any_lazy:
            data = as_concrete_data(data)

        return [self._get_cube(data, dim_coords_and_dims, aux_coords_and_dims)]

    def _build_signature(self, cube):
        """
        Generate the signature that defines this cube.

        Args:

        * cube:
            The source cube to create the cube signature from.

        Returns:
            The cube signature.

        """

        return _CubeSignature(cube.metadata, cube.shape,
                              cube.dtype, cube._cell_measures_and_dims)

    def _add_cube(self, cube, coord_payload):
        """Create and add the source-cube skeleton to the ProtoCube."""
        skeleton = _Skeleton(coord_payload.scalar.values,
                             cube.core_data())
        # Attempt to do something sensible with mixed scalar dtypes.
        for i, metadata in enumerate(coord_payload.scalar.metadata):
            if metadata.points_dtype > self._coord_metadata[i].points_dtype:
                self._coord_metadata[i] = metadata
        self._skeletons.append(skeleton)

    @staticmethod
    def _get_new_dims_candidates(scalar_values, row_indices=None):
        """
        Args:

        * scalar_values: list, N*M.
        * all_scalar_values: list, L*M. The original array.
        * row_indices: list, N. The row indices in the original array,
            for determining dim_coords indices.

        Returns:

        * candidate_shapes: list, K.
        * candidate_dim_coords: list, K.

        [[1, 2]]

         --> [(2,)], [[0]]


        [[1, 1, 2, 2, 3, 3],
         [1, 2, 1, 2, 1, 2],
         [1, 2, 1, 2, 2, 1]]

         --> [(3, 2), (3, 2)],
             [[0, 1], [0, 2]]

        """

        if row_indices is None:
            row_indices = np.arange(len(scalar_values))

        ncoords, nvalues = scalar_values.shape
        DEBUG(scalar_values.shape)

        if nvalues < 2:
            return [()], [[]]

        # For each row, get a dictionary of the indices for each distinct
        # element.
        # E.g. the element indices for [10, 20, 10, 30, 30] are:
        # {10: [0, 2],
        #  20: [1],
        #  30: [3, 4]}
        all_element_indices = [_element_indices(a) for a in scalar_values]

        candidate_shapes = []
        candidate_dim_indices = []
        for element_indices in all_element_indices:
            row = row_indices[0]
            DEBUG('Trying row {}'.format(row))
            indices = element_indices.values()
            counts = np.array([len(v) for v in indices])
            # To be a dim coord, the number of occurrences of each element must
            # be the same.
            if _all_same(counts) and counts[0] < nvalues:
                count = counts[0]
                dim_len = nvalues // count
                DEBUG('It is a potential candidate dimension of length {}'
                      .format(dim_len))
                DEBUG('    {}'.format(indices), level=2)

                if count == 1:
                    # If there is only one occurence of each element then this
                    # row can form a dimension with no additional dimensions.
                    DEBUG('Its values are all distinct')
                    new_shape = (dim_len, count) if count > 1 else (dim_len,)
                    DEBUG('Appending shape: {}'.format(new_shape))
                    DEBUG('Appending row: {}'.format(row))
                    candidate_shapes.append(new_shape)
                    candidate_dim_indices.append([row])
                else:
                    DEBUG('Its values are not distinct.')
                    DEBUG('Finding independent coords')
                    for a in scalar_values:
                        DEBUG([list(set(a[idx])) for idx in indices], level=2)
                    independents = []
                    # Otherwise, we need to find a row or rows that can form
                    # the remaining dimensions.
                    # Find which rows are 'independent' of this row. That is,
                    # for each of the other rows check that:
                    # - The sets of values at the indices of each element in
                    #   the current row are all the same.
                    #   E.g: current row [10, 10, 10, 11, 11, 11]
                    #        row1        [20, 21, 22, 20, 21, 23]
                    #        row2        [30, 31, 31, 31, 30, 31]
                    #        row3        [40, 41, 42, 41, 40, 42]
                    #   The indices are:
                    #                    {10: [0, 1, 2], 11: [3, 4, 5]}
                    #   The sets of values for each row at these indices are:
                    #        row1        [20, 21, 22], [20, 21, 23]
                    #        row2        [30, 31], [30, 31]
                    #        row3        [40, 41, 42], [40, 41, 42]
                    #   The set of values at each list of indices for row2 and
                    #   row3 are all the same.
                    # - The number of values for each element at the indices of
                    #   each element in the current row are all equal.
                    #   E.g: The set of values for each element at each list of
                    #   indices for row2 and row3 above are:
                    #        row2        {30: [0], 31: [1, 2]},
                    #                    {30: [1], 31: [0, 2]}
                    #        row3        {40: [0], 41: [1], 42: [2]},
                    #                    {40: [1], 41: [0], 42: [2]}
                    #   So the number of values for each element are:
                    #        row2        {30: 1, 31: 2}, {30: 1, 31: 2}
                    #        row3        {40: 1, 41: 1, 42: 1},
                    #                    {40: 1, 41: 1, 42: 1}
                    #   Therefore the number of values for each element at each
                    #   list of indices are the same for row3.
                    independents = []
                    for i, a in enumerate(scalar_values[1:]):
                        # Check whether the set of values at each list of
                        # indices are all the same for this row.
                        if _all_same([set(a[j]) for j in indices]):
                            sub_eis = [_element_indices(a[idx]) for idx in
                                       indices]
                            # Check whether the number of values for each
                            # element at each list of indices are all the same
                            # for this row.
                            if _all_same([len(j) for sub_ei in sub_eis
                                          for j in sub_ei.values()]):
                                # This row is independent of the current row.
                                independents.append(i + 1)

                    # Recurse with only the rows identified in the step above,
                    # passing only the values at one of the lists of indices.
                    # E.g. current row [10, 10, 10, 11, 11, 11]
                    #      row1        [20, 21, 22, 20, 21, 22]
                    #      row2        [30, 31, 32, 30, 34, 35]
                    #      row3        [41, 40, 42, 40, 41, 42]
                    # We recurse with:
                    #      row1        [20, 21, 22]
                    #      row3        [41, 40, 42]
                    # Then append the current row index and its length as a
                    # dimension coordinate to each element in the list of
                    # dimension coordinates and shapes identified at the next
                    # level of recursion respectively.
                    # E.g. current row length as a dim coord == 2. The next
                    # level of recursion identifies both row1 and row3 as
                    # candidate dimension coordinates, so the return values
                    # are:
                    #      sub_shapes = [(3,), (3,)],
                    #      sub_dim_coords = [[1], [3]]
                    # After appending the current row index and length as a
                    # dim_coord we have:
                    #      new_shapes = [(2, 3), (2, 3)]
                    #      new_dim_coords = [[0, 1], [0, 3]]
                    if independents:
                        # Only look at one subset of the values for the
                        # independent rows.
                        sub_values = \
                            scalar_values[independents][..., indices[0]]
                        DEBUG('There are potential extra dimensions at rows '
                              '{}'.format([row_indices[i]
                                           for i in independents]))
                        DEBUG('Recursing with new array:', shift=4)
                        DEBUG(sub_values, level=2)
                        sub_shapes, sub_dim_indices = \
                            ProtoCube._get_new_dims_candidates(
                                sub_values, row_indices[independents])
                        DEBUG('Extending shapes: {}'.format([(dim_len,) + shape
                                                             for shape in
                                                             sub_shapes]))
                        DEBUG('Extending rows: {}'.format([[row] + dim_indices
                                                           for dim_indices in
                                                           sub_dim_indices]))
                        # Extend each sub- dimension coord list and sub-shape
                        # with the current row index and length as a dimension
                        # coordinate.
                        candidate_shapes.extend(
                            [(dim_len,) + shape for shape in sub_shapes])
                        candidate_dim_indices.extend(
                            [[row] + dim_indices for dim_indices in
                             sub_dim_indices])
                    else:
                        # Add anonymous dimension
                        DEBUG('There are no possible extra dimension '
                              'coordinates. Adding this dimension with an '
                              'anonymous dimension.')
                        candidate_shapes.append((dim_len, count))
                        candidate_dim_indices.append([row])

            # Eliminate this row from the search.
            scalar_values = scalar_values[1:]
            row_indices = row_indices[1:]

        DEBUG('Shapes found at this level: {}'.format(candidate_shapes))
        DEBUG('Dim coords found at this level: {}'.format(
            candidate_dim_indices), shift=-4)

        return candidate_shapes, candidate_dim_indices

DEBUG_LEVEL = 1
ind = 0


def DEBUG(s, shift=0, level=1):
    global ind
    indent = '_' * ind
    if level <= DEBUG_LEVEL:
        print(indent + str(s))

    ind += shift
