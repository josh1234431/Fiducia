/**
 * What to call a control point's two horizontal coordinates.
 *
 * "X and Y" says nothing about which way either one runs, and on a
 * south-oriented system that ambiguity mirrors a block. So the names come
 * from the projection itself: Easting and Northing on UTM, Westing and
 * Southing on the South African Lo zones, longitude and latitude on a
 * geographic system.
 *
 * The stored order never changes — the first coordinate is always the first
 * axis of the projection — only what the operator is told it is.
 */

const DEFAULT = {
  first: 'Easting',
  second: 'Northing',
  firstShort: 'E',
  secondShort: 'N',
  elevation: 'Elevation',
  note: '',
}

const SHORT = {
  Easting: 'E', Northing: 'N', Westing: 'Y', Southing: 'X',
  Longitude: 'Lon', Latitude: 'Lat',
}

export function axisLabels(described) {
  if (!described) return DEFAULT

  if (described.isGeographic) {
    // Coordinates are handled longitude-first throughout, whatever order the
    // definition lists its axes in.
    return {
      first: 'Longitude', second: 'Latitude',
      firstShort: 'Lon', secondShort: 'Lat',
      elevation: 'Elevation',
      note: 'Degrees, longitude first.',
    }
  }

  const names = (described.axes || []).map((axis) => axis.name)
  const first = names[0] || DEFAULT.first
  const second = names[1] || DEFAULT.second
  const southOriented = first === 'Westing' || second === 'Southing'

  return {
    first,
    second,
    firstShort: SHORT[first] || first.slice(0, 1),
    secondShort: SHORT[second] || second.slice(0, 1),
    elevation: 'Elevation',
    // The Lo convention: the surveyor's Y is the westing, X the southing.
    note: southOriented
      ? 'South-oriented: Y is the westing, X the southing.'
      : '',
  }
}
