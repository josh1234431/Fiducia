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
  sign: 1,
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

const LO_MERIDIANS = [15, 17, 19, 21, 23, 25, 27, 29, 31, 33]

/**
 * A South African Lo zone stored north-oriented, as easting and northing:
 * how ArcGIS and many elevation models keep Lo data. Its values are the
 * surveyor's Y and X with their signs turned.
 */
export function isLoEastingNorthing(described) {
  if (!described || described.isGeographic) return false
  if (/^ZALO\d+(_WGS84)?_EN$/.test(described.identifier || '')) return true
  const proj = described.proj4 || ''
  const lon = /\+lon_0=(-?[\d.]+)/.exec(proj)
  return /\+proj=tmerc/.test(proj) && !/\+axis=wsu/.test(proj)
    && /\+lat_0=0(\s|$)/.test(proj) && /\+x_0=0(\s|$)/.test(proj) && /\+y_0=0(\s|$)/.test(proj)
    && !!lon && LO_MERIDIANS.includes(Number(lon[1]))
}

/**
 * Names for the two horizontal coordinates, and the sign to show them with.
 * With loAsYX, a north-oriented Lo zone is shown as the surveyor's Y and X:
 * sign -1 turns the stored easting and northing into westing and southing
 * for display and entry. What is stored never changes.
 */
export function axisLabels(described, { loAsYX = false } = {}) {
  if (!described) return DEFAULT
  if (loAsYX && isLoEastingNorthing(described)) {
    return {
      first: 'Y', second: 'X', firstShort: 'Y', secondShort: 'X',
      elevation: 'Elevation',
      sign: -1,
      note: "Shown as the surveyor's Y (westing) and X (southing): the stored easting and "
        + 'northing with their signs turned.',
    }
  }

  if (described.isGeographic) {
    // Coordinates are handled longitude-first throughout, whatever order the
    // definition lists its axes in.
    return {
      sign: 1,
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
    sign: 1,
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
