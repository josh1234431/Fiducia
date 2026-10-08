/**
 * How a photograph is turned for display.
 *
 * Rotation and flipping change the view, never the data: the file is not
 * resampled and every measurement stays in the scan's own pixel coordinates,
 * so interior orientation, control and the adjustment are untouched. Two
 * coordinate systems are involved:
 *
 *   image    (col, row) on the scan as stored
 *   display  (u, v) on the turned picture the operator sees
 *
 * The turn is a quarter-turn rotation clockwise, then an optional mirror
 * left-to-right. Everything on the canvas converts through the two functions
 * here, so tiles, markers, residual arrows and clicks cannot disagree.
 */

export const IDENTITY = { rotate: 0, flipX: false }

export function normalise(orientation) {
  const rotate = (((Math.round((orientation?.rotate || 0) / 90) * 90) % 360) + 360) % 360
  return { rotate, flipX: !!orientation?.flipX }
}

/** Size of the picture as displayed. */
export function displaySize(width, height, orientation) {
  const { rotate } = normalise(orientation)
  return rotate % 180 === 0 ? { width, height } : { width: height, height: width }
}

/** Image pixel -> display pixel. */
export function toDisplay(col, row, width, height, orientation) {
  const { rotate, flipX } = normalise(orientation)
  let u
  let v
  if (rotate === 90) { u = height - row; v = col }
  else if (rotate === 180) { u = width - col; v = height - row }
  else if (rotate === 270) { u = row; v = width - col }
  else { u = col; v = row }
  if (flipX) u = displaySize(width, height, orientation).width - u
  return { u, v }
}

/** Display pixel -> image pixel. The exact inverse of toDisplay. */
export function toImage(u, v, width, height, orientation) {
  const { rotate, flipX } = normalise(orientation)
  if (flipX) u = displaySize(width, height, orientation).width - u
  if (rotate === 90) return { col: v, row: height - u }
  if (rotate === 180) return { col: width - u, row: height - v }
  if (rotate === 270) return { col: width - v, row: u }
  return { col: u, row: v }
}

/** The same turn as a CSS matrix(), for the layer that holds the tiles. */
export function cssMatrix(width, height, orientation) {
  const o = toDisplay(0, 0, width, height, orientation)
  const x = toDisplay(1, 0, width, height, orientation)
  const y = toDisplay(0, 1, width, height, orientation)
  return `matrix(${x.u - o.u}, ${x.v - o.v}, ${y.u - o.u}, ${y.v - o.v}, ${o.u}, ${o.v})`
}

/** Turn a direction (such as a residual vector) without moving it. */
export function turnVector(dx, dy, width, height, orientation) {
  const o = toDisplay(0, 0, width, height, orientation)
  const p = toDisplay(dx, dy, width, height, orientation)
  return { dx: p.u - o.u, dy: p.v - o.v }
}

export function rotateBy(orientation, degrees) {
  const current = normalise(orientation)
  // Mirrored, a clockwise turn of the picture is an anticlockwise turn of
  // the scan underneath it, so the operator's ⟳ always turns what they see.
  const step = current.flipX ? -degrees : degrees
  return normalise({ ...current, rotate: current.rotate + step })
}

export function flip(orientation) {
  const current = normalise(orientation)
  return { ...current, flipX: !current.flipX }
}
