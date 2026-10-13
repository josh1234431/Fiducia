/**
 * One rule for colouring numbers, used everywhere a residual or an accuracy
 * is shown, so a worse number is never greener than a better one.
 *
 * Fit -- how well the measurements agree with the solution: residuals on the
 * photos, control point residuals, control RMS. A careful measurement is good
 * to about a pixel, so a fit is amber above FIT_PIXELS pixels: on the photo
 * directly, and on the ground through the block's ground pixel. The engine
 * uses the same figure (plausibility.PIXELS_WARN).
 *
 * Accuracy -- how close the result is to the truth, from independent check
 * points. Whether that is good depends on the job, so it is only coloured
 * against an accuracy target set on the Project step, and left uncoloured
 * without one.
 */

export const FIT_PIXELS = 3

/** The block's typical ground pixel in metres, or null when it cannot be told. */
export function groundPixel(project) {
  const model = project?.model
  if (model?.groundPixelM) return model.groundPixelM
  if (project?.droneBlock?.gsdM) return project.droneBlock.gsdM
  // Solved before the engine recorded it: a digital camera's pixel and the
  // photos' height above the control.
  const pitch = project?.camera?.pixelPitchMm
  const focal = project?.camera?.focalMm
  const heights = (project?.gcps || []).map((g) => Number(g.z)).filter(Number.isFinite)
  const exterior = Object.values(model?.exterior || {})
  if (pitch > 0 && focal > 0 && heights.length && exterior.length) {
    const ground = heights.reduce((a, b) => a + b, 0) / heights.length
    const above = exterior.map((eo) => eo[2] - ground).filter((h) => h > 0).sort((a, b) => a - b)
    if (above.length) return pitch * above[Math.floor(above.length / 2)] / focal
  }
  return null
}

/** A length as people read it: 1.7 cm, 0.50 m. */
export function length(metres) {
  return metres < 0.1 ? `${(metres * 100).toFixed(1)} cm` : `${metres.toFixed(2)} m`
}

/** 'good', 'warn' or '' for a fit residual given in pixels. */
export function fitTonePx(px) {
  if (px == null || !Number.isFinite(px)) return ''
  return px > FIT_PIXELS ? 'warn' : 'good'
}

/** 'good', 'warn' or '' for a fit residual on the ground, in metres. */
export function fitToneM(metres, project) {
  const pixel = groundPixel(project)
  if (metres == null || !pixel) return ''
  return fitTonePx(metres / pixel)
}

/** The fit rule in words, with what it means on the ground for this block. */
export function fitRule(project) {
  const pixel = groundPixel(project)
  return pixel
    ? `Amber above ${FIT_PIXELS} pixels (${length(FIT_PIXELS * pixel)} on the ground at this block's ${length(pixel)} ground pixel).`
    : `Amber above ${FIT_PIXELS} pixels.`
}

/** 'good', 'warn' or '' for an accuracy against the project's target. */
export function accuracyTone(metres, target) {
  if (metres == null || !Number.isFinite(metres) || !(target > 0)) return ''
  return metres <= target ? 'good' : 'warn'
}
