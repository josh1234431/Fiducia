import { useEffect, useMemo, useRef, useState } from 'react'
import { useStore } from '../state/store'
import { api } from '../lib/api'

/**
 * A point cloud seen from above and from the side.
 *
 * Above: the cloud's highest points, by height or class. Drag across it to
 * draw a section. Beside: every point in that corridor, seen side on, where
 * points can be labelled with the class chosen in the LiDAR step — drag a box
 * over them, Shift-drag to clear. Labels are saved with the project as they
 * are made, and train the classifier in the LiDAR step.
 */
export default function PointCloudView() {
  const path = useStore((s) => s.lidarCloud)
  const section = useStore((s) => s.lidarSection)
  const setSection = useStore((s) => s.setLidarSection)
  const brush = useStore((s) => s.lidarBrush)
  const bumpLabels = useStore((s) => s.bumpLidarLabels)
  const reference = useStore((s) => s.reference)
  const project = useStore((s) => s.project)
  const toast = useStore((s) => s.toast)

  const colour = useStore((s) => s.lidarColour)
  const setColour = useStore((s) => s.setLidarColour)
  const [overview, setOverview] = useState(null)
  const [cut, setCut] = useState(null)
  const [labels, setLabels] = useState({})
  const [draft, setDraft] = useState(null)

  const colours = reference?.lidarClassColours || {}
  const spots = project?.lidarLearning?.uncertainSpots || []

  const [waiting, setWaiting] = useState(0)
  useEffect(() => {
    let live = true
    let timer = null
    setOverview(null)
    api.lidar.overview(path, colour === 'height' ? 'height' : 'class')
      .then((result) => live && setOverview(result))
      .catch((error) => {
        // Still being indexed: look again shortly.
        if (/indexed/i.test(error.message || '')) {
          timer = setTimeout(() => live && setWaiting((w) => w + 1), 2000)
        } else {
          toast(error.message, 'bad')
        }
      })
    return () => { live = false; clearTimeout(timer) }
  }, [path, colour, waiting]) // eslint-disable-line react-hooks/exhaustive-deps

  useEffect(() => {
    if (!section) { setCut(null); return }
    let live = true
    api.lidar.section(path, section.start, section.end, section.width)
      .then((result) => {
        if (!live) return
        setCut(result)
        setLabels(result.labels || {})
      })
      .catch((error) => toast(error.message, 'bad'))
    return () => { live = false }
  }, [path, section]) // eslint-disable-line react-hooks/exhaustive-deps

  // -- plan view -------------------------------------------------------------

  const planRef = useRef(null)

  function toMap(event) {
    const box = planRef.current.getBoundingClientRect()
    const [west, south, east, north] = overview.bounds
    const scale = Math.min(box.width / (east - west), box.height / (north - south))
    const offsetX = (box.width - (east - west) * scale) / 2
    const offsetY = (box.height - (north - south) * scale) / 2
    return [
      west + (event.clientX - box.left - offsetX) / scale,
      north - (event.clientY - box.top - offsetY) / scale,
    ]
  }

  function toScreen([x, y]) {
    const box = planRef.current?.getBoundingClientRect()
    if (!box || !overview) return [0, 0]
    const [west, south, east, north] = overview.bounds
    const scale = Math.min(box.width / (east - west), box.height / (north - south))
    const offsetX = (box.width - (east - west) * scale) / 2
    const offsetY = (box.height - (north - south) * scale) / 2
    return [offsetX + (x - west) * scale, offsetY + (north - y) * scale]
  }

  function onPlanDown(event) {
    if (!overview) return
    event.currentTarget.setPointerCapture(event.pointerId)
    const point = toMap(event)
    setDraft({ start: point, end: point })
  }

  function onPlanMove(event) {
    if (draft) setDraft({ ...draft, end: toMap(event) })
  }

  function onPlanUp() {
    if (!draft) return
    const length = Math.hypot(draft.end[0] - draft.start[0], draft.end[1] - draft.start[1])
    if (length > (overview.cellSize || 1) * 3) {
      setSection({ start: draft.start, end: draft.end, width: section?.width || 2 })
    }
    setDraft(null)
  }

  function showSpot(spot) {
    const half = 15
    setSection({ start: [spot.x - half, spot.y], end: [spot.x + half, spot.y], width: section?.width || 2 })
  }

  const line = draft || section

  // -- section view ------------------------------------------------------------

  const canvasRef = useRef(null)
  const wrapRef = useRef(null)
  const [view, setView] = useState(null)      // { x0, x1, z0, z1 }
  const [box, setBox] = useState(null)        // label rectangle being dragged
  const [size, setSize] = useState({ width: 0, height: 0 })

  useEffect(() => {
    const element = wrapRef.current
    if (!element) return undefined
    const observer = new ResizeObserver(([entry]) => {
      setSize({ width: entry.contentRect.width, height: entry.contentRect.height })
    })
    observer.observe(element)
    return () => observer.disconnect()
  }, [cut !== null]) // eslint-disable-line react-hooks/exhaustive-deps

  const extent = useMemo(() => {
    if (!cut?.z?.length) return null
    // Fit to the real points: known noise and the odd extreme stray (a bird
    // far above) would otherwise squash the section flat. Scrolling out still
    // shows them.
    const real = cut.z.filter((_, i) => cut.classification[i] !== 7 && cut.classification[i] !== 18)
    const sorted = (real.length ? real : cut.z).slice().sort((a, b) => a - b)
    const low = sorted[Math.floor((sorted.length - 1) * 0.002)]
    const high = sorted[Math.ceil((sorted.length - 1) * 0.998)]
    const pad = Math.max((high - low) * 0.06, 0.5)
    return { x0: 0, x1: cut.length, z0: low - pad, z1: high + pad }
  }, [cut])

  useEffect(() => { setView(extent) }, [extent])

  const zRange = useMemo(() => (extent ? [extent.z0, extent.z1] : [0, 1]), [extent])

  function project2d(along, z) {
    return [
      ((along - view.x0) / (view.x1 - view.x0)) * size.width,
      size.height - ((z - view.z0) / (view.z1 - view.z0)) * size.height,
    ]
  }

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas || !cut || !view || !size.width) return
    const ratio = window.devicePixelRatio || 1
    canvas.width = Math.round(size.width * ratio)
    canvas.height = Math.round(size.height * ratio)
    const context = canvas.getContext('2d')
    context.setTransform(ratio, 0, 0, ratio, 0, 0)
    context.clearRect(0, 0, size.width, size.height)
    const [zLow, zHigh] = zRange
    const pixel = Math.max(1.6, Math.min(3, size.width / Math.max(cut.shown, 1) * 40))
    const ramp = (t) => {
      const stops = [[24, 40, 92], [23, 145, 127], [111, 208, 112], [240, 200, 80], [250, 250, 245]]
      const p = Math.min(Math.max(t, 0), 1) * (stops.length - 1)
      const i = Math.min(Math.floor(p), stops.length - 2)
      const f = p - i
      return `rgb(${stops[i].map((v, k) => Math.round(v * (1 - f) + stops[i + 1][k] * f)).join(',')})`
    }
    const labelled = []
    for (let i = 0; i < cut.shown; i += 1) {
      const [sx, sy] = project2d(cut.along[i], cut.z[i])
      if (sx < -2 || sx > size.width + 2 || sy < -2 || sy > size.height + 2) continue
      const label = labels[String(cut.index[i])]
      if (label !== undefined) { labelled.push([sx, sy, label]); continue }
      context.fillStyle = colour === 'height'
        ? ramp((cut.z[i] - zLow) / (zHigh - zLow))
        : (colours[String(cut.classification[i])] || '#b6bec4')
      context.globalAlpha = colour === 'labels' ? 0.25 : 0.9
      context.fillRect(sx - pixel / 2, sy - pixel / 2, pixel, pixel)
    }
    context.globalAlpha = 1
    // Labelled points on top, larger and ringed, so they read at a glance.
    for (const [sx, sy, label] of labelled) {
      context.fillStyle = colours[String(label)] || '#ffffff'
      context.strokeStyle = 'rgba(255,255,255,0.85)'
      context.lineWidth = 1
      context.beginPath()
      context.arc(sx, sy, pixel + 0.6, 0, Math.PI * 2)
      context.fill()
      context.stroke()
    }
  }, [cut, view, size, labels, colour, colours, zRange]) // eslint-disable-line react-hooks/exhaustive-deps

  function sectionPoint(event) {
    const rect = canvasRef.current.getBoundingClientRect()
    const sx = event.clientX - rect.left
    const sy = event.clientY - rect.top
    return {
      sx, sy,
      along: view.x0 + (sx / size.width) * (view.x1 - view.x0),
      z: view.z0 + ((size.height - sy) / size.height) * (view.z1 - view.z0),
    }
  }

  function onWheel(event) {
    if (!view) return
    const p = sectionPoint(event)
    const factor = event.deltaY > 0 ? 1.15 : 1 / 1.15
    const next = { ...view }
    if (!event.ctrlKey) {
      next.x0 = p.along - (p.along - view.x0) * factor
      next.x1 = p.along + (view.x1 - p.along) * factor
    }
    if (!event.shiftKey) {
      next.z0 = p.z - (p.z - view.z0) * factor
      next.z1 = p.z + (view.z1 - p.z) * factor
    }
    setView(next)
  }

  function onSectionDown(event) {
    if (!view) return
    event.currentTarget.setPointerCapture(event.pointerId)
    const p = sectionPoint(event)
    const labelling = brush !== null && event.button === 0
    setBox({ from: p, to: p, mode: labelling ? (event.shiftKey ? 'clear' : 'label') : 'pan', view })
  }

  function onSectionMove(event) {
    if (!box) return
    const p = sectionPoint(event)
    if (box.mode === 'pan') {
      const dx = ((p.sx - box.from.sx) / size.width) * (box.view.x1 - box.view.x0)
      const dz = ((p.sy - box.from.sy) / size.height) * (box.view.z1 - box.view.z0)
      setView({ x0: box.view.x0 - dx, x1: box.view.x1 - dx, z0: box.view.z0 + dz, z1: box.view.z1 + dz })
      return
    }
    setBox({ ...box, to: p })
  }

  async function onSectionUp() {
    if (!box) return
    const current = box
    setBox(null)
    if (current.mode === 'pan') return
    const a0 = Math.min(current.from.along, current.to.along)
    const a1 = Math.max(current.from.along, current.to.along)
    const z0 = Math.min(current.from.z, current.to.z)
    const z1 = Math.max(current.from.z, current.to.z)
    const chosen = []
    for (let i = 0; i < cut.shown; i += 1) {
      if (cut.along[i] >= a0 && cut.along[i] <= a1 && cut.z[i] >= z0 && cut.z[i] <= z1) {
        chosen.push(cut.index[i])
      }
    }
    if (!chosen.length) return
    const next = { ...labels }
    if (current.mode === 'clear') {
      chosen.forEach((index) => { delete next[String(index)] })
    } else {
      chosen.forEach((index) => { next[String(index)] = brush })
    }
    setLabels(next)
    try {
      await api.lidar.setLabels(current.mode === 'clear'
        ? { path, clear: chosen }
        : { path, set: Object.fromEntries(chosen.map((index) => [index, brush])) })
      bumpLabels()
    } catch (error) {
      toast(error.message, 'bad')
    }
  }

  const boxStyle = box && box.mode !== 'pan' ? {
    left: Math.min(box.from.sx, box.to.sx),
    top: Math.min(box.from.sy, box.to.sy),
    width: Math.abs(box.to.sx - box.from.sx),
    height: Math.abs(box.to.sy - box.from.sy),
  } : null

  const [x1, y1] = line && overview ? toScreen(line.start) : [0, 0]
  const [x2, y2] = line && overview ? toScreen(line.end) : [0, 0]
  const brushName = brush !== null
    ? (reference?.lidarClasses?.[String(brush)] || `Class ${brush}`) : null

  return (
    <div className="cloudview">
      <div className="viewer__toolbar">
        <div className="toolgroup">
          {[['class', 'Class'], ['height', 'Height'], ['labels', 'Labels']].map(([key, name]) => (
            <button
              key={key}
              className={`toolgroup__btn ${colour === key ? 'toolgroup__btn--on' : ''}`}
              aria-pressed={colour === key}
              onClick={() => setColour(key)}
              title={key === 'labels' ? 'Fade everything but labelled points' : `Colour points by ${name.toLowerCase()}`}
            >
              {name}
            </button>
          ))}
        </div>
        {section && (
          <div className="toolgroup">
            <span className="toolgroup__label">Width</span>
            {[1, 2, 5, 10].map((width) => (
              <button
                key={width}
                className={`toolgroup__btn ${section.width === width ? 'toolgroup__btn--on' : ''}`}
                onClick={() => setSection({ ...section, width })}
                title={`Show the points within ${width / 2} m of the line`}
              >
                {width} m
              </button>
            ))}
          </div>
        )}
        {brushName && (
          <div className="toolgroup cloudview__brush">
            <i style={{ background: colours[String(brush)] || '#fff' }} />
            Labelling: {brushName}
          </div>
        )}
      </div>

      <div className="cloudview__plan" ref={planRef}
        onPointerDown={onPlanDown} onPointerMove={onPlanMove} onPointerUp={onPlanUp}>
        {overview
          ? <img src={overview.image} alt="The point cloud from above" draggable={false} />
          : <div className="cloudview__empty">{waiting ? 'Indexing the point cloud…' : 'Drawing the point cloud…'}</div>}
        {overview && (
          <svg className="cloudview__overlay">
            {line && (
              <>
                <line x1={x1} y1={y1} x2={x2} y2={y2} className="cloudview__line" />
                <circle cx={x1} cy={y1} r="4" className="cloudview__end" />
                <circle cx={x2} cy={y2} r="4" className="cloudview__end" />
              </>
            )}
            {spots.map((spot, index) => {
              const [sx, sy] = toScreen([spot.x, spot.y])
              return (
                <g key={index} className="cloudview__spot" onPointerDown={(event) => {
                  event.stopPropagation()
                  showSpot(spot)
                }}>
                  <circle cx={sx} cy={sy} r="7" />
                  <text x={sx} y={sy + 3.5}>?</text>
                </g>
              )
            })}
          </svg>
        )}
        {!section && overview && (
          <div className="cloudview__hint">Drag across the cloud to draw a section</div>
        )}
      </div>

      <div className="cloudview__section" ref={wrapRef}>
        {cut ? (
          <>
            <canvas
              ref={canvasRef}
              style={{ cursor: brush !== null ? 'crosshair' : 'grab' }}
              onWheel={onWheel}
              onPointerDown={onSectionDown}
              onPointerMove={onSectionMove}
              onPointerUp={onSectionUp}
              onDoubleClick={() => setView(extent)}
            />
            {boxStyle && (
              <div className={`cloudview__box ${box.mode === 'clear' ? 'cloudview__box--clear' : ''}`}
                style={{ ...boxStyle, borderColor: box.mode === 'clear' ? undefined : colours[String(brush)] }} />
            )}
            <div className="cloudview__status">
              {cut.total.toLocaleString()} points in {cut.length.toFixed(0)} m
              {cut.shown < cut.total ? ` (${cut.shown.toLocaleString()} shown)` : ''}
              {' · '}scroll to zoom, Shift or Ctrl for one axis, double-click to fit
              {brush !== null ? ' · drag a box to label, Shift-drag to clear' : ' · drag to pan'}
            </div>
          </>
        ) : (
          <div className="cloudview__empty">
            {section ? 'Reading the section…' : 'The section appears here, seen from the side.'}
          </div>
        )}
      </div>
    </div>
  )
}
