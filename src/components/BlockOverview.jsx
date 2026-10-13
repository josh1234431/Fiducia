import { useEffect, useMemo, useState } from 'react'
import { useStore } from '../state/store'
import { api } from '../lib/api'

/**
 * The solved block from above, the default canvas on the Model step.
 *
 * A bundle adjustment is judged by things no single photo shows: how the
 * photos overlap, and whether the control surrounds the block or bunches in
 * one corner. Here every footprint is drawn faintly, so overlap builds up
 * where photos share ground, with the control and check points on top and
 * the tie points behind. Clicking a footprint opens that photo.
 */
export default function BlockOverview() {
  const project = useStore((s) => s.project)
  const setActiveImage = useStore((s) => s.setActiveImage)
  const setBlockView = useStore((s) => s.setBlockView)
  const selectedPoint = useStore((s) => s.selectedPointId)
  const [data, setData] = useState(null)
  const [error, setError] = useState(null)
  const [hover, setHover] = useState(null)

  useEffect(() => {
    let live = true
    api.model.overview()
      .then((result) => { if (live) { setData(result); setError(null) } })
      .catch((e) => { if (live) setError(e.message) })
    return () => { live = false }
  }, [project?.model?.solvedAt, project?.gcps?.length])

  const frame = useMemo(() => {
    if (!data) return null
    const xs = [], ys = []
    for (const f of data.footprints) for (const [x, y] of f.polygon) { xs.push(x); ys.push(y) }
    for (const p of data.points) { xs.push(p.x); ys.push(p.y) }
    if (!xs.length) return null
    const minX = Math.min(...xs), maxX = Math.max(...xs)
    const minY = Math.min(...ys), maxY = Math.max(...ys)
    const span = Math.max(maxX - minX, maxY - minY) || 1
    const pad = span * 0.06
    return {
      minX: minX - pad, maxY: maxY + pad,
      width: maxX - minX + 2 * pad, height: maxY - minY + 2 * pad, span,
    }
  }, [data])

  function open(id) {
    setActiveImage(id)
    setBlockView(false)
  }

  // Ground coordinates to the drawing: x to the right, north up.
  const at = (x, y) => [x - frame.minX, frame.maxY - y]
  const unit = frame ? frame.span / 400 : 1   // a screen-ish pixel, for marks

  const counts = data && {
    photos: data.footprints.length,
    control: data.points.filter((p) => !p.check).length,
    check: data.points.filter((p) => p.check).length,
    ties: data.ties.length,
  }

  return (
    <div className="viewer">
      <div className="viewer__toolbar">
        <div className="toolgroup">
          <span className="toolgroup__btn toolgroup__btn--note">Block overview</span>
        </div>
        {counts && (
          <div className="toolgroup">
            <span className="toolgroup__btn toolgroup__btn--note">
              {counts.photos} photos · {counts.control} control · {counts.check} check
              {counts.ties ? ` · ${counts.ties.toLocaleString()} tie points` : ''}
            </span>
          </div>
        )}
        <div className="toolgroup">
          <button className="toolgroup__btn" onClick={() => setBlockView(false)}
                  title="Show one photo with its points and residuals">
            Photo view
          </button>
        </div>
      </div>

      <div className="viewer__stage overview">
        {error && <div className="viewer__empty"><p>{error}</p></div>}
        {frame && (
          <svg className="overview__map" viewBox={`0 0 ${frame.width} ${frame.height}`}
               preserveAspectRatio="xMidYMid meet">
            {data.ties.map(([x, y], i) => {
              const [u, v] = at(x, y)
              return <circle key={i} cx={u} cy={v} r={unit * 0.9} className="overview__tie" />
            })}
            {data.footprints.map((f) => (
              <polygon
                key={f.id}
                points={f.polygon.map(([x, y]) => at(x, y).join(',')).join(' ')}
                className={`overview__footprint ${hover === f.id ? 'overview__footprint--hover' : ''}`}
                onMouseEnter={() => setHover(f.id)}
                onMouseLeave={() => setHover(null)}
                onClick={() => open(f.id)}
              >
                <title>{f.name}: click to open</title>
              </polygon>
            ))}
            {data.footprints.map((f) => {
              // In the photo's own corner, inside its footprint, clear of the
              // control near the middle.
              const [cx, cy] = at(...f.centre)
              const [u, v] = at(...f.polygon[0])
              return (
                <text key={`${f.id}-name`} x={u + (cx - u) * 0.12} y={v + (cy - v) * 0.12}
                      className="overview__name" fontSize={unit * 7.5}
                      textAnchor="middle" dominantBaseline="middle">
                  {f.name}
                </text>
              )
            })}
            {data.points.map((p) => {
              const [u, v] = at(p.x, p.y)
              const s = unit * (selectedPoint === p.id ? 7 : 4.5)
              return (
                <g key={p.id} className={`overview__point ${p.check ? 'marker--check' : 'marker--gcp'}`}>
                  <title>
                    {p.id} {p.check ? 'check' : 'control'}
                    {p.errorM != null ? `: ${p.errorM.toFixed(2)} m horizontal error` : ''}
                  </title>
                  <path d={`M${u - s},${v} H${u + s} M${u},${v - s} V${v + s}`} strokeWidth={unit * 1.3} />
                  <circle cx={u} cy={v} r={s * 0.55} strokeWidth={unit * 1.1} fill="none" />
                  <text x={u + s * 1.3} y={v - s * 0.5} fontSize={unit * 6}>{p.id}</text>
                </g>
              )
            })}
          </svg>
        )}
      </div>

      <div className="viewer__legend">
        <span className="viewer__legend-item"><span className="overview__swatch" />Photo footprint</span>
        <span className="viewer__legend-item">
          <span className="viewer__legend-mark marker--gcp"><span className="marker__cross" /><span className="marker__ring" /></span>
          Control
        </span>
        <span className="viewer__legend-item">
          <span className="viewer__legend-mark marker--check"><span className="marker__cross" /><span className="marker__ring" /></span>
          Check
        </span>
        <span className="viewer__legend-item"><span className="overview__dot" />Tie point</span>
      </div>
    </div>
  )
}
