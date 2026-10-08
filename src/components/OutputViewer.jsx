import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { useStore } from '../state/store'
import { api, pathTileUrl } from '../lib/api'

const TILE = 256

/**
 * A generated raster on the canvas: a mosaic, an orthophoto, a DEM.
 *
 * The same tile pyramid as the photo viewer, read straight from the output
 * file, so a full-resolution mosaic pans and zooms like a photograph. Single
 * band elevation is coloured by height; imagery is shown as it is.
 */
export default function OutputViewer({ output, popout = false }) {
  const closeOutput = useStore((s) => s.closeOutput)
  const popOutOutput = useStore((s) => s.popOutOutput)

  const [info, setInfo] = useState(null)
  const [error, setError] = useState(null)
  const containerRef = useRef(null)
  const [size, setSize] = useState({ width: 0, height: 0 })
  const [view, setView] = useState({ scale: 1, x: 0, y: 0 })
  const [panning, setPanning] = useState(null)
  const [enhancement, setEnhancement] = useState('linear2pct')
  const [version] = useState(() => output.version || Date.now())

  useEffect(() => {
    let current = true
    setInfo(null)
    setError(null)
    api.raster.info(output.path)
      .then((result) => { if (current) setInfo(result) })
      .catch((failure) => { if (current) setError(failure.message) })
    return () => { current = false }
  }, [output.path])

  const attach = useCallback((node) => {
    containerRef.current = node
    if (!node) return
    const observer = new ResizeObserver(([entry]) => {
      setSize({ width: entry.contentRect.width, height: entry.contentRect.height })
    })
    observer.observe(node)
  }, [])

  const fit = useCallback(() => {
    if (!info || !size.width) return
    const scale = Math.min(size.width / info.width, size.height / info.height) * 0.96
    setView({ scale, x: (size.width - info.width * scale) / 2, y: (size.height - info.height * scale) / 2 })
  }, [info, size.width, size.height])

  useEffect(() => { fit() }, [info?.width, info?.height, size.width, size.height])

  // Elevation is one band of measurements, not of brightness: colour it.
  const elevation = info && info.bandCount === 1 && !String(info.dtype).startsWith('uint8')
  const colormap = elevation ? 'elevation' : undefined

  const pyramid = useMemo(() => {
    if (!info?.width) return null
    const longest = Math.max(info.width, info.height)
    return { longest, maxZoom: Math.max(0, Math.ceil(Math.log2(longest / TILE))) }
  }, [info?.width, info?.height])

  const zoom = pyramid
    ? Math.min(pyramid.maxZoom, Math.max(0, Math.round(
      pyramid.maxZoom - Math.log2(1 / Math.max(view.scale, 1e-6)))))
    : 0

  function tilesAt(level, margin = 1) {
    if (!pyramid || !size.width) return []
    const srcTile = (pyramid.longest / (TILE * 2 ** level)) * TILE
    const cols = Math.ceil(info.width / srcTile)
    const rows = Math.ceil(info.height / srcTile)
    const left = -view.x / view.scale - srcTile * margin
    const top = -view.y / view.scale - srcTile * margin
    const right = (size.width - view.x) / view.scale + srcTile * margin
    const bottom = (size.height - view.y) / view.scale + srcTile * margin
    const out = []
    for (let ty = 0; ty < rows; ty += 1) {
      for (let tx = 0; tx < cols; tx += 1) {
        const x0 = tx * srcTile
        const y0 = ty * srcTile
        if (x0 + srcTile < left || x0 > right || y0 + srcTile < top || y0 > bottom) continue
        out.push({
          key: `${level}/${tx}/${ty}/${enhancement}`,
          url: pathTileUrl(output.path, level, tx, ty,
            { enhancement, colormap, version: output.version || version }),
          left: x0, top: y0, size: srcTile,
        })
      }
    }
    return out
  }

  const base = zoom > 0 ? tilesAt(0, 0) : []
  const tiles = tilesAt(zoom)

  function onWheel(event) {
    const rect = containerRef.current.getBoundingClientRect()
    const px = event.clientX - rect.left
    const py = event.clientY - rect.top
    const next = Math.min(Math.max(view.scale * Math.exp(-event.deltaY * 0.0016), 0.005), 40)
    const ratio = next / view.scale
    setView({ scale: next, x: px - (px - view.x) * ratio, y: py - (py - view.y) * ratio })
  }

  const name = output.label || output.path.split(/[\\/]/).pop()
  const canPopOut = !popout && !!window.fiducia?.windows

  return (
    <div className="viewer">
      <div className="viewer__toolbar">
        <div className="toolgroup">
          <span className="toolgroup__btn toolgroup__btn--note" title={output.path}>{name}</span>
        </div>
        <div className="toolgroup">
          <button className="toolgroup__btn" onClick={fit} title="Fit the whole output to the view">Fit</button>
          <button
            className="toolgroup__btn"
            onClick={() => setView((v) => {
              const cx = size.width / 2
              const cy = size.height / 2
              return { scale: 1, x: cx - (cx - v.x) / v.scale, y: cy - (cy - v.y) / v.scale }
            })}
            title="Actual pixels (1:1): one output pixel on one screen pixel"
          >
            1:1
          </button>
        </div>
        {!elevation && (
          <div className="toolgroup">
            <select value={enhancement} onChange={(event) => setEnhancement(event.target.value)}
                    title="Display stretch (does not alter the data)">
              <option value="none">No stretch</option>
              <option value="linear2pct">Linear 2%</option>
              <option value="stddev">2 std dev</option>
              <option value="equalize">Equalise</option>
            </select>
          </div>
        )}
        <div className="toolgroup">
          {canPopOut && (
            <button className="toolgroup__btn" onClick={() => popOutOutput(output)}
                    title="Open in a separate window">
              Pop out
            </button>
          )}
          {!popout && (
            <button className="toolgroup__btn" onClick={closeOutput} title="Back to the photographs">
              Close
            </button>
          )}
        </div>
      </div>

      <div
        ref={attach}
        className={`viewer__stage ${panning ? 'viewer__stage--panning' : ''}`}
        onWheel={onWheel}
        onPointerDown={(event) => {
          event.currentTarget.setPointerCapture(event.pointerId)
          setPanning({ x: event.clientX, y: event.clientY, ox: view.x, oy: view.y })
        }}
        onPointerMove={(event) => {
          if (!panning) return
          setView((v) => ({ ...v, x: panning.ox + event.clientX - panning.x,
                                  y: panning.oy + event.clientY - panning.y }))
        }}
        onPointerUp={(event) => {
          event.currentTarget.releasePointerCapture?.(event.pointerId)
          setPanning(null)
        }}
      >
        {error && <div className="viewer__empty"><p>{error}</p></div>}
        {!error && !info && <div className="viewer__empty"><p>Opening {name}…</p></div>}
        {info && (
          <div
            className="viewer__layer"
            style={{
              transform: `translate(${view.x}px, ${view.y}px) scale(${view.scale})`,
              width: info.width,
              height: info.height,
            }}
          >
            {[...base.map((t) => ({ ...t, under: true })), ...tiles].map((tile) => (
              <img
                key={`${tile.under ? 'u' : 't'}:${tile.key}`}
                className={`viewer__tile ${tile.under ? 'viewer__tile--under' : ''}`}
                src={tile.url}
                alt=""
                draggable={false}
                style={{ left: tile.left, top: tile.top, width: tile.size, height: tile.size }}
              />
            ))}
          </div>
        )}
      </div>

      {info && (
        <div className="datastrip">
          <span className="datastrip__cell"><span className="datastrip__value">{name}</span></span>
          <span className="datastrip__cell">
            <span className="datastrip__key">size</span>
            <span className="datastrip__value">{info.width}×{info.height}</span>
          </span>
          <span className="datastrip__cell">
            <span className="datastrip__key">bands</span>
            <span className="datastrip__value">{info.bandCount}</span>
          </span>
          {info.transform && (
            <span className="datastrip__cell">
              <span className="datastrip__key">pixel</span>
              <span className="datastrip__value">{Math.abs(info.transform[0]).toFixed(2)} m</span>
            </span>
          )}
          <span className="datastrip__cell">
            <span className="datastrip__key">zoom</span>
            <span className="datastrip__value">{(view.scale * 100).toFixed(0)}%</span>
          </span>
        </div>
      )}
    </div>
  )
}

/**
 * The mosaic preview, large, while settings are being tuned: the same
 * render the Mosaic panel asks for, given the whole canvas.
 */
export function MosaicPreviewStage() {
  const preview = useStore((s) => s.mosaicPreview)
  const seams = useStore((s) => s.mosaicSeams)
  const setSeams = useStore((s) => s.setMosaicSeams)

  return (
    <div className="viewer">
      <div className="viewer__toolbar">
        <div className="toolgroup">
          <span className="toolgroup__btn toolgroup__btn--note">Mosaic preview</span>
        </div>
        <div className="toolgroup">
          <button className={`toolgroup__btn ${seams ? 'toolgroup__btn--on' : ''}`}
                  onClick={() => setSeams(!seams)} title="Show which image each part comes from">
            Seams
          </button>
        </div>
      </div>
      <div className="viewer__stage mosaic-stage">
        {preview ? (
          <div className="mosaic-stage__frame">
            <img className="mosaic-stage__img" alt="Mosaic preview"
                 src={`data:image/png;base64,${preview.previewPng}`} />
            {seams && (
              <img className="mosaic-stage__img mosaic-stage__seams" alt="Seams"
                   src={`data:image/png;base64,${preview.seamPng}`} />
            )}
          </div>
        ) : (
          <div className="viewer__empty"><p>Rendering preview…</p></div>
        )}
      </div>
      {preview && (
        <div className="datastrip">
          <span className="datastrip__cell">
            <span className="datastrip__key">full output</span>
            <span className="datastrip__value">
              {preview.fullWidth}×{preview.fullHeight} at {preview.pixelSize.toFixed(2)} m
            </span>
          </span>
          <span className="datastrip__cell">
            <span className="datastrip__key">order</span>
            <span className="datastrip__value">{preview.order.join(', ')}</span>
          </span>
          <span className="datastrip__cell">
            <span className="datastrip__key">preview</span>
            <span className="datastrip__value">reduced; generate for full resolution</span>
          </span>
        </div>
      )}
    </div>
  )
}
