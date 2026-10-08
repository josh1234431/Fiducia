import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { useStore, selectActiveImage, selectObservationsFor, workflowOf } from '../state/store'
import { api, tileUrl } from '../lib/api'
import {
  cssMatrix, displaySize, flip, normalise, rotateBy, toDisplay, toImage, turnVector,
} from '../lib/orientation'

/**
 * Image viewer — the persistent image canvas.
 *
 * One canvas, always on screen, that every processing step draws onto. On a
 * legacy workstation each task opens its own viewer window, so measuring fiducials
 * then collecting GCPs means closing and reopening the same photo, losing your
 * position each time. Here the photo stays put and the step changes what is
 * overlaid on it.
 *
 * Rendering is tile-based over a pyramid of overviews, so a 12000-pixel scan
 * pans and zooms at the same cost as a thumbnail. Only tiles inside the
 * viewport are requested, and the browser's own image cache does the rest.
 */

const TILE = 256

/** Each open view's centre, by photo id: { col, row, scale }. */
export const viewCentres = new Map()

export default function ImageViewer({ imageId: fixedId, onImageChange, popout = false } = {}) {
  const activeImage = useStore(selectActiveImage)
  const activeStep = useStore((s) => s.activeStep)
  const project = useStore((s) => s.project)
  const setActiveImage = useStore((s) => s.setActiveImage)
  const pickMode = useStore((s) => s.pickMode)
  const clipDraft = useStore((s) => s.clipDraft)
  const pendingPoint = useStore((s) => s.pendingPoint)
  const loupeEnabled = useStore((s) => s.loupe)
  const popOut = useStore((s) => s.popOut)
  const poppedOut = useStore((s) => s.poppedOut)

  // A popped-out window shows the image it was opened on, not whichever image
  // happens to be active in the main window.
  const image = fixedId
    ? project?.images?.find((entry) => entry.id === fixedId) || null
    : activeImage
  const chooseImage = onImageChange || setActiveImage
  const call = useStore((s) => s.call)

  // How the photo is turned on screen. Display only: measurements are always
  // stored in the scan's own pixel coordinates.
  //
  // Measurements put whole numbers at pixel centres, as the engine does:
  // (0, 0) is the middle of the first pixel. The display layer has its edge
  // at 0, so the centre of pixel (0, 0) is drawn at (0.5, 0.5). The half
  // pixel is added here and removed in toImagePixel, and nowhere else.
  const orientation = normalise(image?.orientation)
  const shown = displaySize(image?.width || 0, image?.height || 0, orientation)
  const place = (col, row) => toDisplay(col + 0.5, row + 0.5, image.width, image.height, orientation)
  const canPopOut = !popout && !!window.fiducia?.windows

  const containerRef = useRef(null)
  const [size, setSize] = useState({ width: 0, height: 0 })
  const [view, setView] = useState({ scale: 1, x: 0, y: 0 })
  const [panning, setPanning] = useState(false)
  const [cursor, setCursor] = useState(null)
  const [pointer, setPointer] = useState(null)   // cursor in stage pixels, for the magnifier
  const [enhancement, setEnhancement] = useState('linear2pct')
  const [showOverlays, setShowOverlays] = useState(true)
  const [fitToken, setFitToken] = useState(0)

  const observations = useStore((s) =>
    image ? selectObservationsFor(s, image.id) : [])
  const gcps = project?.gcps || []
  const camera = project?.camera || {}
  const isFilm = (project?.mathModel?.kind || 'aerial_film') === 'aerial_film'

  // -- sizing -----------------------------------------------------------

  // A callback ref rather than an effect on mount: the stage element only
  // exists once an image is open, so an effect with an empty dependency list
  // would run while the ref is still null and never attach the observer —
  // leaving the viewer permanently convinced it has zero size, and drawing
  // no tiles at all.
  const observerRef = useRef(null)

  const attachStage = useCallback((node) => {
    containerRef.current = node
    observerRef.current?.disconnect()
    if (!node) return

    setSize({ width: node.clientWidth, height: node.clientHeight })
    const observer = new ResizeObserver(([entry]) => {
      setSize({ width: entry.contentRect.width, height: entry.contentRect.height })
    })
    observer.observe(node)
    observerRef.current = observer
  }, [])

  useEffect(() => () => observerRef.current?.disconnect(), [])

  const fit = useCallback(() => {
    if (!shown.width || !size.width) return
    const scale = Math.min(size.width / shown.width, size.height / shown.height) * 0.92
    setView({
      scale,
      x: (size.width - shown.width * scale) / 2,
      y: (size.height - shown.height * scale) / 2,
    })
  }, [shown.width, shown.height, size.width, size.height])

  // Not on orientation: turning the photo keeps the view where it is (see
  // turn below) rather than zooming back out to the whole frame.
  useEffect(() => { fit() }, [image?.id, size.width, size.height, fitToken])

  // Where this view is centred, for lining another view up with it. Kept
  // outside React state: it changes with every pan, and nothing redraws on it.
  useEffect(() => {
    if (!image || !size.width || !view.scale) return
    const centre = toImage((size.width / 2 - view.x) / view.scale,
      (size.height / 2 - view.y) / view.scale, image.width, image.height, orientation)
    viewCentres.set(image.id, { col: centre.col - 0.5, row: centre.row - 0.5, scale: view.scale })
  }, [view, size.width, size.height, image?.id, orientation.rotate, orientation.flipX])

  // A request to centre this photo on a pixel, from the Line up button, in
  // this window or relayed from the main one to a popped-out window.
  const viewRequest = useStore((s) => s.viewRequest)
  const appliedRequest = useRef(null)
  useEffect(() => {
    if (!viewRequest || !image || viewRequest.imageId !== image.id || !size.width) return
    if (appliedRequest.current === viewRequest.token) return
    appliedRequest.current = viewRequest.token
    const scale = Math.min(Math.max(viewRequest.scale || view.scale, 0.01), 40)
    const at = place(viewRequest.col, viewRequest.row)
    setView({ scale, x: size.width / 2 - at.u * scale, y: size.height / 2 - at.v * scale })
  }, [viewRequest?.token, image?.id, size.width, size.height])

  // -- interaction ------------------------------------------------------

  const toImagePixel = useCallback(
    (clientX, clientY) => {
      const rect = containerRef.current?.getBoundingClientRect()
      if (!rect || !image) return null
      const edge = toImage(
        (clientX - rect.left - view.x) / view.scale,
        (clientY - rect.top - view.y) / view.scale,
        image.width, image.height, orientation,
      )
      return { col: edge.col - 0.5, row: edge.row - 0.5 }
    },
    [view, image?.width, image?.height, orientation.rotate, orientation.flipX],
  )

  /**
   * Turn the photo, keeping whatever is in the middle of the view there.
   *
   * Takes a change rather than a result, and applies it to the latest stored
   * orientation and the latest view: two clicks inside one frame would
   * otherwise both start from what was on screen, and count as one.
   */
  function turn(change) {
    if (!image) return
    const { id, width: w, height: h } = image
    const latest = useStore.getState().project?.images?.find((entry) => entry.id === id)
    const current = normalise(latest?.orientation)
    const next = change(current)

    setView((v) => {
      const centre = toImage((size.width / 2 - v.x) / v.scale, (size.height / 2 - v.y) / v.scale,
        w, h, current)
      const moved = toDisplay(centre.col, centre.row, w, h, next)
      return { ...v, x: size.width / 2 - moved.u * v.scale, y: size.height / 2 - moved.v * v.scale }
    })

    // Shown at once, saved behind it.
    useStore.setState((state) => ({
      project: state.project && {
        ...state.project,
        images: state.project.images.map((entry) =>
          entry.id === id ? { ...entry, orientation: next } : entry),
      },
    }))
    call(() => api.images.patch(id, { orientation: next }), { refresh: false })
  }

  const onWheel = useCallback(
    (event) => {
      event.preventDefault()
      const rect = containerRef.current.getBoundingClientRect()
      const px = event.clientX - rect.left
      const py = event.clientY - rect.top

      // Zoom about the cursor, so the feature under the pointer stays put.
      const factor = Math.exp(-event.deltaY * 0.0016)
      const next = Math.min(Math.max(view.scale * factor, 0.01), 40)
      const ratio = next / view.scale

      setView({
        scale: next,
        x: px - (px - view.x) * ratio,
        y: py - (py - view.y) * ratio,
      })
    },
    [view],
  )

  const onPointerDown = useCallback(
    (event) => {
      if (event.button === 1 || event.button === 0) {
        // Middle-drag always pans. Left-drag pans unless a pick is armed.
        if (event.button === 1 || !pickMode) {
          event.currentTarget.setPointerCapture(event.pointerId)
          setPanning({ x: event.clientX, y: event.clientY, ox: view.x, oy: view.y })
        }
      }
    },
    [view, pickMode],
  )

  const onPointerMove = useCallback(
    (event) => {
      if (panning) {
        setView((v) => ({
          ...v,
          x: panning.ox + (event.clientX - panning.x),
          y: panning.oy + (event.clientY - panning.y),
        }))
      }
      const pixel = toImagePixel(event.clientX, event.clientY)
      if (pixel) setCursor(pixel)
      const rect = containerRef.current?.getBoundingClientRect()
      if (rect) setPointer({ x: event.clientX - rect.left, y: event.clientY - rect.top })
    },
    [panning, toImagePixel],
  )

  const onPointerUp = useCallback((event) => {
    if (panning) {
      event.currentTarget.releasePointerCapture?.(event.pointerId)
      setPanning(false)
    }
  }, [panning])

  const onClick = useCallback(
    (event) => {
      if (!pickMode || panning) return
      const pixel = toImagePixel(event.clientX, event.clientY)
      if (pixel && image) pickMode.onPick(pixel.col, pixel.row, image)
    },
    [pickMode, panning, toImagePixel, image],
  )

  // -- tiles ------------------------------------------------------------
  //
  // Zooming changes pyramid level, and a new level's tiles are empty squares
  // until they arrive. Three layers keep that from ever showing as black:
  //
  //   base      one coarse tile of the whole photo, always underneath
  //   fallback  the last level that finished loading, kept until the new
  //             one has -- already in the browser's cache, so it is free
  //   current   the level for this zoom, each tile fading in once loaded
  //
  // Fast machines also prefetch the neighbouring levels once the view is
  // still, so a zoom usually lands on tiles that are already there. Slow
  // machines skip that and rely on the layers above, which cost nothing extra.

  const pyramid = useMemo(() => {
    if (!image?.width) return null
    const longest = Math.max(image.width, image.height)
    return { longest, maxZoom: Math.max(0, Math.ceil(Math.log2(longest / TILE))) }
  }, [image?.width, image?.height])

  // Choose the pyramid level whose native resolution is closest to the
  // scale we are drawing at, so we never download more pixels than the
  // screen can show.
  const zoom = pyramid
    ? Math.min(pyramid.maxZoom, Math.max(0, Math.round(
      pyramid.maxZoom - Math.log2(1 / Math.max(view.scale, 1e-6)))))
    : 0

  const tilesAt = useCallback((level, marginTiles = 1) => {
    if (!pyramid || !size.width || level < 0 || level > pyramid.maxZoom) return []
    const srcTile = (pyramid.longest / (TILE * 2 ** level)) * TILE
    const cols = Math.ceil(image.width / srcTile)
    const rows = Math.ceil(image.height / srcTile)

    // The screen's corners back on the scan, so culling works however the
    // photo is turned, plus a margin so panning does not reveal blanks.
    const margin = srcTile * marginTiles
    const corners = [
      [-view.x, -view.y], [size.width - view.x, -view.y],
      [-view.x, size.height - view.y], [size.width - view.x, size.height - view.y],
    ].map(([u, v]) => toImage(u / view.scale, v / view.scale, image.width, image.height, orientation))
    const left = Math.min(...corners.map((c) => c.col)) - margin
    const top = Math.min(...corners.map((c) => c.row)) - margin
    const right = Math.max(...corners.map((c) => c.col)) + margin
    const bottom = Math.max(...corners.map((c) => c.row)) + margin

    const out = []
    for (let ty = 0; ty < rows; ty += 1) {
      for (let tx = 0; tx < cols; tx += 1) {
        const x0 = tx * srcTile
        const y0 = ty * srcTile
        if (x0 + srcTile < left || x0 > right || y0 + srcTile < top || y0 > bottom) continue
        out.push({
          key: `${level}/${tx}/${ty}/${enhancement}`,
          url: tileUrl(image.id, level, tx, ty, { enhancement }),
          left: x0,
          top: y0,
          size: srcTile,
        })
      }
    }
    return out
  }, [pyramid, image?.id, image?.width, image?.height, view, size, enhancement,
      orientation.rotate, orientation.flipX])

  const tiles = useMemo(() => tilesAt(zoom), [tilesAt, zoom])
  const base = useMemo(() => (zoom > 0 ? tilesAt(0, 0) : []), [tilesAt, zoom])

  // Which tiles have arrived, and how long tiles take -- the second decides
  // whether this machine prefetches.
  const loadedRef = useRef(new Set())
  const requestedAt = useRef(new Map())
  const tileMs = useRef(null)
  const [, setLoadedCount] = useState(0)
  const [settled, setSettled] = useState(null)   // last level fully loaded

  // A tile that fails to load is retried a few times with a growing delay,
  // under a fresh URL so the browser does not reuse the failed response.
  // Without this, one failed request left that part of the view blurry.
  const [retries, setRetries] = useState({})
  const srcFor = useCallback(
    (tile) => (retries[tile.key] ? `${tile.url}&retry=${retries[tile.key]}` : tile.url),
    [retries],
  )
  const onTileError = useCallback((key) => {
    setRetries((current) => {
      const count = current[key] || 0
      if (count >= 3) return current
      setTimeout(() => setRetries((r) => ({ ...r, [key]: count + 1 })), 400 * 2 ** count)
      return current
    })
  }, [])

  useEffect(() => {
    loadedRef.current = new Set()
    requestedAt.current = new Map()
    setSettled(null)
    setRetries({})
  }, [image?.id, enhancement])

  const now = typeof performance !== 'undefined' ? performance.now() : Date.now()
  for (const tile of tiles) {
    if (!requestedAt.current.has(tile.key)) requestedAt.current.set(tile.key, now)
  }

  const onTileLoad = useCallback((key) => {
    if (loadedRef.current.has(key)) return
    loadedRef.current.add(key)
    const started = requestedAt.current.get(key)
    if (started != null) {
      const took = performance.now() - started
      tileMs.current = tileMs.current == null ? took : tileMs.current * 0.8 + took * 0.2
    }
    setLoadedCount((n) => n + 1)
  }, [])

  const allLoaded = tiles.length > 0 && tiles.every((tile) => loadedRef.current.has(tile.key))
  useEffect(() => {
    if (allLoaded && settled !== zoom) setSettled(zoom)
  }, [allLoaded, zoom, settled])

  // Only tiles that have already arrived. After zooming out from 1:1 the
  // previous level covers the whole new view in full-resolution tiles --
  // thousands of them for a large scan -- and requesting those queued them
  // ahead of the tiles the view actually needs, so it never sharpened.
  const fallback = useMemo(
    () => (settled != null && settled !== zoom
      ? tilesAt(settled, 0).filter((tile) => loadedRef.current.has(tile.key))
      : []),
    [tilesAt, settled, zoom],
  )

  // Prefetch only where it will not hurt: plenty of cores and memory, and
  // tiles already arriving quickly. navigator.deviceMemory is Chromium's,
  // which is what Fiducia runs in.
  const fastMachine = useMemo(() => {
    const cores = navigator.hardwareConcurrency || 4
    const memory = navigator.deviceMemory ?? 8
    return cores >= 8 && memory >= 8
  }, [])

  useEffect(() => {
    if (!fastMachine || !allLoaded) return undefined
    if (tileMs.current != null && tileMs.current > 150) return undefined
    // Prefetches are abandoned as soon as the view moves, so they can never
    // queue ahead of the tiles the new view actually needs.
    const probes = []
    const timer = setTimeout(() => {
      const next = [...tilesAt(zoom - 1, 0), ...tilesAt(zoom + 1, 0)]
        .filter((tile) => !loadedRef.current.has(tile.key))
        .slice(0, 24)
      for (const tile of next) {
        const probe = new Image()
        probe.decoding = 'async'
        probe.src = tile.url
        probes.push(probe)
      }
    }, 180)
    return () => {
      clearTimeout(timer)
      for (const probe of probes) {
        if (!probe.complete) probe.src = ''
      }
    }
  }, [fastMachine, allLoaded, tilesAt, zoom])

  // -- overlays ---------------------------------------------------------

  const markers = useMemo(() => {
    if (!image || !showOverlays) return []
    const out = []

    // Fiducial marks, while working interior orientation.
    if (isFilm && (activeStep === 'images' || activeStep === 'camera')) {
      Object.entries(image.fiducials || {}).forEach(([slot, [col, row]]) => {
        out.push({
          id: `fid-${slot}`, col, row, kind: 'fiducial',
          label: slot.replace(/_/g, ' '),
        })
      })
    }

    // Control, check and tie points.
    if (activeStep === 'points' || activeStep === 'model' || activeStep === 'ortho') {
      observations.forEach((observation) => {
        const gcp = gcps.find((g) => g.id === observation.pointId)
        const kind = gcp ? (gcp.isCheckPoint ? 'check' : 'gcp') : 'tie'
        out.push({
          id: `${observation.pointId}`,
          col: observation.col,
          row: observation.row,
          kind,
          label: observation.pointId,
        })
      })
    }

    // A clicked point waiting to be accepted, drawn where it will go.
    if (pendingPoint && pendingPoint.imageId === image.id) {
      out.push({
        id: '__pending',
        col: pendingPoint.col,
        row: pendingPoint.row,
        kind: 'pending',
        label: `${pendingPoint.id || 'New point'} (not saved)`,
      })
    }

    return out
  }, [image, showOverlays, activeStep, observations, gcps, isFilm, pendingPoint])

  const selectedPoint = useStore((s) => s.selectedPointId)

  // -- residual vectors -------------------------------------------------
  //
  // The discipline's own diagnostic. A residual is a direction and a
  // magnitude, and at true scale it is invisible — a good one is microns on
  // a frame measured in centimetres. So the vectors are exaggerated by a
  // stated factor, which is exactly how a printed residual plot works.

  const [showResiduals, setShowResiduals] = useState(true)

  const residualPlot = useMemo(() => {
    if (!image || !project?.model?.residuals || !showOverlays || !showResiduals) return null
    if (activeStep !== 'model' && activeStep !== 'points') return null

    // Millimetres on the focal plane to pixels on this scan.
    let mmPerPixel = camera.pixelPitchMm || 0
    if (!mmPerPixel && image.fiducialFit) {
      const { ax, ay } = image.fiducialFit
      mmPerPixel = Math.sqrt(Math.abs(ax[1] * ay[2] - ax[2] * ay[1])) || 0
    }
    if (!mmPerPixel) return null

    const vectors = []
    for (const observation of observations) {
      const entry = project.model.residuals[`${image.id}|${observation.pointId}`]
      if (!entry) continue
      const dxPx = entry[0] / mmPerPixel
      const dyPx = -entry[1] / mmPerPixel   // film y is up, row index is down
      const magnitude = Math.hypot(dxPx, dyPx)
      vectors.push({
        id: observation.pointId,
        col: observation.col,
        row: observation.row,
        dx: dxPx,
        dy: dyPx,
        magnitude,
      })
    }
    if (!vectors.length) return null

    // Pick an exaggeration that makes the largest vector about 46 screen
    // pixels, rounded to a readable factor.
    const worst = Math.max(...vectors.map((v) => v.magnitude)) || 1e-6
    const raw = 46 / (worst * view.scale)
    const nice = [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000]
    const factor = nice.reduce((best, n) => (Math.abs(n - raw) < Math.abs(best - raw) ? n : best), 1)
    const median = vectors.map((v) => v.magnitude).sort((a, b) => a - b)[
      Math.floor(vectors.length / 2)
    ]

    return { vectors, factor, worst, median, mmPerPixel }
  }, [image, project?.model, observations, showOverlays, showResiduals, activeStep,
      camera.pixelPitchMm, view.scale])

  // -- render -----------------------------------------------------------

  if (!image) {
    const images = project?.images || []
    return (
      <div className="viewer">
        <div className="viewer__empty">
          {images.length === 0 ? (
            <>
              <h2>No imagery loaded</h2>
              {workflowOf(project).id === 'drone' ? (
                <p className="dim" style={{ maxWidth: '40ch', margin: 0 }}>
                  Add the flight's photos in the Photos step, then align them. That is
                  all the setup a drone block needs.
                </p>
              ) : (
                <p className="dim" style={{ maxWidth: '40ch', margin: 0 }}>
                  Add photographs in the Images step. Scanned film, digital
                  frames, GeoTIFF and PCIDSK <span className="data">.pix</span>
                  {' '}all open directly.
                </p>
              )}
            </>
          ) : (
            <>
              <h2>Open a photograph</h2>
              <div className="rows" style={{ minWidth: 300, textAlign: 'left' }}>
                {images.slice(0, 8).map((entry) => (
                  <button
                    key={entry.id}
                    className="row"
                    onClick={() => chooseImage(entry.id)}
                    disabled={!entry.online}
                  >
                    <div className="row__main">
                      <div className="row__name">
                        {entry.name}
                        {!entry.online && <span className="chip chip--bad">offline</span>}
                      </div>
                      <div className="row__meta">{entry.width} × {entry.height}</div>
                    </div>
                  </button>
                ))}
              </div>
            </>
          )}
        </div>
        <FiducialFrame />
      </div>
    )
  }

  // While a clip is being drawn, the region runs from its first corner to the
  // cursor, and replaces the saved one on screen.
  const drafting = clipDraft && clipDraft.imageId === image.id
  const clip = drafting
    ? (cursor
      ? [Math.min(clipDraft.col, cursor.col), Math.min(clipDraft.row, cursor.row),
         Math.max(clipDraft.col, cursor.col), Math.max(clipDraft.row, cursor.row)]
      : [clipDraft.col, clipDraft.row, clipDraft.col, clipDraft.row])
    : image.clipRegion

  return (
    <div className="viewer">
      <div className="viewer__toolbar">
        <div className="toolgroup">
          <select
            value={image.id}
            onChange={(event) => chooseImage(event.target.value)}
            title={popout ? 'Image in this window' : 'Active image'}
          >
            {(project?.images || []).map((entry) => (
              <option key={entry.id} value={entry.id} disabled={!entry.online}>
                {entry.name}{entry.online ? '' : '  (offline)'}
              </option>
            ))}
          </select>
        </div>

        <div className="toolgroup">
          <button
            className="toolgroup__btn"
            onClick={() => setFitToken((t) => t + 1)}
            title="Fit the whole image to the view"
          >
            Fit
          </button>
          <button
            className="toolgroup__btn"
            onClick={() => setView((v) => ({ ...v, scale: 1 }))}
            title="Actual pixels (1:1): one pixel of the photo on one pixel of the screen, the sharpest view for measuring"
          >
            1:1
          </button>
        </div>

        <div className="toolgroup">
          <button
            className="toolgroup__btn"
            onClick={() => turn((o) => rotateBy(o, -90))}
            title="Rotate 90° anticlockwise (display only)"
            aria-label="Rotate anticlockwise"
          >
            ⟲
          </button>
          <button
            className="toolgroup__btn"
            onClick={() => turn((o) => rotateBy(o, 90))}
            title="Rotate 90° clockwise (display only)"
            aria-label="Rotate clockwise"
          >
            ⟳
          </button>
          <button
            className={`toolgroup__btn ${orientation.flipX ? 'toolgroup__btn--on' : ''}`}
            onClick={() => turn(flip)}
            title="Mirror horizontally (display only)"
            aria-label="Flip"
            aria-pressed={orientation.flipX}
          >
            ⇋
          </button>
          {(orientation.rotate !== 0 || orientation.flipX) && (
            <span className="toolgroup__btn toolgroup__btn--note" title="Display orientation">
              {orientation.rotate}°{orientation.flipX ? ' mirrored' : ''}
            </span>
          )}
        </div>

        <div className="toolgroup">
          <select
            value={enhancement}
            onChange={(event) => setEnhancement(event.target.value)}
            title="Display stretch (does not alter the data)"
          >
            <option value="none">No stretch</option>
            <option value="linear">Linear</option>
            <option value="linear2pct">Linear 2%</option>
            <option value="stddev">2 std dev</option>
            <option value="equalize">Equalise</option>
            <option value="root">Root</option>
          </select>
        </div>

        <div className="toolgroup">
          <button
            className={`toolgroup__btn ${showOverlays ? 'toolgroup__btn--on' : ''}`}
            onClick={() => setShowOverlays((v) => !v)}
            title="Show measured points and marks"
          >
            Points
          </button>
          {project?.model?.residuals && (
            <button
              className={`toolgroup__btn ${showResiduals ? 'toolgroup__btn--on' : ''}`}
              onClick={() => setShowResiduals((v) => !v)}
              title="Residual vectors"
            >
              Residuals
            </button>
          )}
        </div>

        {canPopOut && (
          <div className="toolgroup">
            <button
              className={`toolgroup__btn ${poppedOut.includes(image.id) ? 'toolgroup__btn--on' : ''}`}
              onClick={() => popOut(image)}
              title="Open in a separate window"
            >
              {poppedOut.includes(image.id) ? 'Popped out' : 'Pop out'}
            </button>
          </div>
        )}

        {pickMode && (
          <div className="toolgroup" style={{ borderColor: 'var(--accent)' }}>
            <span className="toolgroup__btn toolgroup__btn--on">
              {pickMode.label} — click on the image
            </span>
            <button className="toolgroup__btn" onClick={() => pickMode.onCancel?.()}>
              Esc
            </button>
          </div>
        )}
      </div>

      <div
        ref={attachStage}
        className={[
          'viewer__stage',
          panning ? 'viewer__stage--panning' : '',
          pickMode ? 'viewer__stage--picking' : '',
        ].join(' ')}
        onWheel={onWheel}
        onPointerDown={onPointerDown}
        onPointerMove={onPointerMove}
        onPointerUp={onPointerUp}
        onPointerLeave={() => { setCursor(null); setPointer(null) }}
        onClick={onClick}
      >
        <div
          className="viewer__layer"
          style={{
            transform: `translate(${view.x}px, ${view.y}px) scale(${view.scale}) `
              + cssMatrix(image.width, image.height, orientation),
            width: image.width,
            height: image.height,
          }}
        >
          {[['base', base], ['fallback', fallback]].map(([layer, list]) => list.map((tile) => (
            <img
              key={`${layer}:${tile.key}`}
              className="viewer__tile viewer__tile--under"
              src={srcFor(tile)}
              alt=""
              draggable={false}
              onError={() => onTileError(tile.key)}
              style={{ left: tile.left, top: tile.top, width: tile.size, height: tile.size }}
            />
          )))}
          {tiles.map((tile) => (
            <img
              key={tile.key}
              className={`viewer__tile ${loadedRef.current.has(tile.key) ? '' : 'viewer__tile--loading'}`}
              src={srcFor(tile)}
              alt=""
              draggable={false}
              onLoad={() => onTileLoad(tile.key)}
              onError={() => onTileError(tile.key)}
              style={{
                left: tile.left,
                top: tile.top,
                width: tile.size,
                height: tile.size,
              }}
            />
          ))}

          {clip && (
            <div
              className={`clipbox ${drafting ? 'clipbox--draft' : ''}`}
              style={{
                left: clip[0] + 0.5,
                top: clip[1] + 0.5,
                width: clip[2] - clip[0],
                height: clip[3] - clip[1],
                borderWidth: Math.max(1, 1.5 / view.scale),
              }}
            />
          )}
        </div>

        {residualPlot && (
          <svg className="residuals">
            {residualPlot.vectors.map((vector) => {
              const at = place(vector.col, vector.row)
              const along = turnVector(vector.dx, vector.dy, image.width, image.height, orientation)
              const x = view.x + at.u * view.scale
              const y = view.y + at.v * view.scale
              const x2 = x + along.dx * residualPlot.factor * view.scale
              const y2 = y + along.dy * residualPlot.factor * view.scale
              const over = vector.magnitude > residualPlot.median * 3
              const angle = Math.atan2(y2 - y, x2 - x)
              const head = 5
              return (
                <g key={vector.id}>
                  <circle className="residuals__origin" cx={x} cy={y} r={2.5} />
                  <line
                    className={`residuals__vector ${over ? 'residuals__vector--over' : ''}`}
                    x1={x} y1={y} x2={x2} y2={y2}
                  />
                  <polygon
                    className={`residuals__head ${over ? 'residuals__head--over' : ''}`}
                    points={[
                      `${x2},${y2}`,
                      `${x2 - head * Math.cos(angle - 0.42)},${y2 - head * Math.sin(angle - 0.42)}`,
                      `${x2 - head * Math.cos(angle + 0.42)},${y2 - head * Math.sin(angle + 0.42)}`,
                    ].join(' ')}
                  />
                </g>
              )
            })}
          </svg>
        )}

        {/* Markers sit outside the scaled layer so they keep a constant
            screen size however far you zoom in. */}
        {markers.map((marker) => (
          <div
            key={marker.id}
            className={[
              'marker',
              `marker--${marker.kind}`,
              selectedPoint === marker.id ? 'marker--selected' : '',
            ].join(' ')}
            style={{
              left: view.x + place(marker.col, marker.row).u * view.scale,
              top: view.y + place(marker.col, marker.row).v * view.scale,
            }}
            title={marker.label}
          >
            <span className="marker__cross" />
            <span className="marker__ring" />
            {view.scale > 0.08 && <span className="marker__label">{marker.label}</span>}
          </div>
        ))}

        {pickMode?.loupe && loupeEnabled && view.scale < 1 && cursor && pointer && pyramid && (
          <Loupe
            image={image}
            cursor={cursor}
            pointer={pointer}
            stage={size}
            orientation={orientation}
            level={pyramid.maxZoom}
            srcTile={(pyramid.longest / (TILE * 2 ** pyramid.maxZoom)) * TILE}
            enhancement={enhancement}
          />
        )}
      </div>

      {residualPlot && (
        <div className="residuals__scalebar">
          <span className="residuals__rule" />
          <span className="residuals__cell">
            exaggerated ×{residualPlot.factor.toLocaleString()}
          </span>
          <span className="residuals__cell">
            worst {(residualPlot.worst * residualPlot.mmPerPixel * 1000).toFixed(1)} µm
          </span>
        </div>
      )}

      {/* The data strip. What a Wild RC30 prints along the edge of every
          exposure: which frame, what camera, how it is being read. */}
      <div className="datastrip">
        <span className="datastrip__cell">
          <span className="datastrip__value">{image.name}</span>
        </span>
        <span className="datastrip__cell">
          <span className="datastrip__key">frame</span>
          <span className="datastrip__value">
            {image.width}×{image.height}
          </span>
        </span>
        <span className="datastrip__cell">
          <span className="datastrip__key">bands</span>
          <span className="datastrip__value">{image.bandCount}</span>
        </span>
        {camera.focalMm > 0 && (
          <span className="datastrip__cell">
            <span className="datastrip__key">f</span>
            <span className="datastrip__value">{Number(camera.focalMm).toFixed(2)}mm</span>
          </span>
        )}
        {camera.imageScale > 0 && (
          <span className="datastrip__cell">
            <span className="datastrip__key">scale</span>
            <span className="datastrip__value">1:{camera.imageScale.toLocaleString()}</span>
          </span>
        )}
        <span className="datastrip__cell">
          <span className="datastrip__key">zoom</span>
          <span className="datastrip__value">{(view.scale * 100).toFixed(0)}%</span>
        </span>
        {image.exterior && (
          <span className="datastrip__cell">
            <span className="datastrip__key">alt</span>
            <span className="datastrip__value">
              {Number(image.exterior[2]).toFixed(0)}m
            </span>
          </span>
        )}
        {/* Last, so the fixed readouts do not shift as the cursor moves. */}
        {cursor && (
          <span className="datastrip__cell">
            <span className="datastrip__key">col row</span>
            <span className="datastrip__value datastrip__value--live">
              {cursor.col.toFixed(1)} {cursor.row.toFixed(1)}
            </span>
          </span>
        )}
      </div>
    </div>
  )
}

/**
 * Registration marks for an empty stage.
 *
 * These appear only when no photograph is open. Once there is an exposure on
 * the canvas, the exposure is the frame — and marks laid over the imagery
 * would be exactly the kind of decoration that gets in the way of the
 * measurement this whole application exists to make.
 */
function FiducialFrame() {
  const marks = [
    { top: 14, left: 14 }, { top: 14, right: 14 },
    { bottom: 14, left: 14 }, { bottom: 14, right: 14 },
  ]
  return (
    <div className="fiducial-frame" aria-hidden="true">
      {marks.map((position, index) => (
        <span
          key={index}
          className="fiducial-frame__mark"
          style={{ ...position, width: 12, height: 12 }}
        />
      ))}
    </div>
  )
}

const LOUPE = 200

/**
 * The magnifier: the photo at 1:1 around the cursor, while a point is being
 * measured with the view zoomed out. Drawn from the finest pyramid level,
 * whose tiles the viewer already requests, and turned the same way as the
 * view, so the crosshair sits on exactly the pixel the click will record.
 */
function Loupe({ image, cursor, pointer, stage, orientation, level, srcTile, enhancement }) {
  const half = LOUPE / 2
  const reach = half * 1.5                    // image pixels either side, any rotation
  const columns = Math.ceil(image.width / srcTile)
  const rows = Math.ceil(image.height / srcTile)
  const tx0 = Math.max(0, Math.floor((cursor.col - reach) / srcTile))
  const tx1 = Math.min(columns - 1, Math.floor((cursor.col + reach) / srcTile))
  const ty0 = Math.max(0, Math.floor((cursor.row - reach) / srcTile))
  const ty1 = Math.min(rows - 1, Math.floor((cursor.row + reach) / srcTile))

  const tiles = []
  for (let ty = ty0; ty <= ty1; ty += 1) {
    for (let tx = tx0; tx <= tx1; tx += 1) {
      tiles.push({ key: `${tx}/${ty}`, tx, ty })
    }
  }

  // The cursor is in pixel-centre coordinates; the layer has its edge at 0.
  const centre = toDisplay(cursor.col + 0.5, cursor.row + 0.5, image.width, image.height, orientation)

  // Above and to the right of the pointer, moved aside near the stage edges
  // so it never covers the spot being measured.
  let left = pointer.x + 28
  let top = pointer.y - LOUPE - 28
  if (left + LOUPE > stage.width) left = pointer.x - LOUPE - 28
  if (top < 0) top = pointer.y + 28

  return (
    <div className="loupe" style={{ left, top }} aria-hidden="true">
      <div
        className="loupe__layer"
        style={{
          transform: `translate(${half - centre.u}px, ${half - centre.v}px) `
            + cssMatrix(image.width, image.height, orientation),
          width: image.width,
          height: image.height,
        }}
      >
        {tiles.map((tile) => (
          <img
            key={tile.key}
            src={tileUrl(image.id, level, tile.tx, tile.ty, { enhancement })}
            alt=""
            draggable={false}
            style={{ left: tile.tx * srcTile, top: tile.ty * srcTile, width: srcTile, height: srcTile }}
          />
        ))}
      </div>
      <span className="loupe__cross" />
      <span className="loupe__label">1:1</span>
    </div>
  )
}
