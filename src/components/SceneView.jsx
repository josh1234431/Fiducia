import { useEffect, useMemo, useRef, useState } from 'react'
import { useStore } from '../state/store'
import { api } from '../lib/api'

/**
 * The 3D model: its points, and the cameras that took the photos, to turn
 * round and measure on.
 *
 * Drag to turn, right-drag or Shift-drag to move, scroll to come closer.
 * While picking, a click takes the point nearest the cursor, the one in
 * front where several line up, for a distance or the scale.
 */

// -- small matrix helpers (column-major, as WebGL wants) ----------------------------

function perspective(fovY, aspect, near, far) {
  const f = 1 / Math.tan(fovY / 2)
  const nf = 1 / (near - far)
  return [f / aspect, 0, 0, 0, 0, f, 0, 0, 0, 0, (far + near) * nf, -1, 0, 0, 2 * far * near * nf, 0]
}

function lookAt(eye, target, up) {
  const z = normalise(sub(eye, target))
  const x = normalise(cross(up, z))
  const y = cross(z, x)
  return [x[0], y[0], z[0], 0, x[1], y[1], z[1], 0, x[2], y[2], z[2], 0,
    -dot(x, eye), -dot(y, eye), -dot(z, eye), 1]
}

function multiply(a, b) {
  const out = new Array(16).fill(0)
  for (let c = 0; c < 4; c++) {
    for (let r = 0; r < 4; r++) {
      let sum = 0
      for (let k = 0; k < 4; k++) sum += a[k * 4 + r] * b[c * 4 + k]
      out[c * 4 + r] = sum
    }
  }
  return out
}

const sub = (a, b) => [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
const add = (a, b) => [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
const scale = (a, s) => [a[0] * s, a[1] * s, a[2] * s]
const dot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
const cross = (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]
const length = (a) => Math.hypot(a[0], a[1], a[2])
const normalise = (a) => scale(a, 1 / (length(a) || 1))

/** A CSS colour token as [r, g, b] from 0 to 1. */
function cssColour(name, fallback) {
  const probe = document.createElement('span')
  probe.style.color = `var(${name}, ${fallback})`
  document.body.appendChild(probe)
  const value = getComputedStyle(probe).color
  probe.remove()
  const parts = value.match(/[\d.]+/g)?.map(Number) || [200, 200, 200]
  return [parts[0] / 255, parts[1] / 255, parts[2] / 255]
}

// -- shaders ---------------------------------------------------------------------

const POINT_VS = `#version 300 es
in vec3 position;
in vec3 colour;
uniform mat4 mvp;
uniform float size;
out vec3 tint;
void main() {
  gl_Position = mvp * vec4(position, 1.0);
  gl_PointSize = size;
  tint = colour;
}`

const POINT_FS = `#version 300 es
precision mediump float;
in vec3 tint;
uniform vec3 solid;
uniform float useSolid;
out vec4 colour;
void main() {
  vec2 p = gl_PointCoord * 2.0 - 1.0;
  if (dot(p, p) > 1.0) discard;
  colour = vec4(mix(tint, solid, useSolid), 1.0);
}`

const LINE_VS = `#version 300 es
in vec3 position;
uniform mat4 mvp;
void main() { gl_Position = mvp * vec4(position, 1.0); }`

const LINE_FS = `#version 300 es
precision mediump float;
uniform vec3 solid;
uniform float alpha;
out vec4 colour;
void main() { colour = vec4(solid, alpha); }`

function program(gl, vs, fs) {
  const compile = (type, source) => {
    const shader = gl.createShader(type)
    gl.shaderSource(shader, source)
    gl.compileShader(shader)
    if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(shader))
    return shader
  }
  const p = gl.createProgram()
  gl.attachShader(p, compile(gl.VERTEX_SHADER, vs))
  gl.attachShader(p, compile(gl.FRAGMENT_SHADER, fs))
  gl.linkProgram(p)
  if (!gl.getProgramParameter(p, gl.LINK_STATUS)) throw new Error(gl.getProgramInfoLog(p))
  return p
}

export default function SceneView() {
  const project = useStore((s) => s.project)
  const layer = useStore((s) => s.sceneLayer)
  const setLayer = useStore((s) => s.setSceneLayer)
  const showCameras = useStore((s) => s.sceneCameras)
  const setShowCameras = useStore((s) => s.setSceneCameras)
  const picks = useStore((s) => s.scenePicks)
  const addPick = useStore((s) => s.addScenePick)
  const picking = useStore((s) => s.scenePicking)
  const setPicking = useStore((s) => s.setScenePicking)
  const setActiveImage = useStore((s) => s.setActiveImage)
  const setStep = useStore((s) => s.setStep)

  const scene = project?.scene
  const dense = scene?.dense
  const units = scene?.transform?.scaleSource === 'none' ? 'units' : 'm'
  const shown = layer === 'dense' && dense ? 'dense' : 'sparse'

  const canvasRef = useRef(null)
  const stageRef = useRef(null)
  const gl = useRef(null)
  const cloud = useRef(null)           // { count, positions, buffers }
  const orbit = useRef(null)           // { target, yaw, pitch, distance }
  const drag = useRef(null)
  const frame = useRef(0)
  const [view, setView] = useState(null)
  const [loaded, setLoaded] = useState(null)   // { which, count }
  const [error, setError] = useState(null)
  const [pointSize, setPointSize] = useState(null)

  // What the view is drawn from changes when the photos are aligned again,
  // the scale is set, or a dense cloud is built.
  const version = [scene?.alignedAt, scene?.transform?.scale, dense?.builtAt].join('|')

  useEffect(() => {
    // A new alignment or scale moves every point: the view and picks start again.
    let live = true
    orbit.current = null
    useStore.getState().clearScenePicks()
    api.scene.view().then((result) => { if (live) { setView(result); setError(null) } })
      .catch((e) => { if (live) setError(e.message) })
    return () => { live = false }
  }, [version])

  // -- WebGL ----------------------------------------------------------------------
  useEffect(() => {
    const canvas = canvasRef.current
    const context = canvas.getContext('webgl2', { antialias: true, premultipliedAlpha: true })
    if (!context) { setError('This computer cannot draw in 3D (WebGL 2 is not available).'); return }
    gl.current = {
      context,
      points: program(context, POINT_VS, POINT_FS),
      lines: program(context, LINE_VS, LINE_FS),
      lineBuffer: context.createBuffer(),
      markBuffer: context.createBuffer(),
    }
    const observer = new ResizeObserver(() => draw())
    observer.observe(stageRef.current)
    return () => observer.disconnect()
  }, []) // eslint-disable-line react-hooks/exhaustive-deps

  useEffect(() => {
    if (!view || !gl.current) return
    let live = true
    setLoaded(null)
    api.scene.cloud(shown, view.origin).then((result) => {
      if (!live) return
      const g = gl.current.context
      const positions = g.createBuffer()
      g.bindBuffer(g.ARRAY_BUFFER, positions)
      g.bufferData(g.ARRAY_BUFFER, result.positions, g.STATIC_DRAW)
      const colours = g.createBuffer()
      g.bindBuffer(g.ARRAY_BUFFER, colours)
      g.bufferData(g.ARRAY_BUFFER, result.colours, g.STATIC_DRAW)
      if (cloud.current) {
        g.deleteBuffer(cloud.current.buffers[0])
        g.deleteBuffer(cloud.current.buffers[1])
      }
      cloud.current = { count: result.count, positions: result.positions, buffers: [positions, colours] }
      if (!orbit.current) orbit.current = home(result.positions, view)
      setLoaded({ which: shown, count: result.count })
    }).catch((e) => { if (live) setError(e.message) })
    return () => { live = false }
  }, [view, shown])

  useEffect(() => { draw() })

  /** A view of the whole model, from the side the cameras mostly stood. */
  function home(positions, viewInfo) {
    const n = positions.length / 3
    const step = Math.max(1, Math.floor(n / 20000))
    const xs = [], ys = [], zs = []
    for (let i = 0; i < n; i += step) {
      xs.push(positions[3 * i]); ys.push(positions[3 * i + 1]); zs.push(positions[3 * i + 2])
    }
    const median = (v) => { const s = [...v].sort((a, b) => a - b); return s[Math.floor(s.length / 2)] || 0 }
    const spread = (v) => {
      const s = [...v].sort((a, b) => a - b)
      return (s[Math.floor(s.length * 0.95)] - s[Math.floor(s.length * 0.05)]) || 1
    }
    const target = [median(xs), median(ys), median(zs)]
    const size = Math.max(spread(xs), spread(ys), spread(zs))
    const cams = viewInfo.cameras.map((c) => sub(c.centre, viewInfo.origin))
    let yaw = -Math.PI / 2
    if (cams.length) {
      const mean = scale(cams.reduce((a, c) => add(a, c), [0, 0, 0]), 1 / cams.length)
      const away = sub(mean, target)
      if (Math.hypot(away[0], away[1]) > 1e-6) yaw = Math.atan2(away[1], away[0])
    }
    return { target, yaw, pitch: 0.45, distance: 1.6 * size }
  }

  function matrices() {
    const canvas = canvasRef.current
    const o = orbit.current
    const eye = add(o.target, scale([Math.cos(o.pitch) * Math.cos(o.yaw), Math.cos(o.pitch) * Math.sin(o.yaw),
      Math.sin(o.pitch)], o.distance))
    const projection = perspective(50 * Math.PI / 180, canvas.width / Math.max(canvas.height, 1),
      o.distance / 1000, o.distance * 200)
    return { mvp: multiply(projection, lookAt(eye, o.target, [0, 0, 1])), eye }
  }

  // Drawn on the next frame by the latest render, whatever asked for it.
  const latest = useRef(null)
  latest.current = render
  function draw() {
    cancelAnimationFrame(frame.current)
    frame.current = requestAnimationFrame(() => latest.current?.())
  }

  function render() {
    const g = gl.current?.context
    const canvas = canvasRef.current
    const stage = stageRef.current
    if (!g || !canvas || !stage) return
    const ratio = window.devicePixelRatio || 1
    const width = Math.round(stage.clientWidth * ratio), height = Math.round(stage.clientHeight * ratio)
    if (canvas.width !== width || canvas.height !== height) { canvas.width = width; canvas.height = height }
    g.viewport(0, 0, width, height)
    g.clearColor(0, 0, 0, 0)
    g.clear(g.COLOR_BUFFER_BIT | g.DEPTH_BUFFER_BIT)
    if (!cloud.current || !orbit.current || !view) return
    g.enable(g.DEPTH_TEST)
    const { mvp } = matrices()

    const pts = gl.current.points
    g.useProgram(pts)
    g.uniformMatrix4fv(g.getUniformLocation(pts, 'mvp'), false, mvp)
    const size = (pointSize ?? (shown === 'dense' ? 2 : 3)) * ratio
    g.uniform1f(g.getUniformLocation(pts, 'size'), size)
    g.uniform1f(g.getUniformLocation(pts, 'useSolid'), 0)
    const [positions, colours] = cloud.current.buffers
    const pLoc = g.getAttribLocation(pts, 'position')
    const cLoc = g.getAttribLocation(pts, 'colour')
    g.bindBuffer(g.ARRAY_BUFFER, positions)
    g.enableVertexAttribArray(pLoc)
    g.vertexAttribPointer(pLoc, 3, g.FLOAT, false, 0, 0)
    g.bindBuffer(g.ARRAY_BUFFER, colours)
    g.enableVertexAttribArray(cLoc)
    g.vertexAttribPointer(cLoc, 3, g.UNSIGNED_BYTE, true, 0, 0)
    g.drawArrays(g.POINTS, 0, cloud.current.count)
    g.disableVertexAttribArray(cLoc)

    const lines = gl.current.lines
    g.useProgram(lines)
    g.uniformMatrix4fv(g.getUniformLocation(lines, 'mvp'), false, mvp)
    const lLoc = g.getAttribLocation(lines, 'position')
    g.enable(g.BLEND)
    g.blendFunc(g.SRC_ALPHA, g.ONE_MINUS_SRC_ALPHA)
    if (showCameras && frustums.length) {
      g.bindBuffer(g.ARRAY_BUFFER, gl.current.lineBuffer)
      g.bufferData(g.ARRAY_BUFFER, frustums, g.DYNAMIC_DRAW)
      g.enableVertexAttribArray(lLoc)
      g.vertexAttribPointer(lLoc, 3, g.FLOAT, false, 0, 0)
      g.uniform3fv(g.getUniformLocation(lines, 'solid'), palette.camera)
      g.uniform1f(g.getUniformLocation(lines, 'alpha'), 0.85)
      g.drawArrays(g.LINES, 0, frustums.length / 3)
    }

    // The picked points, on top of everything.
    if (picks.length) {
      g.disable(g.DEPTH_TEST)
      const local = picks.map((p) => sub([p.x, p.y, p.z], view.origin))
      if (local.length === 2) {
        g.bindBuffer(g.ARRAY_BUFFER, gl.current.markBuffer)
        g.bufferData(g.ARRAY_BUFFER, new Float32Array([...local[0], ...local[1]]), g.DYNAMIC_DRAW)
        g.enableVertexAttribArray(lLoc)
        g.vertexAttribPointer(lLoc, 3, g.FLOAT, false, 0, 0)
        g.uniform3fv(g.getUniformLocation(lines, 'solid'), palette.mark)
        g.uniform1f(g.getUniformLocation(lines, 'alpha'), 1)
        g.drawArrays(g.LINES, 0, 2)
      }
      g.useProgram(pts)
      g.uniform1f(g.getUniformLocation(pts, 'size'), 11 * ratio)
      g.uniform1f(g.getUniformLocation(pts, 'useSolid'), 1)
      g.uniform3fv(g.getUniformLocation(pts, 'solid'), palette.mark)
      g.bindBuffer(g.ARRAY_BUFFER, gl.current.markBuffer)
      g.bufferData(g.ARRAY_BUFFER, new Float32Array(local.flat()), g.DYNAMIC_DRAW)
      g.enableVertexAttribArray(pLoc)
      g.vertexAttribPointer(pLoc, 3, g.FLOAT, false, 0, 0)
      g.vertexAttrib3f(cLoc, 1, 1, 1)
      g.drawArrays(g.POINTS, 0, local.length)
    }
    g.disable(g.BLEND)
  }

  const palette = useMemo(() => ({
    camera: cssColour('--point-check', '#8fb0d6'),
    mark: cssColour('--point-fiducial', '#f08a6e'),
  }), [])

  // Each camera as a small pyramid: its centre, and its photo's frame in front.
  const frustums = useMemo(() => {
    if (!view?.cameras?.length || !cloud.current) return new Float32Array()
    const o = orbit.current
    const s = (o ? o.distance : 10) * 0.035
    const out = []
    for (const c of view.cameras) {
      const C = sub(c.centre, view.origin)
      const [right, down, forward] = c.axes
      const hw = c.width / (2 * c.focalPx), hh = c.height / (2 * c.focalPx)
      const corner = (sx, sy) => add(C, scale(add(forward, add(scale(right, sx * hw), scale(down, sy * hh))), s))
      const k = [corner(-1, -1), corner(1, -1), corner(1, 1), corner(-1, 1)]
      for (let i = 0; i < 4; i++) out.push(...C, ...k[i], ...k[i], ...k[(i + 1) % 4])
      // A tick on the top edge, so upside-down photos show.
      const top = add(C, scale(add(forward, scale(down, -hh * 1.35)), s))
      out.push(...k[0], ...top, ...top, ...k[1])
    }
    return new Float32Array(out)
  }, [view, loaded]) // eslint-disable-line react-hooks/exhaustive-deps

  // -- interaction -------------------------------------------------------------------

  function onPointerDown(event) {
    if (!orbit.current) return
    event.currentTarget.setPointerCapture(event.pointerId)
    drag.current = {
      x: event.clientX, y: event.clientY, moved: false,
      pan: event.button === 2 || event.shiftKey,
    }
  }

  function onPointerMove(event) {
    const d = drag.current
    if (!d) return
    const dx = event.clientX - d.x, dy = event.clientY - d.y
    if (Math.abs(dx) + Math.abs(dy) > 2) d.moved = true
    d.x = event.clientX; d.y = event.clientY
    const o = orbit.current
    if (d.pan) {
      const yawRight = [-Math.sin(o.yaw), Math.cos(o.yaw), 0]
      const upish = [-Math.sin(o.pitch) * Math.cos(o.yaw), -Math.sin(o.pitch) * Math.sin(o.yaw), Math.cos(o.pitch)]
      const perPixel = o.distance * 0.9 / Math.max(stageRef.current.clientHeight, 1)
      o.target = add(o.target, add(scale(yawRight, -dx * perPixel), scale(upish, dy * perPixel)))
    } else {
      o.yaw -= dx * 0.008
      o.pitch = Math.max(-1.45, Math.min(1.45, o.pitch + dy * 0.008))
    }
    draw()
  }

  function onPointerUp(event) {
    const d = drag.current
    drag.current = null
    if (d && !d.moved && picking && event.button === 0) pick(event)
  }

  function onWheel(event) {
    if (!orbit.current) return
    orbit.current.distance *= Math.exp(event.deltaY * 0.0012)
    draw()
  }

  /** The point nearest the cursor on screen, the front one where several are close. */
  function pick(event) {
    if (!cloud.current) return
    const rect = canvasRef.current.getBoundingClientRect()
    const px = event.clientX - rect.left, py = event.clientY - rect.top
    const { mvp } = matrices()
    const w = rect.width, h = rect.height
    const positions = cloud.current.positions
    const n = cloud.current.count
    const radius = 8
    let best = -1, bestDepth = Infinity, bestScreen = Infinity
    for (let i = 0; i < n; i++) {
      const x = positions[3 * i], y = positions[3 * i + 1], z = positions[3 * i + 2]
      const cw = mvp[3] * x + mvp[7] * y + mvp[11] * z + mvp[15]
      if (cw <= 0) continue
      const sx = ((mvp[0] * x + mvp[4] * y + mvp[8] * z + mvp[12]) / cw * 0.5 + 0.5) * w
      const sy = (0.5 - (mvp[1] * x + mvp[5] * y + mvp[9] * z + mvp[13]) / cw * 0.5) * h
      const gap = Math.hypot(sx - px, sy - py)
      if (gap > radius) continue
      // In front first; between points at much the same depth, the nearer the cursor.
      if (cw < bestDepth * 0.995 || (cw < bestDepth * 1.005 && gap < bestScreen)) {
        best = i; bestDepth = cw; bestScreen = gap
      }
    }
    if (best < 0) return
    const [ox, oy, oz] = view.origin
    addPick({ x: positions[3 * best] + ox, y: positions[3 * best + 1] + oy, z: positions[3 * best + 2] + oz })
  }

  const distance = picks.length === 2
    ? Math.hypot(picks[0].x - picks[1].x, picks[0].y - picks[1].y, picks[0].z - picks[1].z)
    : null

  return (
    <div className="viewer">
      <div className="viewer__toolbar">
        <div className="toolgroup" role="group" aria-label="Cloud">
          <button className={`toolgroup__btn ${shown === 'sparse' ? 'toolgroup__btn--on' : ''}`}
                  onClick={() => setLayer('sparse')} aria-pressed={shown === 'sparse'}
                  title="The tie points the photos were placed with">
            Sparse
          </button>
          <button className={`toolgroup__btn ${shown === 'dense' ? 'toolgroup__btn--on' : ''}`}
                  onClick={() => setLayer('dense')} disabled={!dense} aria-pressed={shown === 'dense'}
                  title={dense ? 'The dense point cloud' : 'Build the dense cloud first'}>
            Dense
          </button>
        </div>
        <div className="toolgroup">
          <button className={`toolgroup__btn toolgroup__btn--switch ${showCameras ? 'toolgroup__btn--on' : ''}`}
                  aria-pressed={showCameras}
                  onClick={() => setShowCameras(!showCameras)} title="Show where each photo was taken">
            Cameras
          </button>
          <button className={`toolgroup__btn toolgroup__btn--switch ${picking ? 'toolgroup__btn--on' : ''}`}
                  aria-pressed={picking}
                  onClick={() => setPicking(!picking)}
                  title="Click points on the model to measure between them, or to set the scale">
            Pick points
          </button>
        </div>
        <div className="toolgroup">
          <button className="toolgroup__btn" title="Smaller points"
                  onClick={() => setPointSize(Math.max(1, (pointSize ?? (shown === 'dense' ? 2 : 3)) - 1))}>−</button>
          <button className="toolgroup__btn" title="Larger points"
                  onClick={() => setPointSize(Math.min(12, (pointSize ?? (shown === 'dense' ? 2 : 3)) + 1))}>+</button>
          <button className="toolgroup__btn" title="Back to the whole model"
                  onClick={() => { if (cloud.current) { orbit.current = home(cloud.current.positions, view); draw() } }}>
            Fit
          </button>
        </div>
        {loaded && (
          <div className="toolgroup">
            <span className="toolgroup__btn toolgroup__btn--note">
              {loaded.count.toLocaleString()} {loaded.which} points
              {distance != null ? ` · picked points ${distance.toFixed(units === 'm' ? 3 : 4)} ${units} apart` : ''}
            </span>
          </div>
        )}
      </div>

      <div ref={stageRef}
           className={`viewer__stage scene3d ${picking ? 'viewer__stage--picking' : ''}`}
           onPointerDown={onPointerDown} onPointerMove={onPointerMove} onPointerUp={onPointerUp}
           onWheel={onWheel} onContextMenu={(e) => e.preventDefault()}>
        <canvas ref={canvasRef} className="scene3d__canvas" />
        {error && <div className="viewer__empty"><p>{error}</p></div>}
        {!error && !loaded && <div className="viewer__empty"><p>Loading the model…</p></div>}
      </div>

      {view && (
        <div className="viewer__legend">
          <span className="viewer__legend-item"><span className="scene3d__swatch scene3d__swatch--camera" />Camera</span>
          <span className="viewer__legend-item"><span className="scene3d__swatch scene3d__swatch--mark" />Picked point</span>
          <span className="viewer__legend-item">
            {view.cameras.length} photos placed
            {scene?.unplaced?.length ? ` · ${scene.unplaced.length} not placed` : ''}
          </span>
          {scene?.unplaced?.length > 0 && (
            <button className="btn btn--ghost btn--sm"
                    onClick={() => {
                      const first = project.images.find((i) => i.name === scene.unplaced[0])
                      if (first) { setActiveImage(first.id); setStep('images') }
                    }}>
              Show the first
            </button>
          )}
        </div>
      )}
    </div>
  )
}
