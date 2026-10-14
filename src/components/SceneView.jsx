import { useEffect, useMemo, useRef, useState } from 'react'
import { useStore } from '../state/store'
import { api } from '../lib/api'

/**
 * The 3D model: its points or its surface, and the cameras that took the
 * photos, to turn round, walk through and measure on.
 *
 * Orbit: drag to turn, right-drag or Shift-drag to move, scroll to come
 * closer. Walk: W A S D (or the arrows) to move, Q and E to go down and up,
 * Shift to hurry, drag to look round, scroll to step forward and back.
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
/** A unit direction from a heading (from +x, anticlockwise) and an elevation. */
const direction = (yaw, pitch) => [Math.cos(pitch) * Math.cos(yaw), Math.cos(pitch) * Math.sin(yaw), Math.sin(pitch)]
const clampPitch = (p) => Math.max(-1.5, Math.min(1.5, p))

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

// Everything outside the crop box is left undrawn.
const CROP = `
uniform vec3 cropMin;
uniform vec3 cropMax;
uniform float useCrop;
bool cropped(vec3 p) {
  return useCrop > 0.5 && (any(lessThan(p, cropMin)) || any(greaterThan(p, cropMax)));
}`

const POINT_VS = `#version 300 es
in vec3 position;
in vec3 colour;
uniform mat4 mvp;
uniform float size;
out vec3 tint;
out vec3 local;
void main() {
  gl_Position = mvp * vec4(position, 1.0);
  gl_PointSize = size;
  tint = colour;
  local = position;
}`

const POINT_FS = `#version 300 es
precision highp float;
in vec3 tint;
in vec3 local;
uniform vec3 solid;
uniform float useSolid;
${CROP}
out vec4 colour;
void main() {
  vec2 p = gl_PointCoord * 2.0 - 1.0;
  if (dot(p, p) > 1.0 || cropped(local)) discard;
  colour = vec4(mix(tint, solid, useSolid), 1.0);
}`

const MESH_VS = `#version 300 es
in vec3 position;
in vec3 normal;
in vec3 colour;
uniform mat4 mvp;
out vec3 tint;
out vec3 n;
out vec3 local;
void main() {
  gl_Position = mvp * vec4(position, 1.0);
  tint = colour;
  n = normal;
  local = position;
}`

// Lit from the viewer, with some light from above, so shape reads even
// where the photos' colours are flat; both sides lit alike.
const MESH_FS = `#version 300 es
precision highp float;
in vec3 tint;
in vec3 n;
in vec3 local;
uniform vec3 light;
uniform float lit;
${CROP}
out vec4 colour;
void main() {
  if (cropped(local)) discard;
  vec3 m = normalize(n);
  float shade = 0.32 + 0.5 * abs(dot(m, light)) + 0.18 * max(m.z, 0.0);
  colour = vec4(tint * mix(1.0, shade, lit), 1.0);
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

function buffer(g, target, data) {
  const b = g.createBuffer()
  g.bindBuffer(target, b)
  g.bufferData(target, data, g.STATIC_DRAW)
  return b
}

const WALK_KEYS = new Set(['w', 'a', 's', 'd', 'q', 'e', 'arrowup', 'arrowdown', 'arrowleft', 'arrowright',
  'pageup', 'pagedown'])

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
  const crop = useStore((s) => s.sceneCrop)
  const setActiveImage = useStore((s) => s.setActiveImage)
  const setStep = useStore((s) => s.setStep)

  const scene = project?.scene
  const dense = scene?.dense
  const mesh = scene?.mesh
  const units = scene?.transform?.scaleSource === 'none' ? 'units' : 'm'
  const shown = layer === 'mesh' && mesh ? 'mesh' : layer === 'dense' && dense ? 'dense' : 'sparse'

  const canvasRef = useRef(null)
  const stageRef = useRef(null)
  const gl = useRef(null)
  const model = useRef(null)           // what is drawn: points or a mesh, and its GPU buffers
  const cam = useRef(null)             // orbit: target, yaw, pitch, distance; walk: eye, lookYaw, lookPitch
  const drag = useRef(null)
  const keys = useRef(new Set())
  const frame = useRef(0)
  const walking = useRef(0)
  const [view, setView] = useState(null)
  const [loaded, setLoaded] = useState(null)   // { which, count, triangles }
  const [error, setError] = useState(null)
  const [pointSize, setPointSize] = useState(null)
  const [walk, setWalk] = useState(false)
  const [lit, setLit] = useState(true)

  // What the view is drawn from changes when the photos are aligned again,
  // the scale is set, or a cloud or mesh is built.
  const version = [scene?.alignedAt, scene?.transform?.scale, dense?.builtAt, mesh?.builtAt].join('|')

  useEffect(() => {
    // A new alignment or scale moves every point: the view and picks start again.
    let live = true
    cam.current = null
    setWalk(false)
    useStore.getState().clearScenePicks()
    api.scene.view().then((result) => {
      if (!live) return
      setView(result)
      setError(null)
      useStore.getState().setSceneCrop(result.crop || null)
    }).catch((e) => { if (live) setError(e.message) })
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
      mesh: program(context, MESH_VS, MESH_FS),
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
    const g = gl.current.context
    const request = shown === 'mesh' ? api.scene.meshData(view.origin) : api.scene.cloud(shown, view.origin)
    request.then((result) => {
      if (!live) return
      const previous = model.current
      if (shown === 'mesh') {
        model.current = {
          kind: 'mesh', count: result.vertices, triangles: result.triangles, positions: result.positions,
          buffers: [buffer(g, g.ARRAY_BUFFER, result.positions), buffer(g, g.ARRAY_BUFFER, result.normals),
            buffer(g, g.ARRAY_BUFFER, result.colours), buffer(g, g.ELEMENT_ARRAY_BUFFER, result.indices)],
        }
      } else {
        model.current = {
          kind: 'points', count: result.count, positions: result.positions,
          buffers: [buffer(g, g.ARRAY_BUFFER, result.positions), buffer(g, g.ARRAY_BUFFER, result.colours)],
        }
      }
      previous?.buffers.forEach((b) => g.deleteBuffer(b))
      if (!cam.current) cam.current = home(result.positions, view)
      setLoaded({ which: shown, count: model.current.count, triangles: model.current.triangles })
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
    return { target, yaw, pitch: 0.45, distance: 1.6 * size, size, walk: false }
  }

  /** The eye, and where it looks, for either way of moving. */
  function pose() {
    const c = cam.current
    if (c.walk) return { eye: c.eye, target: add(c.eye, direction(c.lookYaw, c.lookPitch)), near: c.size / 3000 }
    return { eye: add(c.target, scale(direction(c.yaw, c.pitch), c.distance)), target: c.target, near: c.distance / 1000 }
  }

  function matrices() {
    const canvas = canvasRef.current
    const { eye, target, near } = pose()
    const projection = perspective(55 * Math.PI / 180, canvas.width / Math.max(canvas.height, 1), near,
      cam.current.size * 400)
    return { mvp: multiply(projection, lookAt(eye, target, [0, 0, 1])), eye, target }
  }

  // Drawn on the next frame by the latest render, whatever asked for it.
  const latest = useRef(null)
  latest.current = render
  function draw() {
    cancelAnimationFrame(frame.current)
    frame.current = requestAnimationFrame(() => latest.current?.())
  }

  function setCrop(g, p) {
    const box = crop && view
    g.uniform1f(g.getUniformLocation(p, 'useCrop'), box ? 1 : 0)
    if (box) {
      g.uniform3fv(g.getUniformLocation(p, 'cropMin'), sub(crop.min, view.origin))
      g.uniform3fv(g.getUniformLocation(p, 'cropMax'), sub(crop.max, view.origin))
    }
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
    if (!model.current || !cam.current || !view) return
    g.enable(g.DEPTH_TEST)
    const { mvp, eye, target } = matrices()
    const m = model.current

    const pts = gl.current.points
    if (m.kind === 'mesh') {
      const p = gl.current.mesh
      g.useProgram(p)
      g.uniformMatrix4fv(g.getUniformLocation(p, 'mvp'), false, mvp)
      g.uniform3fv(g.getUniformLocation(p, 'light'), normalise(sub(eye, target)))
      g.uniform1f(g.getUniformLocation(p, 'lit'), lit ? 1 : 0)
      setCrop(g, p)
      const [positions, normals, colours, indices] = m.buffers
      const attributes = [['position', positions, 3, g.FLOAT, false], ['normal', normals, 3, g.FLOAT, false],
        ['colour', colours, 3, g.UNSIGNED_BYTE, true]]
      const locations = attributes.map(([name, b, size, type, norm]) => {
        const loc = g.getAttribLocation(p, name)
        g.bindBuffer(g.ARRAY_BUFFER, b)
        g.enableVertexAttribArray(loc)
        g.vertexAttribPointer(loc, size, type, norm, 0, 0)
        return loc
      })
      g.bindBuffer(g.ELEMENT_ARRAY_BUFFER, indices)
      g.drawElements(g.TRIANGLES, m.triangles * 3, g.UNSIGNED_INT, 0)
      locations.forEach((loc) => g.disableVertexAttribArray(loc))
    } else {
      g.useProgram(pts)
      g.uniformMatrix4fv(g.getUniformLocation(pts, 'mvp'), false, mvp)
      const size = (pointSize ?? (shown === 'dense' ? 2 : 3)) * ratio
      g.uniform1f(g.getUniformLocation(pts, 'size'), size)
      g.uniform1f(g.getUniformLocation(pts, 'useSolid'), 0)
      setCrop(g, pts)
      const [positions, colours] = m.buffers
      const pLoc = g.getAttribLocation(pts, 'position')
      const cLoc = g.getAttribLocation(pts, 'colour')
      g.bindBuffer(g.ARRAY_BUFFER, positions)
      g.enableVertexAttribArray(pLoc)
      g.vertexAttribPointer(pLoc, 3, g.FLOAT, false, 0, 0)
      g.bindBuffer(g.ARRAY_BUFFER, colours)
      g.enableVertexAttribArray(cLoc)
      g.vertexAttribPointer(cLoc, 3, g.UNSIGNED_BYTE, true, 0, 0)
      g.drawArrays(g.POINTS, 0, m.count)
      g.disableVertexAttribArray(cLoc)
    }

    const lines = gl.current.lines
    g.useProgram(lines)
    g.uniformMatrix4fv(g.getUniformLocation(lines, 'mvp'), false, mvp)
    const lLoc = g.getAttribLocation(lines, 'position')
    g.enable(g.BLEND)
    g.blendFunc(g.SRC_ALPHA, g.ONE_MINUS_SRC_ALPHA)
    const drawLines = (data, colour, alpha) => {
      g.bindBuffer(g.ARRAY_BUFFER, gl.current.lineBuffer)
      g.bufferData(g.ARRAY_BUFFER, data, g.DYNAMIC_DRAW)
      g.enableVertexAttribArray(lLoc)
      g.vertexAttribPointer(lLoc, 3, g.FLOAT, false, 0, 0)
      g.uniform3fv(g.getUniformLocation(lines, 'solid'), colour)
      g.uniform1f(g.getUniformLocation(lines, 'alpha'), alpha)
      g.drawArrays(g.LINES, 0, data.length / 3)
    }
    if (showCameras && frustums.length) drawLines(frustums, palette.camera, 0.85)
    if (crop) drawLines(cropLines(sub(crop.min, view.origin), sub(crop.max, view.origin)), palette.crop, 0.9)

    // The picked points, on top of everything.
    if (picks.length) {
      g.disable(g.DEPTH_TEST)
      const local = picks.map((p) => sub([p.x, p.y, p.z], view.origin))
      if (local.length === 2) drawLines(new Float32Array([...local[0], ...local[1]]), palette.mark, 1)
      g.useProgram(pts)
      g.uniformMatrix4fv(g.getUniformLocation(pts, 'mvp'), false, mvp)
      g.uniform1f(g.getUniformLocation(pts, 'size'), 11 * ratio)
      g.uniform1f(g.getUniformLocation(pts, 'useSolid'), 1)
      g.uniform1f(g.getUniformLocation(pts, 'useCrop'), 0)
      g.uniform3fv(g.getUniformLocation(pts, 'solid'), palette.mark)
      const pLoc = g.getAttribLocation(pts, 'position')
      const cLoc = g.getAttribLocation(pts, 'colour')
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
    crop: cssColour('--signal-warn', '#d4a259'),
  }), [])

  // Each camera as a small pyramid: its centre, and its photo's frame in front.
  const frustums = useMemo(() => {
    if (!view?.cameras?.length || !model.current) return new Float32Array()
    const s = (cam.current ? cam.current.size * 1.6 : 10) * 0.035
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

  function cropLines(lo, hi) {
    const c = (x, y, z) => [x ? hi[0] : lo[0], y ? hi[1] : lo[1], z ? hi[2] : lo[2]]
    const out = []
    for (const [a, b] of [[[0, 0, 0], [1, 0, 0]], [[0, 1, 0], [1, 1, 0]], [[0, 0, 1], [1, 0, 1]], [[0, 1, 1], [1, 1, 1]],
      [[0, 0, 0], [0, 1, 0]], [[1, 0, 0], [1, 1, 0]], [[0, 0, 1], [0, 1, 1]], [[1, 0, 1], [1, 1, 1]],
      [[0, 0, 0], [0, 0, 1]], [[1, 0, 0], [1, 0, 1]], [[0, 1, 0], [0, 1, 1]], [[1, 1, 0], [1, 1, 1]]]) {
      out.push(...c(...a), ...c(...b))
    }
    return new Float32Array(out)
  }

  // -- moving ------------------------------------------------------------------------

  function setWalking(on) {
    const c = cam.current
    if (!c || on === c.walk) { setWalk(on); return }
    if (on) {
      // Standing where the orbit's eye was, looking the same way.
      c.eye = add(c.target, scale(direction(c.yaw, c.pitch), c.distance))
      c.lookYaw = c.yaw + Math.PI
      c.lookPitch = -c.pitch
      stageRef.current?.focus()
    } else {
      c.target = add(c.eye, scale(direction(c.lookYaw, c.lookPitch), c.distance))
      c.yaw = c.lookYaw - Math.PI
      c.pitch = -c.lookPitch
    }
    c.walk = on
    setWalk(on)
    draw()
  }

  /** Move the walker for dt seconds with the keys held. */
  function move(dt, k) {
    const c = cam.current
    const speed = c.size * 0.3 * (k.has('shift') ? 4 : 1) * dt
    const ahead = [Math.cos(c.lookYaw), Math.sin(c.lookYaw), 0]
    const side = [Math.sin(c.lookYaw), -Math.cos(c.lookYaw), 0]
    let way = [0, 0, 0]
    if (k.has('w') || k.has('arrowup')) way = add(way, ahead)
    if (k.has('s') || k.has('arrowdown')) way = sub(way, ahead)
    if (k.has('d') || k.has('arrowright')) way = add(way, side)
    if (k.has('a') || k.has('arrowleft')) way = sub(way, side)
    if (k.has('e') || k.has('pageup')) way = add(way, [0, 0, 1])
    if (k.has('q') || k.has('pagedown')) way = sub(way, [0, 0, 1])
    c.eye = add(c.eye, scale(normalise(way), length(way) ? speed : 0))
  }

  function step(time) {
    const c = cam.current
    if (!c?.walk || !keys.current.size) { walking.current = 0; return }
    const last = walking.current || time
    walking.current = time
    move(Math.min(0.1, (time - last) / 1000), keys.current)
    latest.current?.()
    requestAnimationFrame(step)
  }

  // While walking, the keys work wherever the focus is, except in a text box.
  useEffect(() => {
    if (!walk) return undefined
    function down(event) {
      const key = event.key.toLowerCase()
      const target = event.target
      if (target instanceof HTMLElement && (target.tagName === 'INPUT' || target.tagName === 'TEXTAREA'
        || target.tagName === 'SELECT' || target.isContentEditable)) return
      if (!(WALK_KEYS.has(key) || key === 'shift') || event.ctrlKey || event.metaKey || event.altKey) return
      event.preventDefault()
      const idle = !keys.current.size
      keys.current.add(key)
      // A tap moves a step at once; held, the frames carry on.
      if (key !== 'shift' && !event.repeat) { move(1 / 15, new Set([...keys.current])); draw() }
      if (idle && key !== 'shift') { walking.current = 0; requestAnimationFrame(step) }
    }
    function up(event) { keys.current.delete(event.key.toLowerCase()) }
    function blur() { keys.current.clear() }
    window.addEventListener('keydown', down)
    window.addEventListener('keyup', up)
    window.addEventListener('blur', blur)
    return () => {
      window.removeEventListener('keydown', down)
      window.removeEventListener('keyup', up)
      window.removeEventListener('blur', blur)
      keys.current.clear()
    }
  }, [walk]) // eslint-disable-line react-hooks/exhaustive-deps

  function onPointerDown(event) {
    if (!cam.current) return
    stageRef.current?.focus()
    event.currentTarget.setPointerCapture(event.pointerId)
    drag.current = {
      x: event.clientX, y: event.clientY, moved: false,
      pan: !walk && (event.button === 2 || event.shiftKey),
    }
  }

  function onPointerMove(event) {
    const d = drag.current
    if (!d) return
    const dx = event.clientX - d.x, dy = event.clientY - d.y
    if (Math.abs(dx) + Math.abs(dy) > 2) d.moved = true
    d.x = event.clientX; d.y = event.clientY
    const c = cam.current
    if (c.walk) {
      c.lookYaw -= dx * 0.004
      c.lookPitch = clampPitch(c.lookPitch - dy * 0.004)
    } else if (d.pan) {
      const yawRight = [-Math.sin(c.yaw), Math.cos(c.yaw), 0]
      const upish = [-Math.sin(c.pitch) * Math.cos(c.yaw), -Math.sin(c.pitch) * Math.sin(c.yaw), Math.cos(c.pitch)]
      const perPixel = c.distance * 0.9 / Math.max(stageRef.current.clientHeight, 1)
      c.target = add(c.target, add(scale(yawRight, -dx * perPixel), scale(upish, dy * perPixel)))
    } else {
      c.yaw -= dx * 0.008
      c.pitch = clampPitch(c.pitch + dy * 0.008)
    }
    draw()
  }

  function onPointerUp(event) {
    const d = drag.current
    drag.current = null
    if (d && !d.moved && picking && event.button === 0) pick(event)
  }

  function onWheel(event) {
    const c = cam.current
    if (!c) return
    if (c.walk) c.eye = add(c.eye, scale(direction(c.lookYaw, c.lookPitch), -event.deltaY * c.size * 0.0006))
    else c.distance *= Math.exp(event.deltaY * 0.0012)
    draw()
  }

  /** The point (or vertex) nearest the cursor on screen, the front one where several are close. */
  function pick(event) {
    const m = model.current
    if (!m) return
    const rect = canvasRef.current.getBoundingClientRect()
    const px = event.clientX - rect.left, py = event.clientY - rect.top
    const { mvp } = matrices()
    const w = rect.width, h = rect.height
    const positions = m.positions
    const radius = 8
    const box = crop && view ? [sub(crop.min, view.origin), sub(crop.max, view.origin)] : null
    let best = -1, bestDepth = Infinity, bestScreen = Infinity
    for (let i = 0; i < m.count; i++) {
      const x = positions[3 * i], y = positions[3 * i + 1], z = positions[3 * i + 2]
      if (box && (x < box[0][0] || y < box[0][1] || z < box[0][2] || x > box[1][0] || y > box[1][1] || z > box[1][2])) continue
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
  const defaultSize = shown === 'dense' ? 2 : 3

  return (
    <div className="viewer">
      <div className="viewer__toolbar">
        <div className="toolgroup" role="group" aria-label="What to show">
          {[['sparse', 'Sparse', true, 'The tie points the photos were placed with'],
            ['dense', 'Dense', !!dense, dense ? 'The dense point cloud' : 'Build the dense cloud first'],
            ['mesh', 'Mesh', !!mesh, mesh ? 'The surface mesh' : 'Build the mesh first']].map(([id, label, ok, title]) => (
            <button key={id} className={`toolgroup__btn ${shown === id ? 'toolgroup__btn--on' : ''}`}
                    onClick={() => setLayer(id)} disabled={!ok} aria-pressed={shown === id} title={title}>
              {label}
            </button>
          ))}
        </div>
        <div className="toolgroup" role="group" aria-label="How to move">
          <button className={`toolgroup__btn ${!walk ? 'toolgroup__btn--on' : ''}`} aria-pressed={!walk}
                  onClick={() => setWalking(false)} title="Turn the model round in front of you">
            Orbit
          </button>
          <button className={`toolgroup__btn ${walk ? 'toolgroup__btn--on' : ''}`} aria-pressed={walk}
                  onClick={() => setWalking(true)}
                  title="Walk through it: W A S D to move, Q and E down and up, drag to look">
            Walk
          </button>
        </div>
        <div className="toolgroup">
          <button className={`toolgroup__btn toolgroup__btn--switch ${showCameras ? 'toolgroup__btn--on' : ''}`}
                  aria-pressed={showCameras}
                  onClick={() => setShowCameras(!showCameras)} title="Show where each photo was taken">
            Cameras
          </button>
          {shown === 'mesh' && (
            <button className={`toolgroup__btn toolgroup__btn--switch ${lit ? 'toolgroup__btn--on' : ''}`}
                    aria-pressed={lit} onClick={() => setLit(!lit)}
                    title="Light the surface so its shape shows; off shows the photos' colours alone">
              Shading
            </button>
          )}
          <button className={`toolgroup__btn toolgroup__btn--switch ${picking ? 'toolgroup__btn--on' : ''}`}
                  aria-pressed={picking}
                  onClick={() => setPicking(!picking)}
                  title="Click points on the model to measure between them, or to set the scale">
            Pick points
          </button>
        </div>
        <div className="toolgroup">
          {shown !== 'mesh' && (
            <>
              <button className="toolgroup__btn" title="Smaller points"
                      onClick={() => setPointSize(Math.max(1, (pointSize ?? defaultSize) - 1))}>−</button>
              <button className="toolgroup__btn" title="Larger points"
                      onClick={() => setPointSize(Math.min(12, (pointSize ?? defaultSize) + 1))}>+</button>
            </>
          )}
          <button className="toolgroup__btn" title="Back to the whole model"
                  onClick={() => {
                    if (model.current) { cam.current = home(model.current.positions, view); setWalk(false); draw() }
                  }}>
            Fit
          </button>
        </div>
        {loaded && (
          <div className="toolgroup">
            <span className="toolgroup__btn toolgroup__btn--note">
              {loaded.which === 'mesh'
                ? `${loaded.triangles.toLocaleString()} triangles`
                : `${loaded.count.toLocaleString()} ${loaded.which} points`}
              {distance != null ? ` · picked points ${distance.toFixed(units === 'm' ? 3 : 4)} ${units} apart` : ''}
            </span>
          </div>
        )}
      </div>

      <div ref={stageRef} tabIndex={0}
           className={`viewer__stage scene3d ${picking ? 'viewer__stage--picking' : ''} ${walk ? 'scene3d--walk' : ''}`}
           onPointerDown={onPointerDown} onPointerMove={onPointerMove} onPointerUp={onPointerUp}
           onWheel={onWheel} onContextMenu={(e) => e.preventDefault()}>
        <canvas ref={canvasRef} className="scene3d__canvas" />
        {error && <div className="viewer__empty"><p>{error}</p></div>}
        {!error && !loaded && <div className="viewer__empty"><p>Loading the model…</p></div>}
        {walk && (
          <div className="scene3d__keys">
            <kbd>W</kbd><kbd>A</kbd><kbd>S</kbd><kbd>D</kbd> move · <kbd>Q</kbd><kbd>E</kbd> down, up ·
            {' '}<kbd>Shift</kbd> faster · drag to look
          </div>
        )}
      </div>

      {view && (
        <div className="viewer__legend">
          <span className="viewer__legend-item"><span className="scene3d__swatch scene3d__swatch--camera" />Camera</span>
          <span className="viewer__legend-item"><span className="scene3d__swatch scene3d__swatch--mark" />Picked point</span>
          {crop && <span className="viewer__legend-item"><span className="scene3d__swatch scene3d__swatch--crop" />Crop box</span>}
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
