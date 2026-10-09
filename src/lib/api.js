/**
 * Client for the processing engine.
 *
 * Two channels:
 *   - REST for commands and queries.
 *   - A WebSocket for everything the engine pushes: job progress, autosave
 *     confirmations, and project changes made by background work.
 *
 * The socket reconnects on its own with backoff. An engine restart should
 * cost the operator a flicker in the status strip, not a lost session.
 */

let basePort = 8731
let socket = null
let reconnectTimer = null
let reconnectDelay = 400
const listeners = new Set()

export function setPort(port) {
  if (port && port !== basePort) {
    basePort = port
    closeSocket()
    connect()
  }
}

export function baseUrl() {
  return `http://127.0.0.1:${basePort}`
}

/** URL for a raster tile. Used directly as an <img> src, so no fetch here. */
export function tileUrl(imageId, z, x, y, options = {}) {
  const params = new URLSearchParams()
  if (options.enhancement) params.set('enhancement', options.enhancement)
  if (options.bands?.length) params.set('bands', options.bands.join(','))
  if (options.gamma != null) params.set('gamma', String(options.gamma))
  if (options.colormap) params.set('colormap', options.colormap)
  return `${baseUrl()}/images/${imageId}/tile/${z}/${x}/${y}.png?${params}`
}

export function pathTileUrl(path, z, x, y, options = {}) {
  const params = new URLSearchParams({ path })
  // Outputs are rewritten in place; a new version keeps old tiles out of the cache.
  if (options.version) params.set('v', String(options.version))
  if (options.enhancement) params.set('enhancement', options.enhancement)
  if (options.bands?.length) params.set('bands', options.bands.join(','))
  if (options.gamma != null) params.set('gamma', String(options.gamma))
  if (options.colormap) params.set('colormap', options.colormap)
  return `${baseUrl()}/raster/tile/${z}/${x}/${y}.png?${params}`
}

export class EngineError extends Error {
  constructor(message, status, detail) {
    super(message)
    this.name = 'EngineError'
    this.status = status
    this.detail = detail
  }
}

async function request(method, path, body, options = {}) {
  const response = await fetch(`${baseUrl()}${path}`, {
    method,
    headers: body !== undefined ? { 'Content-Type': 'application/json' } : undefined,
    body: body !== undefined ? JSON.stringify(body) : undefined,
  })

  if (!response.ok) {
    let detail = response.statusText
    let storage = null
    const body = await response.text().catch(() => '')
    try {
      const payload = JSON.parse(body)
      detail = payload.detail ?? detail
      storage = payload.storage ?? null
    } catch {
      if (body) detail = body
    }
    const error = new EngineError(
      typeof detail === 'string' ? detail : JSON.stringify(detail),
      response.status,
      detail,
    )
    // Set when the failure was the disk rather than the request: a removed
    // drive, a full card, a read-only stick.
    error.storage = storage
    throw error
  }

  if (options.raw) return response.text()
  const type = response.headers.get('content-type') || ''
  return type.includes('application/json') ? response.json() : response.text()
}

export const api = {
  health: () => request('GET', '/health'),

  project: {
    create: (directory, name, description, author) =>
      request('POST', '/project/new', { directory, name, description, author }),
    open: (directory, author) => request('POST', '/project/open', { directory, author }),
    get: () => request('GET', '/project'),
    close: () => request('POST', '/project/close'),
    patch: (patch) => request('PATCH', '/project', patch),
    snapshot: (reason) => request('POST', '/project/snapshot', { reason }),
    snapshots: () => request('GET', '/project/snapshots'),
    restore: (path) => request('POST', '/project/restore', { path }),
    archive: (destination, includeOutputs) =>
      request('POST', '/project/archive', { destination, includeOutputs }),
    relink: (payload = {}) => request('POST', '/project/relink', payload),
    storage: (force = false) => request('GET', `/project/storage?force=${force}`),
    saveAs: (directory) => request('POST', '/project/save-as', { directory }),
  },

  projections: {
    list: () => request('GET', '/projections'),
    describe: (identifier) =>
      request('GET', `/projections/describe?identifier=${encodeURIComponent(identifier)}`),
  },

  reference: () => request('GET', '/reference'),

  camera: {
    set: (camera) => request('POST', '/camera', camera),
    distortionTable: (payload) => request('POST', '/camera/distortion-table', payload),
  },

  images: {
    add: (paths) => request('POST', '/images/add', { paths }),
    // The pixel of another photo that sees the same ground.
    transfer: (fromImageId, col, row, toImageId) =>
      request('POST', '/images/transfer', { fromImageId, col, row, toImageId }),
    remove: (id) => request('DELETE', `/images/${id}`),
    patch: (id, payload) => request('PATCH', `/images/${id}`, payload),
    info: (id) => request('GET', `/images/${id}/info`),
    setFiducial: (id, payload) => request('POST', `/images/${id}/fiducials`, payload),
  },

  raster: {
    info: (path) => request('GET', `/raster/info?path=${encodeURIComponent(path)}`),
    sample: (path, x, y) =>
      request('GET', `/raster/sample?path=${encodeURIComponent(path)}&x=${x}&y=${y}`),
  },

  points: {
    upsertGcp: (payload) => request('POST', '/points/gcp', payload),
    deleteGcp: (id) => request('DELETE', `/points/gcp/${id}`),
    deleteObservation: (pointId, imageId) =>
      request('DELETE', `/points/observation?pointId=${pointId}&imageId=${imageId}`),
    autoTie: (payload) => request('POST', '/points/tie/auto', payload),
    deleteTie: (id) => request('DELETE', `/points/tie/${id}`),
  },

  drone: {
    align: (payload = {}) => request('POST', '/drone/align', payload),
    dense: (payload = {}) => request('POST', '/drone/dense', payload),
  },

  // Automatic measurement. Everything here returns proposals, never
  // measurements — the operator accepts them explicitly.
  auto: {
    fiducials: (payload) => request('POST', '/auto/fiducials', payload),
    applyFiducials: (results) => request('POST', '/auto/fiducials/apply', { results }),
    gcps: (payload) => request('POST', '/auto/gcps', payload),
    applyGcps: (accepted, replaceAutomatic = false) =>
      request('POST', '/auto/gcps/apply', { accepted, replaceAutomatic }),
    // Measure existing control on every other photo that sees it.
    transferGcps: (pointIds) => request('POST', '/auto/gcps/transfer', { pointIds }),
    // One image, or several tiles of one orthomosaic searched as one.
    setReferenceImage: (paths) => request('POST', '/auto/reference-image',
      { paths: Array.isArray(paths) ? paths : [paths] }),
    place: (imageIds) => request('POST', '/auto/place', { imageIds }),
  },

  certificate: {
    status: () => request('GET', '/certificate/status'),
    // Direct to the engine: memory only, for running without the desktop
    // shell. The desktop app goes through window.fiducia.reader instead,
    // which also keeps the key encrypted between sessions.
    configure: (settings) => request('POST', '/certificate/settings', settings),
    read: (path) => request('POST', '/certificate/read', { path }),
  },

  terrain: {
    statistics: (path) => request('POST', '/terrain/statistics', { path }),
    merge: (payload) => request('POST', '/terrain/merge', payload),
    fill: (payload) => request('POST', '/terrain/fill', payload),
    smooth: (payload) => request('POST', '/terrain/smooth', payload),
    hillshade: (payload) => request('POST', '/terrain/hillshade', payload),
    contours: (payload) => request('POST', '/terrain/contours', payload),
    bareEarth: (payload) => request('POST', '/terrain/bare-earth', payload),
    volume: (surface, base, bounds) =>
      request('POST', '/terrain/volume', { surface, base, bounds }),
    profile: (path, line, samples) =>
      request('POST', '/terrain/profile', { path, line, samples }),
  },

  // Everything that crosses the boundary of a project: survey files, flight
  // logs, camera libraries, and layers for a GIS.
  exchange: {
    sniff: (path) => request('POST', '/exchange/sniff', { path }),
    importControl: (payload) => request('POST', '/exchange/control/import', payload),
    exportControl: (payload) => request('POST', '/exchange/control/export', payload),
    importExterior: (payload) => request('POST', '/exchange/exterior/import', payload),
    exportExterior: (payload) => request('POST', '/exchange/exterior/export', payload),
    footprints: (path) => request('POST', '/exchange/footprints', { path }),
    saveCamera: (path, label) => request('POST', '/exchange/camera/save', { path, label }),
    loadCamera: (path, label) => request('POST', '/exchange/camera/load', { path, label }),
  },

  model: {
    readiness: () => request('GET', '/model/readiness'),
    // Residuals for one photo from its own control; saves nothing.
    preview: (imageId) => request('POST', '/model/preview', { imageId }),
    compute: (payload = {}) => request('POST', '/model/compute', payload),
    setCheckPoint: (pointId, isCheckPoint) =>
      request('POST', '/model/checkpoint', { pointId, isCheckPoint }),
    residuals: (units = 'ground', show = 'all') =>
      request('GET', `/model/residuals?units=${units}&show=${show}`),
  },

  ortho: {
    generate: (payload) => request('POST', '/ortho/generate', payload),
    footprint: (payload) => request('POST', '/ortho/footprint', payload),
  },

  mosaic: {
    preview: (payload) => request('POST', '/mosaic/preview', payload),
    generate: (payload) => request('POST', '/mosaic/generate', payload),
  },

  dem: {
    pairs: () => request('POST', '/dem/pairs', {}),
    extract: (payload) => request('POST', '/dem/extract', payload),
  },

  lidar: {
    inspect: (path) => request('POST', '/lidar/inspect', { path }),
    rasterize: (payload) => request('POST', '/lidar/rasterize', payload),
    classifyGround: (payload) => request('POST', '/lidar/classify-ground', payload),
    height: (payload) => request('POST', '/lidar/height', payload),
    compare: (derived, reference) => request('POST', '/lidar/compare', { derived, reference }),
  },

  satellite: {
    rpc: (path) => request('POST', '/satellite/rpc', { path }),
    refine: (payload) => request('POST', '/satellite/refine', payload),
  },

  reports: {
    project: (payload = {}) => request('POST', '/reports/project', payload, { raw: !payload.savePath }),
    residual: (payload = {}) => request('POST', '/reports/residual', payload, { raw: !payload.savePath }),
  },

  jobs: {
    list: (activeOnly = false) => request('GET', `/jobs?activeOnly=${activeOnly}`),
    cancel: (id) => request('POST', `/jobs/${id}/cancel`),
    dismiss: (id) => request('POST', `/jobs/${id}/dismiss`),
    clear: () => request('POST', '/jobs/clear'),
  },
}

// -- live channel ----------------------------------------------------------

export function subscribe(listener) {
  listeners.add(listener)
  return () => listeners.delete(listener)
}

function emit(event) {
  listeners.forEach((listener) => {
    try { listener(event) } catch (error) { console.error('live listener failed', error) }
  })
}

function closeSocket() {
  if (reconnectTimer) { clearTimeout(reconnectTimer); reconnectTimer = null }
  if (socket) {
    socket.onclose = null
    socket.close()
    socket = null
  }
}

export function connect() {
  if (socket && (socket.readyState === WebSocket.OPEN || socket.readyState === WebSocket.CONNECTING)) {
    return
  }

  try {
    socket = new WebSocket(`ws://127.0.0.1:${basePort}/live`)
  } catch {
    scheduleReconnect()
    return
  }

  socket.onopen = () => {
    reconnectDelay = 400
    emit({ type: 'connection', connected: true })
    // A heartbeat keeps intermediaries from dropping an idle socket during a
    // long processing run where the engine has nothing to say.
    socket._ping = setInterval(() => {
      if (socket?.readyState === WebSocket.OPEN) socket.send('ping')
    }, 20000)
  }

  socket.onmessage = (event) => {
    try { emit(JSON.parse(event.data)) } catch { /* ignore malformed frames */ }
  }

  socket.onclose = () => {
    if (socket?._ping) clearInterval(socket._ping)
    socket = null
    emit({ type: 'connection', connected: false })
    scheduleReconnect()
  }

  socket.onerror = () => { socket?.close() }
}

function scheduleReconnect() {
  if (reconnectTimer) return
  reconnectTimer = setTimeout(() => {
    reconnectTimer = null
    reconnectDelay = Math.min(reconnectDelay * 1.6, 5000)
    connect()
  }, reconnectDelay)
}
