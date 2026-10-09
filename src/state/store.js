/**
 * Application state.
 *
 * The engine owns the project; this store is a cache of it plus purely local
 * concerns (which step is open, viewport position, theme). Any mutation goes
 * to the engine first and the returned state is adopted — so what is on screen
 * is always what was journalled to disk, and there is no way for the two to
 * drift.
 */

import { create } from 'zustand'
import { api, connect, subscribe, setPort } from '../lib/api'

function readIntroSeen() {
  try {
    // Only the main window introduces the app; pop-out image windows never do.
    if (new URLSearchParams(window.location.hash.slice(1)).has('popout')) return true
    // Launched with FIDUCIA_SHOW_INTRO=1 (see electron/main.js).
    if (new URLSearchParams(window.location.search).has('intro')) return false
    return localStorage.getItem('fiducia.introSeen') === '1'
  } catch {
    return true
  }
}

/** The operator's name, recorded as the author of their projects and outputs. */
function readAuthor() {
  try {
    return (localStorage.getItem('fiducia.author') || '').trim()
  } catch {
    return ''
  }
}

function isPopout() {
  try {
    return new URLSearchParams(window.location.hash.slice(1)).has('popout')
  } catch {
    return false
  }
}

// The name is asked for at the end of the introduction. Someone who has seen
// the introduction but has no name recorded is taken straight to that page.
const INTRO_LAST_PAGE = 2
const introSeen = readIntroSeen()
const needsAuthor = !isPopout() && !readAuthor()
const initialIntro = {
  introOpen: !introSeen || needsAuthor,
  introPage: introSeen && needsAuthor ? INTRO_LAST_PAGE : 0,
}

/** Whether the magnifier appears while measuring points; on unless turned off. */
function readLoupe() {
  try {
    return localStorage.getItem('fiducia.loupe') !== '0'
  } catch {
    return true
  }
}

function readRailOpen() {
  try {
    return localStorage.getItem('fiducia.railOpen') !== '0'
  } catch {
    return true
  }
}

export const STEPS = [
  {
    id: 'project',
    name: 'Project',
    detail: 'Setup',
    blurb: 'Math model, projection and output grid',
  },
  {
    id: 'camera',
    name: 'Camera',
    detail: 'Interior orientation',
    blurb: 'Interior orientation from the calibration certificate',
  },
  {
    id: 'images',
    name: 'Images',
    detail: 'Fiducials and clipping',
    blurb: 'Imagery, fiducial measurement and clip regions',
  },
  {
    id: 'points',
    name: 'Control',
    detail: 'GCPs and tie points',
    blurb: 'Ground control and tie points',
  },
  {
    id: 'model',
    name: 'Model',
    detail: 'Bundle adjustment',
    blurb: 'Bundle adjustment and residuals',
  },
  {
    id: 'ortho',
    name: 'Ortho',
    detail: 'Orthorectification',
    blurb: 'Orthorectified imagery from the solved model',
  },
  {
    id: 'mosaic',
    name: 'Mosaic',
    detail: 'Mosaicking',
    blurb: 'Cutlines, colour balancing and blending',
  },
  {
    id: 'dem',
    name: 'Stereo DEM',
    detail: 'Dense matching',
    blurb: 'Elevation extraction from stereo pairs',
  },
  {
    id: 'terrain',
    name: 'Terrain',
    detail: 'Editing and analysis',
    blurb: 'Repair, shaded relief, contours and volumes',
  },
  {
    id: 'lidar',
    name: 'LiDAR',
    detail: 'Point clouds',
    blurb: 'Point cloud rasterisation and validation',
  },
  {
    id: 'reports',
    name: 'Reports',
    detail: 'Residuals and QA',
    blurb: 'Project and residual reports',
  },
]

/**
 * What a project processes, chosen when it is created. It decides which
 * steps lead the rail, in what order and under what names -- a drone block
 * needs no calibration certificate and makes its surface before its
 * orthophotos. Nothing is taken away: every other step stays one click away
 * under "More tools", and the workflow can be changed on the Project step.
 */
export const WORKFLOWS = [
  {
    id: 'drone',
    name: 'Drone photos',
    hint: 'GPS-tagged photos from a drone. Camera, positions and tie points come from the photos.',
    mathModel: 'aerial_digital',
    steps: ['project', 'images', 'points', 'model', 'dem', 'ortho', 'mosaic', 'terrain', 'reports'],
    labels: {
      project: { detail: 'Setup' },
      images: { name: 'Photos', detail: 'Align from GPS', blurb: 'Add the photos and align them from their GPS' },
      points: { name: 'Control', detail: 'Optional GCPs', blurb: 'Ground control, if you have it: ties the block to the survey' },
      model: { name: 'Solve', detail: 'Bundle adjustment', blurb: 'Solve the block: orientations, lens and tie points together' },
      dem: { name: 'Surface', detail: 'Dense surface model', blurb: 'A dense surface model from overlapping photos' },
    },
  },
  {
    id: 'film',
    name: 'Scanned film photos',
    hint: 'Aerial film with fiducial marks and a calibration certificate.',
    mathModel: 'aerial_film',
    steps: ['project', 'camera', 'images', 'points', 'model', 'ortho', 'mosaic', 'dem', 'terrain', 'reports'],
    labels: {},
  },
  {
    id: 'satellite',
    name: 'Satellite scenes',
    hint: 'Scenes with rational polynomial coefficients, refined with control.',
    mathModel: 'satellite_rpc',
    steps: ['project', 'images', 'points', 'model', 'ortho', 'mosaic', 'dem', 'terrain', 'reports'],
    labels: {},
  },
  {
    id: 'all',
    name: 'Everything',
    hint: 'Every tool in the classic order, and any math model.',
    mathModel: null,
    steps: null,
    labels: {},
  },
]

/** The project's workflow; older projects are given the one they evidently are. */
export function workflowOf(project) {
  const id = project?.workflow
    || (project?.droneBlock ? 'drone'
      : project?.mathModel?.kind === 'satellite_rpc' ? 'satellite'
        : (project?.mathModel?.kind || 'aerial_film') === 'aerial_film' ? 'film'
          : 'all')
  return WORKFLOWS.find((w) => w.id === id) || WORKFLOWS[WORKFLOWS.length - 1]
}

/** Steps in this project's order and words: the workflow's, then the rest. */
export function stepsFor(project) {
  const workflow = workflowOf(project)
  const byId = Object.fromEntries(STEPS.map((step) => [step.id, step]))
  const order = workflow.steps || STEPS.map((step) => step.id)
  const primary = order.map((id) => ({ ...byId[id], ...(workflow.labels[id] || {}) }))
  const more = STEPS.filter((step) => !order.includes(step.id))
  return { workflow, primary, more, all: [...primary, ...more] }
}

const initialViewport = { zoom: 0, centerX: 0.5, centerY: 0.5, scale: 1 }

export const useStore = create((set, get) => ({
  // -- connection ------------------------------------------------------
  connected: false,
  engineReady: false,
  engineMessage: 'Starting engine…',
  engineFatal: false,

  // -- project ---------------------------------------------------------
  project: null,
  summary: null,
  loading: false,
  error: null,

  // -- local UI --------------------------------------------------------
  theme: localStorage.getItem('fiducia.theme') || 'dark',
  activeStep: 'project',
  activeImageId: null,
  referenceLayer: null,       // path of a geocoded reference image
  viewport: { ...initialViewport },
  inspectorOpen: true,
  railOpen: readRailOpen(),   // the step menu on the left; remembered per machine
  loupe: readLoupe(),         // magnifier while measuring points; remembered per machine
  pendingPoint: null,         // { imageId, col, row }: a clicked point not yet accepted
  stereoPair: null,           // { leftId, rightId }: the pair shown side by side in Stereo DEM
  lidarCloud: null,           // path of the point cloud open in the LiDAR step
  lidarSection: null,         // { start: [x, y], end: [x, y], width }: the side view's corridor
  lidarBrush: null,           // the class new labels get in the side view, or null to pan
  lidarColour: 'height',      // how the point cloud view colours points: height, class or labels
  lidarLabelsVersion: 0,      // bumped when labels change, so counts refresh
  viewRequest: null,          // { imageId, col, row, scale, token }: move any view of that photo there
  ...initialIntro,            // the first-launch introduction, and the page it opens on
  author: readAuthor(),
  paletteOpen: false,         // the command palette
  settingsOpen: false,        // the settings window
  settingsSection: null,      // the section it scrolls to when opened
  toasts: [],
  activity: [],               // messages shown in the activity bar, newest first

  // When a panel wants the operator to click a location on the image, it arms
  // a pick instead of opening a modal. The canvas stays live and in position
  // throughout — this is the mechanism that replaces the dialog maze.
  pickMode: null,             // { label, onPick(col, row, image), onCancel() }
  clipDraft: null,            // { imageId, col, row }: first corner of a clip being drawn
  selectedPointId: null,

  // -- reference data --------------------------------------------------
  projections: [],
  reference: null,

  // -- work ------------------------------------------------------------
  jobs: [],
  readiness: null,
  saveState: 'idle',          // idle | saving | saved
  lastSavedAt: null,
  storage: null,              // { kind, message, path } while the project folder can't be written

  // ====================================================================

  init: async () => {
    // The desktop shell tells us which port it started the engine on.
    if (window.fiducia?.engine) {
      const info = await window.fiducia.engine.info()
      if (info?.port) setPort(info.port)
      set({ engineReady: !!info?.ready })

      window.fiducia.engine.onStatus((status) => {
        set({
          engineReady: !!status.ready,
          engineMessage: status.message || '',
          engineFatal: !!status.fatal,
        })
        if (status.port) setPort(status.port)
        if (status.ready) get().refreshReference()
      })
    }

    subscribe((event) => get().handleLiveEvent(event))
    connect()

    get().linkWindows()

    // Watch the project folder. The engine checks only that it still exists
    // while all is well, so this is cheap; it is what notices a USB stick
    // pulled out between edits, before the next edit fails.
    setInterval(() => {
      if (get().project && get().connected) get().checkStorage()
    }, 3000)

    await get().refreshReference()
  },

  // -- windows across monitors ----------------------------------------------
  //
  // The main window owns every measurement: panels arm a pick with a handler
  // that knows what to do with the click. A popped-out window only mirrors
  // that state -- it shows the same prompt and crosshair -- and sends its
  // click back, where the main window's handler runs exactly as if the click
  // had happened on its own canvas.

  popoutImageId: null,        // set in a popped-out window: the image it shows
  poppedOut: [],              // in the main window: images currently in their own window

  linkWindows: () => {
    const bridge = window.fiducia?.windows
    if (!bridge) return
    const popout = get().popoutImageId

    if (popout) {
      bridge.onMessage((message) => {
        // An image window has nothing to show once its project is closed.
        if (message.type === 'project-closed') { window.close(); return }
        if (message.type === 'goto') {
          const { type, ...request } = message
          set({ viewRequest: request })
          return
        }
        if (message.type !== 'context') return
        set({
          activeStep: message.activeStep,
          selectedPointId: message.selectedPointId,
          pickMode: message.pick
            ? {
                label: message.pick,
                remote: true,
                onPick: (col, row, image) =>
                  bridge.relay({ type: 'pick', col, row, imageId: image.id }),
                onCancel: () => bridge.relay({ type: 'pick-cancel' }),
              }
            : null,
        })
      })
      bridge.relay({ type: 'hello' })
      return
    }

    const announce = () => {
      const { activeStep, selectedPointId, pickMode } = get()
      bridge.relay({ type: 'context', activeStep, selectedPointId,
                     pick: pickMode?.label || null })
    }

    bridge.onMessage((message) => {
      if (message.type === 'hello') announce()
      else if (message.type === 'popouts') set({ poppedOut: message.open || [] })
      else if (message.type === 'pick') {
        const { pickMode, project } = get()
        const image = project?.images?.find((i) => i.id === message.imageId)
        if (pickMode && image) pickMode.onPick(message.col, message.row, image)
      } else if (message.type === 'pick-cancel') {
        get().pickMode?.onCancel?.()
      }
    })

    useStore.subscribe((state, previous) => {
      if (state.pickMode !== previous.pickMode
          || state.activeStep !== previous.activeStep
          || state.selectedPointId !== previous.selectedPointId) announce()
    })

    bridge.list().then((open) => set({ poppedOut: open || [] })).catch(() => {})
  },

  popOut: async (image) => {
    const bridge = window.fiducia?.windows
    if (!bridge || !image) return
    await bridge.popout(image.id, image.name)
  },

  checkStorage: async (force = false) => {
    try {
      const status = await api.project.storage(force)
      get().setStorage(status.ok ? null : status)
      return status.ok
    } catch (error) {
      if (error.storage) get().setStorage(error.storage)
      return false
    }
  },

  setStorage: (next) => {
    const previous = get().storage
    if (!previous && !next) return
    if (previous && next && previous.message === next.message) return
    set({ storage: next })
    if (previous && !next) {
      get().toast('The project folder is available. Saving has resumed.', 'good')
      get().refresh()
    }
  },

  saveProjectElsewhere: async () => {
    const folder = await window.fiducia?.dialog.openDirectory({
      title: 'Choose where to save the project',
    })
    if (!folder) return false
    const name = get().project?.name || 'Project'
    const safe = name.replace(/[\\/:*?"<>|]/g, '').trim() || 'Project'
    const separator = folder.endsWith('\\') || folder.endsWith('/') ? '' : '\\'
    try {
      const payload = await api.project.saveAs(`${folder}${separator}${safe}`)
      set({ project: payload.state, summary: payload.summary, storage: null })
      get().toast(`Project saved to ${payload.summary.directory}. Work continues from this location.`, 'good', 9000)
      get().refreshReadiness()
      return true
    } catch (error) {
      get().toast(error.message, 'bad', 12000)
      return false
    }
  },

  handleLiveEvent: (event) => {
    switch (event.type) {
      case 'connection':
        set({ connected: event.connected })
        if (event.connected) get().refresh()
        break

      case 'job': {
        const incoming = event.job
        set((state) => {
          const existing = state.jobs.findIndex((j) => j.id === incoming.id)
          const jobs = [...state.jobs]
          if (existing >= 0) jobs[existing] = incoming
          else jobs.unshift(incoming)
          return { jobs: jobs.slice(0, 60) }
        })

        // The job's own card in the activity bar reports completion and
        // failure, so no separate message is raised for either.
        if (incoming.status === 'done') get().refresh()
        break
      }

      case 'project.changed':
        set({ saveState: 'saving' })
        get().refresh()
        break

      case 'project.saved':
        set({ saveState: 'saved', lastSavedAt: Date.now() })
        // Fade the indicator back to idle so it reads as a moment, not a mode.
        setTimeout(() => {
          if (get().saveState === 'saved') set({ saveState: 'idle' })
        }, 2200)
        break

      case 'project.storage':
        get().setStorage(event.ok ? null : event)
        break

      case 'project.opened':
        set({ summary: event.summary, storage: event.summary?.storage || null })
        get().refresh()
        break

      default:
        break
    }
  },

  refreshReference: async () => {
    try {
      const [projections, reference] = await Promise.all([
        api.projections.list(),
        api.reference(),
      ])
      set({ projections: projections.presets, reference })
    } catch {
      // Engine not up yet; the status handler will retry once it is.
    }
  },

  refresh: async () => {
    try {
      const payload = await api.project.get()
      set({ project: payload.state, summary: payload.summary, error: null })
      get().refreshReadiness()
    } catch (error) {
      if (error.status !== 409) set({ error: error.message })
      else set({ project: null, summary: null })
    }
  },

  refreshReadiness: async () => {
    try {
      set({ readiness: await api.model.readiness() })
    } catch {
      set({ readiness: null })
    }
  },

  // -- project lifecycle ------------------------------------------------

  createProject: async (directory, name, description, workflowId = 'all') => {
    set({ loading: true, error: null })
    try {
      const payload = await api.project.create(directory, name, description, get().author)
      const workflow = WORKFLOWS.find((w) => w.id === workflowId) || WORKFLOWS[WORKFLOWS.length - 1]
      // A drone project starts on its photos: its projection, camera and
      // output grid all come from them.
      set({ project: payload.state, summary: payload.summary,
            activeStep: workflow.id === 'drone' ? 'images' : 'project',
            outputView: null, mosaicPreview: null })
      const patch = { workflow: workflow.id }
      if (workflow.mathModel) {
        patch.mathModel = { ...(payload.state?.mathModel || {}), kind: workflow.mathModel }
      }
      // Start from the coordinate system used last on this computer; most
      // people work in the same zone project after project. Not for a drone
      // block, which is placed in the UTM zone its GPS says it is in.
      let last = ''
      try { last = localStorage.getItem('fiducia.lastProjection') || '' } catch { /* optional */ }
      if (last && workflow.id !== 'drone' && !payload.state?.projection?.output) {
        patch.projection = { ...(payload.state?.projection || {}), output: last, gcpSource: last }
      }
      await get().patchProject(patch)
      get().toast(`Project "${name}" created`, 'good')
      get().refreshReadiness()
      return true
    } catch (error) {
      set({ error: error.message })
      get().toast(error.message, 'bad', 10000)
      return false
    } finally {
      set({ loading: false })
    }
  },

  openProject: async (directory) => {
    set({ loading: true, error: null })
    try {
      const payload = await api.project.open(directory, get().author)
      set({ project: payload.state, summary: payload.summary, outputView: null, mosaicPreview: null })

      const recovered = payload.state?._recoveredOperations || 0
      if (recovered > 0) {
        get().toast(
          `${recovered} unsaved change${recovered === 1 ? '' : 's'} recovered from the journal`,
          'accent',
          9000,
        )
      }

      const offline = (payload.links || []).filter((l) => !l.online)
      if (offline.length) {
        get().toast(
          `${offline.length} image${offline.length === 1 ? '' : 's'} offline. Relink in the Images step.`,
          'warn',
          12000,
        )
      }

      get().refreshReadiness()
      return true
    } catch (error) {
      set({ error: error.message })
      get().toast(error.message, 'bad', 10000)
      return false
    } finally {
      set({ loading: false })
    }
  },

  // Back to the start screen. The engine flushes the project to disk as it
  // closes, so nothing is lost; open image windows close with it.
  closeProject: async () => {
    const name = get().project?.name
    try {
      await api.project.close()
    } catch (error) {
      get().toast(error.message || 'The project could not be closed', 'bad', 10000)
      return false
    }
    window.fiducia?.windows?.relay({ type: 'project-closed' })
    set({
      project: null,
      summary: null,
      activeImageId: null,
      activeStep: 'project',
      readiness: null,
      storage: null,
      selectedPointId: null,
      pickMode: null,
      clipDraft: null,
      pendingPoint: null,
      stereoPair: null,
      lidarCloud: null,
      lidarSection: null,
      lidarBrush: null,
      outputView: null,
      mosaicPreview: null,
      poppedOut: [],
      referenceLayer: null,
    })
    if (name) get().toast(`${name} closed`, 'info')
    return true
  },

  patchProject: async (patch) => {
    try {
      const payload = await api.project.patch(patch)
      set({ project: payload.state, summary: payload.summary })
      get().refreshReadiness()
    } catch (error) {
      if (error.storage) get().setStorage(error.storage)
      get().toast(error.message, 'bad', 10000)
    }
  },

  // -- generic engine call with error surfacing -------------------------

  /**
   * Run an engine call, adopt any returned project state, and surface
   * failures as a toast rather than an unhandled rejection. Every panel
   * action goes through this, which is why no panel has its own try/catch.
   */
  call: async (fn, { successMessage, refresh = true } = {}) => {
    try {
      const result = await fn()
      if (result?.state) set({ project: result.state })
      if (refresh) get().refresh()
      if (successMessage) get().toast(successMessage, 'good')
      return result
    } catch (error) {
      if (error.storage) get().setStorage(error.storage)
      get().toast(error.message || String(error), 'bad', 10000)
      return null
    }
  },

  // -- local UI ---------------------------------------------------------

  setStep: (activeStep) => {
    set({ activeStep })
    const { project } = get()
    if (project) api.project.patch({ ui: { ...project.ui, activeStep } }).catch(() => {})
  },

  setActiveImage: (activeImageId) =>
    set({ activeImageId, viewport: { ...initialViewport } }),

  setViewport: (viewport) => set({ viewport }),
  setSelectedPoint: (selectedPointId) => set({ selectedPointId }),

  /**
   * Arm a pick. Returns a disarm function, and the pick disarms itself after
   * firing unless `sticky` is set — which is what makes collecting a run of
   * fiducials or tie points feel like one continuous action rather than a
   * dialog reopened eight times.
   */
  armPick: ({ label, onPick, onCancel, sticky = false, loupe = false }) => {
    const disarm = () => set({ pickMode: null })
    const mode = {
      label,
      loupe,             // offer the magnifier: a precise point is being measured
      onCancel: () => {
        disarm()
        onCancel?.()
      },
      onPick: (col, row, image) => {
        onPick(col, row, image)
        // A pick that arms the next one (the second corner of a clip region)
        // has already replaced this mode; disarming now would cancel it.
        if (!sticky && get().pickMode === mode) disarm()
      },
    }
    set({ pickMode: mode })
    return disarm
  },

  cancelPick: () => set({ pickMode: null }),
  setClipDraft: (clipDraft) => set({ clipDraft }),
  setPendingPoint: (pendingPoint) => set({ pendingPoint }),
  setStereoPair: (stereoPair) => set({ stereoPair }),
  // A cloud made from the open one covers the same ground, so the section stays.
  setLidarCloud: (lidarCloud, { keepSection = false } = {}) =>
    set((state) => ({ lidarCloud, lidarSection: keepSection ? state.lidarSection : null })),
  setLidarColour: (lidarColour) => set({ lidarColour }),
  setLidarSection: (lidarSection) => set({ lidarSection }),
  setLidarBrush: (lidarBrush) => set({ lidarBrush }),
  bumpLidarLabels: () => set((state) => ({ lidarLabelsVersion: state.lidarLabelsVersion + 1 })),

  // -- outputs on the canvas --------------------------------------------
  //
  // A generated raster (mosaic, ortho, DEM, surface) shown in place of the
  // photos while its step is open; the live mosaic preview; and a request to
  // scroll a step's list of outputs into view.
  outputView: null,           // { path, label, step }
  mosaicPreview: null,        // the latest preview payload from the Mosaic panel
  mosaicSeams: false,         // seam map over the preview
  revealRequest: null,        // { step, token }: scroll that step's outputs into view

  showOutput: (output) => set({ outputView: { ...output, version: Date.now() } }),
  closeOutput: () => set({ outputView: null }),
  setMosaicPreview: (mosaicPreview) => set({ mosaicPreview }),
  setMosaicSeams: (mosaicSeams) => set({ mosaicSeams }),
  revealOutputs: (step) => {
    get().setStep(step)
    set({ revealRequest: { step, token: Date.now() + Math.random() } })
  },
  popOutOutput: async (output) => {
    const bridge = window.fiducia?.windows
    if (!bridge || !output?.path) return
    await bridge.popout(`output:${output.path}`, output.label || output.path.split(/[\\/]/).pop())
  },

  // Centre every view of a photo on a pixel, at a zoom, in this window and
  // any popped-out one showing it.
  goTo: (imageId, col, row, scale) => {
    const request = { imageId, col, row, scale, token: Date.now() + Math.random() }
    set({ viewRequest: request })
    window.fiducia?.windows?.relay({ type: 'goto', ...request })
  },
  setLoupe: (loupe) => {
    try { localStorage.setItem('fiducia.loupe', loupe ? '1' : '0') } catch { /* optional */ }
    set({ loupe })
  },
  setReferenceLayer: (referenceLayer) => set({ referenceLayer }),
  toggleInspector: () => set((s) => ({ inspectorOpen: !s.inspectorOpen })),
  setIntroOpen: (introOpen, introPage = 0) => {
    if (!introOpen) {
      try { localStorage.setItem('fiducia.introSeen', '1') } catch { /* optional */ }
    }
    set({ introOpen, introPage })
  },

  // The name is kept on this computer and sent with project and ortho
  // requests. An open project that has no author yet takes it at once.
  setAuthor: (name) => {
    const author = (name || '').trim()
    try { localStorage.setItem('fiducia.author', author) } catch { /* optional */ }
    set({ author })
    const project = get().project
    if (author && project && !(project.author || '').trim()) {
      get().patchProject({ author })
    }
  },
  toggleRail: () => set((s) => {
    const railOpen = !s.railOpen
    try { localStorage.setItem('fiducia.railOpen', railOpen ? '1' : '0') } catch { /* optional */ }
    return { railOpen }
  }),
  setPaletteOpen: (paletteOpen) => set({ paletteOpen }),
  setSettingsOpen: (settingsOpen, settingsSection = null) =>
    set({ settingsOpen, settingsSection }),

  setTheme: (theme) => {
    localStorage.setItem('fiducia.theme', theme)
    document.documentElement.setAttribute('data-theme', theme)
    set({ theme })
  },

  // Every message is recorded in the activity bar. The same message repeated
  // within a few seconds updates the existing event instead of stacking.
  // Pop-up toasts are still kept for pop-out windows, which have no bar.
  toast: (message, tone = 'info', duration = 5000) => {
    const id = Math.random().toString(36).slice(2)
    const at = Date.now()
    set((state) => {
      const [latest, ...rest] = state.activity
      const activity = latest && latest.message === message && latest.tone === tone
        && at - latest.at < 10000
        ? [{ ...latest, at, count: (latest.count || 1) + 1 }, ...rest]
        : [{ id, message, tone, at }, ...state.activity].slice(0, 40)
      return { activity, toasts: [...state.toasts, { id, message, tone }] }
    })
    setTimeout(() => {
      set((state) => ({ toasts: state.toasts.filter((t) => t.id !== id) }))
    }, duration)
  },

  dismissToast: (id) =>
    set((state) => ({ toasts: state.toasts.filter((t) => t.id !== id) })),

  dismissActivity: (id) =>
    set((state) => ({ activity: state.activity.filter((a) => a.id !== id) })),

  // The engine only broadcasts job updates, never removals, so the local list
  // is pruned here directly; the engine call just keeps /jobs in step.
  dismissJob: (id) => {
    set((state) => ({ jobs: state.jobs.filter((j) => j.id !== id) }))
    api.jobs.dismiss(id).catch(() => {})
  },

  clearFinishedJobs: () => {
    set((state) => ({
      jobs: state.jobs.filter((j) => j.status === 'running' || j.status === 'queued'),
      activity: [],
    }))
    api.jobs.clear().catch(() => {})
  },
}))

// -- derived selectors -----------------------------------------------------

export const selectActiveImage = (state) =>
  state.project?.images?.find((i) => i.id === state.activeImageId) || null

export const selectOnlineImages = (state) =>
  state.project?.images?.filter((i) => i.online) || []

export const selectActiveJobs = (state) =>
  state.jobs.filter((j) => j.status === 'running' || j.status === 'queued')

export const selectObservationsFor = (state, imageId) =>
  state.project?.observations?.filter((o) => o.imageId === imageId) || []

/**
 * Per-step completion, used to light the step rail.
 *
 * Returns 'done' | 'attention' | 'ready' | 'locked' so the operator can see at
 * a glance where the block actually stands — which is the whole point of
 * replacing a dropdown with a rail.
 */
export function stepStatus(state, stepId) {
  const project = state.project
  if (!project) return stepId === 'project' ? 'ready' : 'locked'

  const camera = project.camera || {}
  const projection = project.projection || {}
  const images = project.images || []
  const online = images.filter((i) => i.online)
  const gcps = project.gcps || []
  const completeGcps = gcps.filter(
    (g) => !g.isCheckPoint && g.x != null && g.y != null && g.z != null,
  )
  const isFilm = (project.mathModel?.kind || 'aerial_film') === 'aerial_film'
  // A drone block (or imported GNSS/IMU) is held by its photo positions, so
  // ground control is welcome but not required to solve it.
  const gnssHeld = online.filter((i) => i.exteriorObserved).length >= 3
    && (project.tiePoints || []).length > 0
  const drone = workflowOf(project).id === 'drone'

  if (drone) {
    // The camera, positions and tie points all come from the photos, so the
    // chain is: photos aligned, block solved, surface, orthos.
    const block = project.droneBlock
    switch (stepId) {
      case 'images':
        if (!images.length) return 'ready'
        if (images.some((i) => !i.online)) return 'attention'
        return block ? 'done' : 'ready'
      case 'points':
        if (!online.length) return 'locked'
        if (completeGcps.length >= 3) return 'done'
        return completeGcps.length > 0 ? 'attention' : 'ready'
      case 'dem':
        if (!project.model || !block?.surface) return 'locked'
        if (!block.dense) return 'ready'
        return block.dense.stale ? 'attention' : 'done'
      default:
        break
    }
  }

  switch (stepId) {
    case 'project':
      return projection.output ? 'done' : 'ready'

    case 'camera':
      if (!camera.focalMm) return 'ready'
      if (isFilm && Object.keys(camera.fiducialsMm || {}).length < 4) return 'attention'
      return 'done'

    case 'images':
      if (!images.length) return camera.focalMm ? 'ready' : 'locked'
      if (images.some((i) => !i.online)) return 'attention'
      if (isFilm && online.some((i) => !i.fiducialFit)) return 'attention'
      return 'done'

    case 'points':
      if (!online.length) return 'locked'
      if (completeGcps.length >= 3) return 'done'
      if (completeGcps.length > 0) return 'attention'
      return 'ready'

    case 'model':
      if (completeGcps.length < 3 && !gnssHeld) return 'locked'
      if (!project.model) return 'ready'
      return project.model.converged ? 'done' : 'attention'

    case 'ortho':
      if (!project.model) return 'locked'
      if ((project.orthos || []).length) return 'done'
      return 'ready'

    case 'mosaic':
      if ((project.orthos || []).length < 2) return 'locked'
      if ((project.mosaics || []).length) return 'done'
      return 'ready'

    case 'dem':
      if (!project.model || online.length < 2) return 'locked'
      if ((project.stereoDems || []).length) return 'done'
      return 'ready'

    case 'terrain': {
      const surfaces =
        (project.stereoDems || []).length +
        (project.lidar || []).length +
        (project.surfaces || []).length +
        (project.dem?.referencePath ? 1 : 0)
      if (!surfaces) return 'locked'
      return (project.surfaces || []).length ? 'done' : 'ready'
    }

    case 'lidar':
      return (project.lidar || []).length ? 'done' : 'ready'

    case 'reports':
      return project.model ? 'ready' : 'locked'

    default:
      return 'ready'
  }
}
