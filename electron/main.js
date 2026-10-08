/**
 * Fiducia — Electron main process.
 *
 * Responsibilities, in order of how much they matter:
 *
 *  1. Own the lifecycle of the processing engine (the Python sidecar). It is
 *     started on launch, health-checked before the window is told it is ready,
 *     restarted if it dies, and killed on quit. A crashed engine shows as a
 *     banner in the interface, not a frozen window.
 *  2. Provide native file dialogs, so the operator browses their disk with the
 *     OS picker rather than an in-page file input.
 *  3. Keep the renderer sandboxed. No Node integration, context isolation on,
 *     and a narrow preload bridge — the renderer only gets the handful of
 *     operations listed in preload.js.
 */

import { app, BrowserWindow, dialog, ipcMain, shell, Menu, safeStorage, screen } from 'electron'
import { spawn } from 'node:child_process'
import { cpSync, existsSync, mkdirSync, readFileSync, writeFileSync, renameSync } from 'node:fs'
import path from 'node:path'
import { fileURLToPath } from 'node:url'
import net from 'node:net'

const __dirname = path.dirname(fileURLToPath(import.meta.url))
const isDev = process.env.FIDUCIA_DEV === '1'
const projectRoot = path.resolve(__dirname, '..')

let mainWindow = null
let engineProcess = null
let enginePort = 0
let engineReady = false
let engineRestarts = 0
let quitting = false

const MAX_RESTARTS = 4

/** Find a free TCP port so two copies of Fiducia never collide. */
function findFreePort() {
  return new Promise((resolve, reject) => {
    const server = net.createServer()
    server.unref()
    server.on('error', reject)
    server.listen(0, '127.0.0.1', () => {
      const { port } = server.address()
      server.close(() => resolve(port))
    })
  })
}

/** Locate the Python interpreter that has the engine's dependencies. */
function resolvePython() {
  const candidates = [
    process.env.FIDUCIA_PYTHON,
    path.join(process.env.LOCALAPPDATA || '', 'Fiducia', 'venv', 'Scripts', 'python.exe'),
    // Environments set up before the rename to Fiducia.
    path.join(process.env.LOCALAPPDATA || '', 'Fiduplanus', 'venv', 'Scripts', 'python.exe'),
    path.join(projectRoot, '.venv', 'Scripts', 'python.exe'),
    path.join(projectRoot, '.venv', 'bin', 'python'),
  ].filter(Boolean)

  for (const candidate of candidates) {
    if (existsSync(candidate)) return candidate
  }
  return process.platform === 'win32' ? 'python' : 'python3'
}

async function waitForEngine(port, timeoutMs = 60000) {
  const deadline = Date.now() + timeoutMs
  while (Date.now() < deadline) {
    try {
      const response = await fetch(`http://127.0.0.1:${port}/health`)
      if (response.ok) return true
    } catch {
      // Engine not up yet — expected during the first second or two.
    }
    await new Promise((r) => setTimeout(r, 250))
  }
  return false
}

function notifyRenderer(channel, payload) {
  // Every window, not just the main one: a popped-out image needs to know
  // when the engine restarts on a new port as much as the main window does.
  for (const window of BrowserWindow.getAllWindows()) {
    if (!window.isDestroyed()) window.webContents.send(channel, payload)
  }
}

/**
 * How to start the engine. An installed copy carries its own frozen engine
 * (fiducia-engine.exe, built by packaging/build.ps1) so the operator never needs
 * Python; a development checkout runs server.py with the local environment.
 */
function engineCommand() {
  if (app.isPackaged) {
    const folder = path.join(process.resourcesPath, 'engine')
    return { command: path.join(folder, 'fiducia-engine.exe'), args: [], cwd: folder }
  }
  return {
    command: resolvePython(),
    args: [path.join(projectRoot, 'engine', 'server.py')],
    cwd: projectRoot,
  }
}

async function startEngine() {
  enginePort = await findFreePort()
  const { command, args, cwd } = engineCommand()

  engineProcess = spawn(command, [...args, '--port', String(enginePort)], {
    cwd,
    env: { ...process.env, PYTHONUNBUFFERED: '1', PYTHONIOENCODING: 'utf-8' },
    stdio: ['ignore', 'pipe', 'pipe'],
    windowsHide: true,
  })

  engineProcess.stdout.on('data', (chunk) => {
    const text = chunk.toString()
    if (isDev) process.stdout.write(`[engine] ${text}`)
    notifyRenderer('engine:log', { stream: 'stdout', text })
  })

  engineProcess.stderr.on('data', (chunk) => {
    const text = chunk.toString()
    if (isDev) process.stderr.write(`[engine] ${text}`)
    notifyRenderer('engine:log', { stream: 'stderr', text })
  })

  engineProcess.on('exit', (code, signal) => {
    engineReady = false
    engineProcess = null
    if (quitting) return

    notifyRenderer('engine:status', {
      ready: false,
      code,
      signal,
      message: `Engine exited (${code ?? signal}). Restarting…`,
    })

    if (engineRestarts < MAX_RESTARTS) {
      engineRestarts += 1
      setTimeout(() => { startEngine().catch(() => {}) }, 800)
    } else {
      notifyRenderer('engine:status', {
        ready: false,
        fatal: true,
        message: app.isPackaged
          ? 'The Fiducia engine could not stay running. Restart the app; if ' +
            'it happens again, reinstall Fiducia.'
          : 'The Fiducia engine could not stay running. Check that the Python ' +
            'environment is installed, then restart the app.',
      })
    }
  })

  engineProcess.on('error', (error) => {
    notifyRenderer('engine:status', {
      ready: false,
      fatal: true,
      message: `Could not launch the engine: ${error.message}`,
    })
  })

  engineReady = await waitForEngine(enginePort)
  // Before the window hears the engine is ready, so the certificate section
  // never flashes "add a key" for a key that is already saved.
  if (engineReady) await pushReaderSettings()
  notifyRenderer('engine:status', {
    ready: engineReady,
    port: enginePort,
    message: engineReady ? 'Engine ready' : 'Engine did not respond in time',
  })
  return engineReady
}

function createWindow() {
  mainWindow = new BrowserWindow({
    width: 1680,
    height: 1000,
    minWidth: 1100,
    minHeight: 700,
    show: false,
    backgroundColor: '#0b0d0c',
    title: 'Fiducia',
    icon: path.join(projectRoot, 'resources', 'icon.png'),
    autoHideMenuBar: true,
    webPreferences: {
      preload: path.join(__dirname, 'preload.cjs'),
      contextIsolation: true,
      nodeIntegration: false,
      sandbox: false,
    },
  })

  mainWindow.once('ready-to-show', () => mainWindow.show())

  // FIDUCIA_SHOW_INTRO=1 plays the first-launch introduction again, for
  // demonstrations, however many times it has already been seen.
  const search = process.env.FIDUCIA_SHOW_INTRO === '1' ? 'intro' : ''
  if (isDev) {
    mainWindow.loadURL(`http://127.0.0.1:5273/${search ? `?${search}` : ''}`)
  } else {
    mainWindow.loadFile(path.join(projectRoot, 'dist', 'index.html'), { search })
  }

  // External links open in the real browser, never inside the app shell.
  mainWindow.webContents.setWindowOpenHandler(({ url }) => {
    shell.openExternal(url)
    return { action: 'deny' }
  })

  mainWindow.on('closed', () => {
    mainWindow = null
    // Popped-out images belong to this window; without it they would keep the
    // app alive with nothing to report to.
    for (const window of popouts.values()) {
      if (!window.isDestroyed()) window.close()
    }
  })
}

// -- popped-out images -----------------------------------------------------
//
// Any image can be torn off into its own window and put on another monitor.
// Each window is a full renderer on the same engine, so it stays in sync with
// the project by itself; the only thing relayed between windows is what the
// operator is doing -- which measurement is armed, which step is open -- so a
// click in a popped-out window completes a measurement started in the main one.

const popouts = new Map()   // imageId -> BrowserWindow

function popoutBoundsFile() {
  return path.join(app.getPath('userData'), 'popout-windows.json')
}

function loadPopoutBounds() {
  try {
    return JSON.parse(readFileSync(popoutBoundsFile(), 'utf-8'))
  } catch {
    return {}
  }
}

function savePopoutBounds(imageId, bounds) {
  const all = loadPopoutBounds()
  all[imageId] = bounds
  try {
    writeFileSync(popoutBoundsFile(), JSON.stringify(all), 'utf-8')
  } catch {
    // Placement is a convenience; failing to remember it is not an error.
  }
}

/** Where a new popped-out window should go: on a monitor the main window isn't. */
function placeNewPopout() {
  const displays = screen.getAllDisplays()
  const home = mainWindow ? screen.getDisplayMatching(mainWindow.getBounds()) : displays[0]
  const others = displays.filter((display) => display.id !== home.id)
  // Spread across the other monitors first, then cascade.
  const target = others.length ? others[popouts.size % others.length] : home
  const area = target.workArea
  const offset = (others.length ? Math.floor(popouts.size / others.length) : popouts.size) * 36
  if (others.length && offset === 0) {
    return { x: area.x, y: area.y, width: area.width, height: area.height, maximise: true }
  }
  const width = Math.round(area.width * (others.length ? 0.9 : 0.55))
  const height = Math.round(area.height * (others.length ? 0.9 : 0.7))
  return {
    x: area.x + Math.round((area.width - width) / 2) + offset,
    y: area.y + Math.round((area.height - height) / 2) + offset,
    width,
    height,
  }
}

/** A remembered position is only reused if that monitor is still attached. */
function visibleOnSomeDisplay(bounds) {
  return screen.getAllDisplays().some(({ workArea: a }) =>
    bounds.x < a.x + a.width - 80 && bounds.x + bounds.width > a.x + 80 &&
    bounds.y < a.y + a.height - 80 && bounds.y + bounds.height > a.y + 40)
}

function openPopout(imageId, name) {
  const existing = popouts.get(imageId)
  if (existing && !existing.isDestroyed()) {
    if (existing.isMinimized()) existing.restore()
    existing.focus()
    return { opened: false, focused: true }
  }

  const remembered = loadPopoutBounds()[imageId]
  const placement = remembered && visibleOnSomeDisplay(remembered) ? remembered : placeNewPopout()

  const window = new BrowserWindow({
    x: placement.x,
    y: placement.y,
    width: placement.width,
    height: placement.height,
    minWidth: 480,
    minHeight: 360,
    show: false,
    backgroundColor: '#0b0d0c',
    title: `${name || 'Image'} — Fiducia`,
    autoHideMenuBar: true,
    webPreferences: {
      preload: path.join(__dirname, 'preload.cjs'),
      contextIsolation: true,
      nodeIntegration: false,
      sandbox: false,
    },
  })

  window.once('ready-to-show', () => {
    if (placement.maximise || remembered?.maximised) window.maximize()
    window.show()
  })

  const hash = `popout=${encodeURIComponent(imageId)}`
  if (isDev) {
    window.loadURL(`http://127.0.0.1:5273/#${hash}`)
  } else {
    window.loadFile(path.join(projectRoot, 'dist', 'index.html'), { hash })
  }

  window.webContents.setWindowOpenHandler(({ url }) => {
    shell.openExternal(url)
    return { action: 'deny' }
  })

  let current = imageId
  const remember = () => {
    if (window.isDestroyed()) return
    savePopoutBounds(current, { ...window.getNormalBounds(), maximised: window.isMaximized() })
  }
  window.on('close', remember)
  window.on('closed', () => {
    if (popouts.get(current) === window) popouts.delete(current)
    relay(null, { type: 'popouts', open: [...popouts.keys()] })
  })

  // The operator can switch a popped-out window to a different image.
  window.fiduciaRetarget = (next) => {
    if (popouts.get(current) === window) popouts.delete(current)
    current = next
    popouts.set(next, window)
  }

  popouts.set(imageId, window)
  relay(null, { type: 'popouts', open: [...popouts.keys()] })
  return { opened: true }
}

/** Pass a message to every window except the one it came from. */
function relay(sender, message) {
  for (const window of BrowserWindow.getAllWindows()) {
    if (window.isDestroyed() || window.webContents === sender) continue
    window.webContents.send('windows:message', message)
  }
}

ipcMain.handle('windows:popout', (_event, { imageId, name } = {}) => {
  if (!imageId) return { opened: false }
  return openPopout(imageId, name)
})

ipcMain.handle('windows:retarget', (event, { imageId, name } = {}) => {
  const window = BrowserWindow.fromWebContents(event.sender)
  if (!window || !imageId) return false
  window.fiduciaRetarget?.(imageId)
  window.setTitle(`${name || 'Image'} — Fiducia`)
  relay(null, { type: 'popouts', open: [...popouts.keys()] })
  return true
})

ipcMain.handle('windows:list', () => [...popouts.keys()])

ipcMain.on('windows:relay', (event, message) => relay(event.sender, message))

ipcMain.handle('windows:focusMain', () => {
  if (mainWindow && !mainWindow.isDestroyed()) {
    if (mainWindow.isMinimized()) mainWindow.restore()
    mainWindow.focus()
  }
})

// -- reading service keys --------------------------------------------------
//
// API keys for certificate reading. Encrypted with the operating system's own
// credential protection (DPAPI on Windows, the keychain on macOS) and kept in
// the per-user app data folder -- never in a project bundle, so sharing a
// .fidu folder never shares a key. The renderer can set or forget a key but
// never read one back; the engine is the only thing that receives it.

const PROVIDERS = ['offline', 'anthropic', 'openai']

function readerFile() {
  return path.join(app.getPath('userData'), 'reading-service.json')
}

function loadReader() {
  try {
    const stored = JSON.parse(readFileSync(readerFile(), 'utf-8'))
    return {
      provider: PROVIDERS.includes(stored.provider) ? stored.provider : 'offline',
      models: stored.models || {},
      keys: stored.keys || {},
    }
  } catch {
    return { provider: 'offline', models: {}, keys: {} }
  }
}

function saveReader(settings) {
  const target = readerFile()
  const temporary = `${target}.tmp`
  writeFileSync(temporary, JSON.stringify(settings, null, 2), 'utf-8')
  renameSync(temporary, target)
}

function decryptKey(encoded) {
  if (!encoded) return null
  try {
    return safeStorage.decryptString(Buffer.from(encoded, 'base64'))
  } catch {
    return null   // encrypted under a different Windows account; ask again
  }
}

// Keys entered while encryption is unavailable live only until the app closes.
const sessionKeys = {}

async function pushReaderSettings() {
  if (!enginePort) return null
  const stored = loadReader()
  const keys = {}
  for (const provider of PROVIDERS) {
    keys[provider] = sessionKeys[provider] ?? decryptKey(stored.keys[provider])
  }
  try {
    const response = await fetch(`http://127.0.0.1:${enginePort}/certificate/settings`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ provider: stored.provider, models: stored.models, keys }),
    })
    return response.ok ? await response.json() : null
  } catch {
    return null
  }
}

// -- IPC -------------------------------------------------------------------

ipcMain.handle('reader:update', async (_event, change = {}) => {
  const stored = loadReader()
  const encryption = safeStorage.isEncryptionAvailable()

  if (PROVIDERS.includes(change.provider)) stored.provider = change.provider

  for (const [provider, model] of Object.entries(change.models || {})) {
    if (!PROVIDERS.includes(provider)) continue
    if (model && String(model).trim()) stored.models[provider] = String(model).trim()
    else delete stored.models[provider]
  }

  for (const [provider, key] of Object.entries(change.keys || {})) {
    if (!PROVIDERS.includes(provider)) continue
    const trimmed = key ? String(key).trim() : ''
    delete sessionKeys[provider]
    delete stored.keys[provider]
    if (!trimmed) continue
    if (encryption) {
      stored.keys[provider] = safeStorage.encryptString(trimmed).toString('base64')
    } else {
      sessionKeys[provider] = trimmed
    }
  }

  saveReader(stored)
  const status = await pushReaderSettings()
  return { status, persisted: encryption }
})

ipcMain.handle('engine:info', () => ({ port: enginePort, ready: engineReady }))

ipcMain.handle('dialog:openFiles', async (_event, options = {}) => {
  const result = await dialog.showOpenDialog(mainWindow, {
    title: options.title || 'Select files',
    buttonLabel: options.buttonLabel || 'Select',
    properties: options.multiple === false ? ['openFile'] : ['openFile', 'multiSelections'],
    filters: options.filters || [
      { name: 'Imagery', extensions: ['tif', 'tiff', 'pix', 'jpg', 'jpeg', 'png', 'img', 'ntf'] },
      { name: 'All files', extensions: ['*'] },
    ],
  })
  return result.canceled ? [] : result.filePaths
})

ipcMain.handle('dialog:openDirectory', async (_event, options = {}) => {
  const result = await dialog.showOpenDialog(mainWindow, {
    title: options.title || 'Select a folder',
    properties: ['openDirectory', 'createDirectory'],
  })
  return result.canceled ? null : result.filePaths[0]
})

ipcMain.handle('dialog:saveFile', async (_event, options = {}) => {
  const result = await dialog.showSaveDialog(mainWindow, {
    title: options.title || 'Save',
    defaultPath: options.defaultPath,
    filters: options.filters || [{ name: 'All files', extensions: ['*'] }],
  })
  return result.canceled ? null : result.filePath
})

ipcMain.handle('shell:showItem', (_event, target) => {
  if (target) shell.showItemInFolder(target)
})

ipcMain.handle('shell:openPath', async (_event, target) => {
  if (target) return shell.openPath(target)
  return ''
})

ipcMain.handle('engine:restart', async () => {
  engineRestarts = 0
  if (engineProcess) {
    engineProcess.kill()
    engineProcess = null
  }
  return startEngine()
})

// -- lifecycle -------------------------------------------------------------

/**
 * The app was called Fiduplanus, and Stimulus before that, and Electron keeps
 * settings in a folder named after the app. Bring the saved reading-service
 * key, window positions and interface preferences across once, the first
 * time the new name runs, from the most recent name that has any.
 */
function carryOverSettings() {
  const current = app.getPath('userData')
  const previous = ['fiduplanus', 'stimulus']
    .map((name) => path.join(app.getPath('appData'), name))
    .find((folder) => existsSync(folder))
  if (!previous || existsSync(path.join(current, 'Local Storage'))) return
  mkdirSync(current, { recursive: true })
  for (const entry of ['reading-service.json', 'popout-windows.json', 'Local Storage']) {
    const from = path.join(previous, entry)
    try {
      if (existsSync(from)) cpSync(from, path.join(current, entry), { recursive: true })
    } catch {
      // A preference that fails to copy is re-entered, not a reason to stop.
    }
  }
}

carryOverSettings()

app.whenReady().then(async () => {
  Menu.setApplicationMenu(null)
  createWindow()
  await startEngine()
})

app.on('window-all-closed', () => {
  if (process.platform !== 'darwin') app.quit()
})

app.on('activate', () => {
  if (BrowserWindow.getAllWindows().length === 0) createWindow()
})

app.on('before-quit', () => {
  quitting = true
  if (engineProcess) {
    engineProcess.kill()
    engineProcess = null
  }
})
