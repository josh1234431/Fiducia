/**
 * The bridge between the sandboxed interface and the desktop.
 *
 * Deliberately narrow: the renderer gets native file dialogs, the engine's
 * port, and a subscription to engine status. Everything else — all project
 * state, all processing — goes over HTTP to the engine, which keeps the
 * privileged surface here small enough to audit at a glance.
 */

const { contextBridge, ipcRenderer } = require('electron')

contextBridge.exposeInMainWorld('fiducia', {
  isDesktop: true,
  platform: process.platform,

  engine: {
    info: () => ipcRenderer.invoke('engine:info'),
    restart: () => ipcRenderer.invoke('engine:restart'),
    onStatus: (callback) => {
      const handler = (_event, payload) => callback(payload)
      ipcRenderer.on('engine:status', handler)
      return () => ipcRenderer.removeListener('engine:status', handler)
    },
    onLog: (callback) => {
      const handler = (_event, payload) => callback(payload)
      ipcRenderer.on('engine:log', handler)
      return () => ipcRenderer.removeListener('engine:log', handler)
    },
  },

  dialog: {
    openFiles: (options) => ipcRenderer.invoke('dialog:openFiles', options),
    openDirectory: (options) => ipcRenderer.invoke('dialog:openDirectory', options),
    saveFile: (options) => ipcRenderer.invoke('dialog:saveFile', options),
  },

  // Popped-out image windows, for working across several monitors. Windows
  // only exchange what the operator is doing; the project itself each window
  // reads from the engine directly.
  windows: {
    popout: (imageId, name) => ipcRenderer.invoke('windows:popout', { imageId, name }),
    retarget: (imageId, name) => ipcRenderer.invoke('windows:retarget', { imageId, name }),
    list: () => ipcRenderer.invoke('windows:list'),
    focusMain: () => ipcRenderer.invoke('windows:focusMain'),
    relay: (message) => ipcRenderer.send('windows:relay', message),
    onMessage: (callback) => {
      const handler = (_event, payload) => callback(payload)
      ipcRenderer.on('windows:message', handler)
      return () => ipcRenderer.removeListener('windows:message', handler)
    },
  },

  // Certificate reading service. Write-only for keys: the interface can set
  // or forget one, and gets back whether it is present, never its value.
  reader: {
    update: (change) => ipcRenderer.invoke('reader:update', change),
  },

  shell: {
    showItem: (target) => ipcRenderer.invoke('shell:showItem', target),
    openPath: (target) => ipcRenderer.invoke('shell:openPath', target),
  },
})
