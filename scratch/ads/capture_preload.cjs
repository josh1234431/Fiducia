/**
 * The desktop bridge, as the screenshot needs it: the engine is ready on
 * 8731, and the file dialog hands back the cloud named on the command line.
 */
const { contextBridge } = require('electron')

const cloud = (process.argv.find((a) => a.startsWith('--cloud=')) || '').slice('--cloud='.length)

contextBridge.exposeInMainWorld('fiducia', {
  isDesktop: true,
  engine: {
    info: async () => ({ port: 8731, ready: true }),
    onStatus: () => () => {},
  },
  dialog: { openFiles: async () => [cloud] },
  shell: { showItem: () => {} },
})
