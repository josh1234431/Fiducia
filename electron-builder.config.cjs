/**
 * Installer recipe. Run packaging/build.ps1 rather than this directly: it
 * builds the interface and freezes the engine first.
 *
 * Everything large is built in FIDUCIA_BUILD_DIR (set by build.ps1; by default
 * a Fiducia-build folder beside the checkout), outside OneDrive and AppData.
 */
const path = require('node:path')

const build = process.env.FIDUCIA_BUILD_DIR || path.resolve(__dirname, '..', 'Fiducia-build')

module.exports = {
  appId: 'com.joshuametcalf.fiducia',
  productName: 'Fiducia',
  copyright: 'Copyright 2026 Joshua Metcalf',
  directories: {
    output: path.join(build, 'release'),
    buildResources: 'resources',
  },
  files: ['dist/**', 'electron/**', 'resources/icon.png', 'package.json', 'LICENSE', 'NOTICE'],
  // The frozen engine sits beside the app, not inside its archive, so
  // Windows can run it and its worker processes directly.
  extraResources: [{ from: path.join(build, 'engine', 'fiducia-engine'), to: 'engine' }],
  win: {
    target: [{ target: 'nsis', arch: ['x64'] }],
    icon: 'resources/icon.ico',
  },
  nsis: {
    // Per-user install: no administrator rights needed, which matters on
    // university-managed computers.
    oneClick: false,
    perMachine: false,
    allowToChangeInstallationDirectory: true,
    createDesktopShortcut: true,
    createStartMenuShortcut: true,
    shortcutName: 'Fiducia',
    license: 'LICENSE',
    artifactName: 'Fiducia-Setup-${version}.${ext}',
  },
}
