/**
 * Preferences saved under the app's previous names (Fiduplanus, and Stimulus
 * before that) move across once.
 *
 * This is its own module, imported first by main.jsx, because the store reads
 * these keys while it is being created — an import runs before any code in the
 * importing file, so a migration written inline in main.jsx would run too late.
 */
try {
  for (const key of ['theme', 'recent', 'reportLayout']) {
    if (localStorage.getItem(`fiducia.${key}`) !== null) continue
    for (const previousName of ['fiduplanus', 'stimulus']) {
      const previous = localStorage.getItem(`${previousName}.${key}`)
      if (previous !== null) {
        localStorage.setItem(`fiducia.${key}`, previous)
        break
      }
    }
  }
} catch {
  // Storage unavailable: defaults apply.
}
