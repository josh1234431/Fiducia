import { useEffect, useState } from 'react'
import { useStore } from '../state/store'
import { api } from '../lib/api'
import { BrandMark, Wordmark } from './Brand'
import InfoTip from './InfoTip'

/**
 * The always-visible state of the world: what project is open, whether it is
 * saved, and whether the engine is alive.
 *
 * The autosave indicator is deliberately prominent. The complaint that started
 * this project was "it doesn't have autosave", and the fix is not only to save
 * continuously but to make that visible — otherwise the operator keeps
 * reaching for Ctrl+S anyway.
 */
export default function TitleBar() {
  const project = useStore((s) => s.project)
  const summary = useStore((s) => s.summary)
  const saveState = useStore((s) => s.saveState)
  const lastSavedAt = useStore((s) => s.lastSavedAt)
  const connected = useStore((s) => s.connected)
  const theme = useStore((s) => s.theme)
  const setTheme = useStore((s) => s.setTheme)
  const setPaletteOpen = useStore((s) => s.setPaletteOpen)
  const closeProject = useStore((s) => s.closeProject)
  const call = useStore((s) => s.call)
  const toast = useStore((s) => s.toast)

  const [ago, setAgo] = useState('')

  useEffect(() => {
    if (!lastSavedAt) return undefined
    const tick = () => {
      const seconds = Math.round((Date.now() - lastSavedAt) / 1000)
      setAgo(
        seconds < 5 ? 'just now'
          : seconds < 60 ? `${seconds}s ago`
          : `${Math.round(seconds / 60)}m ago`,
      )
    }
    tick()
    const timer = setInterval(tick, 5000)
    return () => clearInterval(timer)
  }, [lastSavedAt])

  const storage = useStore((s) => s.storage)
  const engineReady = useStore((s) => s.engineReady)
  const engineFatal = useStore((s) => s.engineFatal)
  const engineMessage = useStore((s) => s.engineMessage)
  const activeJobs = useStore((s) =>
    s.jobs.filter((j) => j.status === 'running' || j.status === 'queued').length)

  // Outside the desktop shell only the socket reports anything, so a live
  // connection counts as ready there.
  const desktop = typeof window !== 'undefined' && window.fiducia?.isDesktop
  const healthy = connected && (engineReady || !desktop)
  const engineStatus =
    healthy ? (activeJobs ? 'Engine processing' : 'Engine ready')
      : engineFatal ? 'Engine unavailable'
      : connected || !engineReady ? 'Engine starting'
      : 'Reconnecting to the engine'
  const engineDetail =
    healthy ? (activeJobs ? `${activeJobs} job${activeJobs === 1 ? '' : 's'} in progress.` : 'No jobs in progress.')
      : engineMessage || null

  // Never claim autosave while the folder is refusing writes.
  const shownState = storage ? 'failed' : saveState
  const saveLabel =
    storage ? 'Not saving'
      : saveState === 'saving' ? 'Saving…'
      : saveState === 'saved' ? 'Saved'
      : lastSavedAt ? `Saved ${ago}` : 'Autosave on'

  return (
    <header className="titlebar">
      {project ? (
        <button
          className="titlebar__brand titlebar__brand--home"
          onClick={closeProject}
          title="Close project and return to the start screen"
        >
          <BrandMark />
          <Wordmark className="titlebar__mark" />
        </button>
      ) : (
        <div className="titlebar__brand">
          <BrandMark />
          <Wordmark className="titlebar__mark" />
        </div>
      )}

      {project && (
        <div className="titlebar__project">
          <span className="titlebar__name truncate">{project.name}</span>
          <span className="titlebar__path truncate dim" title={summary?.directory}>
            {summary?.directory}
          </span>
        </div>
      )}

      <div className="titlebar__actions">
        {project && (
          <>
            <div
              className={`save-state save-state--${shownState}`}
              title={storage?.message ||
                'Every change is journalled to disk as it is made.'}
            >
              <span className="save-state__pulse" />
              {saveLabel}
            </div>

            <span className="titlebar__sep" aria-hidden="true" />

            <button
              className="btn btn--ghost btn--sm"
              onClick={() =>
                call(() => api.project.snapshot('manual'), {
                  successMessage: 'Snapshot taken',
                })
              }
              title="Take a named snapshot"
            >
              Snapshot
            </button>
          </>
        )}

        <button
          className="btn btn--ghost btn--sm"
          onClick={() => setPaletteOpen(true)}
          title="Command palette"
        >
          Commands <kbd>Ctrl K</kbd>
        </button>

        <span className="titlebar__sep" aria-hidden="true" />

        <button
          className="btn btn--ghost btn--sm titlebar__icon"
          onClick={() => setTheme(theme === 'dark' ? 'light' : 'dark')}
          title={`Switch to ${theme === 'dark' ? 'light' : 'dark'} theme`}
          aria-label={`Switch to ${theme === 'dark' ? 'light' : 'dark'} theme`}
        >
          {theme === 'dark' ? '◐' : '◑'}
        </button>

        <InfoTip
          className={`engine-dot ${healthy ? 'engine-dot--ok' : 'engine-dot--wait'}`}
          label={engineStatus}
          trigger={null}
          onClick={() => !connected && toast('Reconnecting to the engine…', 'warn')}
        >
          <strong>{engineStatus}</strong>
          {engineDetail && <p>{engineDetail}</p>}
        </InfoTip>
      </div>
    </header>
  )
}
