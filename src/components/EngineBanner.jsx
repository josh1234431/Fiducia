import { useStore } from '../state/store'

/**
 * Surfaces engine trouble instead of letting the window quietly stop working.
 *
 * When a legacy workstation's processing falls over, the interface generally does too.
 * Here the engine is a separate process: if it dies, the shell restarts it and
 * says so, and the project on disk is untouched because every edit was
 * journalled before the crash.
 */
export default function EngineBanner() {
  const ready = useStore((s) => s.engineReady)
  const connected = useStore((s) => s.connected)
  const message = useStore((s) => s.engineMessage)
  const fatal = useStore((s) => s.engineFatal)

  // In the browser (dev, or headless use) there is no desktop shell managing
  // the engine, so only the live socket tells us anything useful.
  const desktop = typeof window !== 'undefined' && window.fiducia?.isDesktop

  if (connected && (ready || !desktop)) return null

  return (
    <div className={`engine-banner ${fatal ? 'engine-banner--fatal' : ''}`}>
      <span className="dot" />
      <strong>{fatal ? 'Engine unavailable' : 'Engine starting'}</strong>
      <span className="muted">
        {message || 'Connecting to the engine…'}
      </span>
      <span className="engine-banner__spacer" />
      {fatal && desktop && (
        <button
          className="btn btn--sm"
          onClick={() => window.fiducia.engine.restart()}
        >
          Restart engine
        </button>
      )}
    </div>
  )
}
