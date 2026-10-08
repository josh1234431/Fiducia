import { useState } from 'react'
import { useStore } from '../state/store'

/**
 * The project folder has stopped accepting writes.
 *
 * Stays up for as long as that is true, because a toast that fades while the
 * USB stick is still on the desk is how work gets lost. It says what happened
 * in terms of the drive, and offers the two ways out: bring the drive back, or
 * put the project somewhere else.
 */
export default function StorageBanner() {
  const storage = useStore((s) => s.storage)
  const project = useStore((s) => s.project)
  const checkStorage = useStore((s) => s.checkStorage)
  const saveProjectElsewhere = useStore((s) => s.saveProjectElsewhere)
  const toast = useStore((s) => s.toast)
  const [busy, setBusy] = useState(false)

  if (!storage || !project) return null

  const heading = {
    missing: 'Project drive disconnected',
    moved: 'Project folder not found',
    readonly: 'Project drive is read-only',
    full: 'Project drive is full',
    denied: 'No permission to save',
    in_use: 'Project file is in use',
  }[storage.kind] || 'The project cannot be saved'

  return (
    <div className="engine-banner engine-banner--fatal storage-banner" role="alert">
      <span className="dot" />
      <div className="storage-banner__text">
        <strong>{heading}</strong>
        <span className="muted">
          {storage.message} Editing is paused until saving is possible.
        </span>
      </div>
      <span className="engine-banner__spacer" />
      <button
        className="btn btn--sm"
        disabled={busy}
        onClick={async () => {
          setBusy(true)
          const ok = await checkStorage(true)
          setBusy(false)
          if (!ok) toast('The location is still unavailable.', 'warn')
        }}
      >
        {busy ? 'Checking…' : 'Check again'}
      </button>
      {window.fiducia?.isDesktop && (
        <button
          className="btn btn--primary btn--sm"
          disabled={busy}
          onClick={async () => {
            setBusy(true)
            await saveProjectElsewhere()
            setBusy(false)
          }}
        >
          Save to another folder…
        </button>
      )}
    </div>
  )
}
