import { useStore } from '../state/store'
import { api } from '../lib/api'

/**
 * Activity bar: a timeline of what has happened in this session.
 *
 * Processing jobs and messages share one strip, newest on the left, older
 * events pushed to the right. A job keeps a single card that moves through
 * queued, running and its outcome, and moves to the front when it finishes,
 * because finishing is the event worth seeing. Nothing pops up over the work.
 */

const ACTIVE = new Set(['running', 'queued'])

function clock(ms) {
  return new Date(ms).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
}

function formatElapsed(seconds) {
  if (seconds == null) return ''
  if (seconds < 60) return `${seconds.toFixed(0)}s`
  return `${Math.floor(seconds / 60)}m ${Math.round(seconds % 60)}s`
}

function DismissButton({ label, onClick }) {
  return (
    <button className="jobbar__dismiss" onClick={onClick} title="Dismiss" aria-label={`Dismiss ${label}`}>
      ✕
    </button>
  )
}

// Jobs that leave a file behind, the step that lists it, and the file to show.
const OUTPUT_STEP = { ortho: 'ortho', mosaic: 'mosaic', dem: 'dem', lidar: 'lidar', terrain: 'terrain' }

function outputOf(job) {
  const result = job.result || {}
  const path = result.merged?.outputPath
    || result.outputPath
    || result.dems?.[result.dems.length - 1]?.outputPath
    || result.orthos?.[0]?.outputPath
  return path ? { path, label: path.split(/[\\/]/).pop() } : null
}

function JobEvent({ job }) {
  const toast = useStore((s) => s.toast)
  const dismissJob = useStore((s) => s.dismissJob)
  const revealOutputs = useStore((s) => s.revealOutputs)
  const showOutput = useStore((s) => s.showOutput)
  const active = ACTIVE.has(job.status)
  const step = OUTPUT_STEP[job.kind]
  const output = job.status === 'done' && step ? outputOf(job) : null

  function view() {
    // Rasters open on the canvas; anything else (contours) is just listed.
    if (/\.(tif|tiff|img|vrt)$/i.test(output.path)) showOutput({ ...output, step })
    revealOutputs(step)
  }

  return (
    <div className={`jobbar__job jobbar__job--${job.status}`} title={job.error || job.message}>
      <div style={{ minWidth: 0, flex: 1 }}>
        <div className="jobbar__label truncate">{job.label}</div>
        <div className="jobbar__message truncate">
          {job.status === 'failed' ? job.error : job.message}
          {job.elapsedSeconds != null && job.status === 'done' && `, ${formatElapsed(job.elapsedSeconds)}`}
        </div>
      </div>

      {active ? (
        <>
          <div className="jobbar__bar">
            <div className="jobbar__fill" style={{ width: `${job.progress * 100}%` }} />
          </div>
          <span className="jobbar__pct">{Math.round(job.progress * 100)}%</span>
          <button
            className="btn btn--ghost btn--sm"
            onClick={async () => {
              await api.jobs.cancel(job.id)
              toast(`Cancelling: ${job.label}`, 'warn')
            }}
            title="Cancel this job"
          >
            Cancel
          </button>
        </>
      ) : (
        <>
          <span className="jobbar__time">{clock((job.finishedAt || job.createdAt) * 1000)}</span>
          {job.status === 'done' && (output ? (
            <button className="btn btn--primary btn--sm jobbar__view" onClick={view}
                    title={`Open ${output.label} and its step`}>
              View
            </button>
          ) : <span className="chip chip--good">done</span>)}
          {job.status === 'failed' && <span className="chip chip--bad">failed</span>}
          {job.status === 'cancelled' && <span className="chip">cancelled</span>}
          <DismissButton label={job.label} onClick={() => dismissJob(job.id)} />
        </>
      )}
    </div>
  )
}

function NoteEvent({ note }) {
  const dismiss = useStore((s) => s.dismissActivity)
  return (
    <div className={`jobbar__job jobbar__note jobbar__note--${note.tone}`} title={note.message}>
      <div className="jobbar__label jobbar__label--note truncate">{note.message}</div>
      {note.count > 1 && <span className="chip">×{note.count}</span>}
      <span className="jobbar__time">{clock(note.at)}</span>
      <DismissButton label={note.message} onClick={() => dismiss(note.id)} />
    </div>
  )
}

export default function JobBar() {
  const jobs = useStore((s) => s.jobs)
  const activity = useStore((s) => s.activity)
  const clearAll = useStore((s) => s.clearFinishedJobs)

  // Running work stays at the front until it finishes; everything else is
  // ordered by when it last changed.
  const events = [
    ...jobs.map((job) => ({
      kind: 'job',
      key: job.id,
      job,
      at: ACTIVE.has(job.status) ? Infinity : (job.finishedAt || job.createdAt) * 1000,
    })),
    ...activity.map((note) => ({ kind: 'note', key: note.id, note, at: note.at })),
  ].sort((a, b) => b.at - a.at).slice(0, 40)

  const finished = events.some((e) => e.kind === 'note' || !ACTIVE.has(e.job.status))

  return (
    <footer className="jobbar" aria-label="Activity">
      {events.length === 0 ? (
        <div className="jobbar__idle">No recent activity</div>
      ) : (
        <div className="jobbar__track">
          {events.map((event) => event.kind === 'job'
            ? <JobEvent key={event.key} job={event.job} />
            : <NoteEvent key={event.key} note={event.note} />)}
        </div>
      )}

      {finished && (
        <button className="btn btn--ghost btn--sm jobbar__clear" onClick={clearAll}>
          Clear
        </button>
      )}
    </footer>
  )
}
