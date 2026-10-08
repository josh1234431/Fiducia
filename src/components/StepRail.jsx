import { useEffect, useRef, useState } from 'react'
import { useStore, STEPS, stepStatus, stepsFor } from '../state/store'
import PanelToggle from './PanelToggle'

const STATUS_TITLE = {
  done: 'Complete',
  attention: 'Needs attention',
  ready: 'Ready',
  locked: 'Requires the preceding steps',
}

/**
 * The completion mark.
 *
 * Four states, each legible at a glance without reading a label: an empty
 * ring is locked, a dashed ring is ready, an amber ring needs attention, and
 * done is a filled disc with a checkmark drawn into it.
 *
 * When a step actually crosses into done the mark pops once and the tick
 * draws itself. That is the only unprompted animation in the interface, and
 * it is earned — it marks real progress through real work.
 */
function StatusMark({ status, celebrate }) {
  return (
    <span
      className={['mark', `mark--${status}`, celebrate ? 'mark--celebrate' : ''].join(' ')}
      title={STATUS_TITLE[status]}
    >
      <svg viewBox="0 0 20 20" aria-hidden="true">
        <circle className="mark__ring" cx="10" cy="10" r="7.5" />
        <path className="mark__tick" d="M6.2 10.3 L8.8 12.9 L13.9 7.4" />
      </svg>
    </span>
  )
}

/**
 * A readout whose value flashes when it changes, so work finishing in the
 * background gets noticed without having to be watched for.
 */
function Readout({ label, value, tone = '' }) {
  const [changed, setChanged] = useState(false)
  const previous = useRef(value)

  useEffect(() => {
    if (previous.current === value) return
    previous.current = value
    setChanged(true)
    const timer = setTimeout(() => setChanged(false), 700)
    return () => clearTimeout(timer)
  }, [value])

  return (
    <div className="readout">
      <span>{label}</span>
      <span
        className={[
          'readout__value',
          tone ? `readout__value--${tone}` : '',
          changed ? 'readout__value--changed' : '',
        ].join(' ')}
      >
        {value}
      </span>
    </div>
  )
}

/**
 * The processing rail.
 *
 * Replaces the classic "Processing step" dropdown. A dropdown shows one step
 * and tells you nothing about the others, so the only way to learn that an
 * image is missing its interior orientation is to walk into that step and
 * look. Here every step carries a live completion mark, and the state of the
 * whole block is readable without navigating anywhere.
 */
export default function StepRail() {
  const toggleRail = useStore((s) => s.toggleRail)
  const activeStep = useStore((s) => s.activeStep)
  const setStep = useStore((s) => s.setStep)
  const summary = useStore((s) => s.summary)
  const model = useStore((s) => s.project?.model)
  const project = useStore((s) => s.project)
  const statuses = useStore((s) =>
    Object.fromEntries(STEPS.map((step) => [step.id, stepStatus(s, step.id)])),
  )
  // The project's own workflow leads; everything else waits under "More
  // tools", never gone. A step opened from there keeps the fold open.
  const { primary, more } = stepsFor(project)
  const [moreOpen, setMoreOpen] = useState(() => {
    try { return localStorage.getItem('fiducia.moreTools') === 'open' } catch { return false }
  })
  const showMore = moreOpen || more.some((step) => step.id === activeStep)
  function toggleMore() {
    const next = !showMore
    setMoreOpen(next)
    try { localStorage.setItem('fiducia.moreTools', next ? 'open' : 'closed') } catch { /* optional */ }
  }

  // Fire the completion animation only on the transition into done, never on
  // first render — otherwise opening a finished project sets off ten of them.
  const previous = useRef(null)
  const [celebrating, setCelebrating] = useState({})

  useEffect(() => {
    const before = previous.current
    previous.current = statuses
    if (!before) return

    const newlyDone = Object.keys(statuses).filter(
      (id) => statuses[id] === 'done' && before[id] && before[id] !== 'done',
    )
    if (!newlyDone.length) return

    setCelebrating((current) => ({
      ...current,
      ...Object.fromEntries(newlyDone.map((id) => [id, true])),
    }))
    const timer = setTimeout(() => {
      setCelebrating((current) => {
        const next = { ...current }
        newlyDone.forEach((id) => delete next[id])
        return next
      })
    }, 900)
    return () => clearTimeout(timer)
  }, [statuses])

  const doneCount = primary.filter((step) => statuses[step.id] === 'done').length

  function renderStep(step) {
    const status = statuses[step.id]
    const locked = status === 'locked'
    return (
      <button
        key={step.id}
        className={[
          'step',
          activeStep === step.id ? 'step--active' : '',
          locked ? 'step--locked' : '',
          status === 'done' ? 'step--done' : '',
        ].join(' ')}
        onClick={() => !locked && setStep(step.id)}
        disabled={locked}
        title={locked ? STATUS_TITLE.locked : step.blurb}
      >
        <StatusMark status={status} celebrate={!!celebrating[step.id]} />
        <span className="step__text">
          <span className="step__name">
            {step.name}
          </span>
          <span className="step__detail">{step.detail}</span>
        </span>
      </button>
    )
  }

  return (
    <nav className="rail">
      <div className="rail__head">
        <PanelToggle side="left" open onToggle={toggleRail} label="steps" shortcut="Ctrl B" />
        <span className="rail__head-label">Steps</span>
      </div>

      {/* Progress down the spine of the rail: how much of the chain is done. */}
      <div className="rail__spine" aria-hidden="true">
        <div
          className="rail__spine-fill"
          style={{ height: `${(doneCount / primary.length) * 100}%` }}
        />
      </div>

      <div className="rail__steps">
        {primary.map(renderStep)}
        {more.length > 0 && (
          <>
            <button
              className={`rail__more ${showMore ? 'rail__more--open' : ''}`}
              onClick={toggleMore}
              aria-expanded={showMore}
              title="Steps this workflow does not usually need"
            >
              <span className="rail__more-chevron" aria-hidden="true">›</span>
              More tools
            </button>
            {showMore && more.map(renderStep)}
          </>
        )}
      </div>

      <div className="rail__readout">
        <Readout
          label="Photographs"
          value={
            summary
              ? summary.imageCount !== summary.onlineCount
                ? `${summary.onlineCount}/${summary.imageCount}`
                : summary.onlineCount
              : 0
          }
          tone={summary && summary.imageCount !== summary.onlineCount ? 'warn' : ''}
        />
        <Readout
          label="Control"
          value={`${summary?.gcpCount ?? 0}, ${summary?.checkPointCount ?? 0}`}
        />
        <Readout label="Tie points" value={summary?.tiePointCount ?? 0} />
        {model && (
          <Readout
            label="Sigma nought"
            value={model.sigma0?.toFixed(4) ?? '—'}
            tone={model.converged ? 'good' : 'warn'}
          />
        )}
      </div>
    </nav>
  )
}
