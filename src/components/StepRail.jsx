import { useEffect, useLayoutEffect, useRef, useState } from 'react'
import { useStore, STEPS, stepStatus, stepsFor, workflowOf } from '../state/store'
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

/** A point count as people say it: 0.55M, 84,210. */
function points(n) {
  return n >= 100000 ? `${(n / 1e6).toFixed(2)}M` : n.toLocaleString()
}

/**
 * The readouts under the steps: the few numbers that say where this project
 * stands, chosen for its kind of data. A LiDAR project has no photographs or
 * tie points to count; a drone block's ground pixel matters more than its
 * control. For photographs the last number is the check-point error, the one
 * that says how accurate the result is. Sigma nought only says the weights
 * suit the residuals, so it stays on the Model step as a diagnostic. No colour
 * here: whether an error is good depends on the job, not on a fixed line.
 */
function readoutsFor({ workflow, summary, model, project, lidar, labelled }) {
  if (workflow === 'lidar') {
    if (!lidar || lidar.indexing) {
      return [
        { label: 'Point cloud', value: lidar?.indexing ? 'indexing…' : 'none open' },
        { label: 'Clouds made', value: project?.lidarClouds?.length ?? 0 },
        { label: 'Rasters made', value: project?.lidar?.length ?? 0 },
      ]
    }
    const ground = (lidar.classHistogram?.['2']?.count || 0) / Math.max(lidar.pointCount, 1)
    return [
      { label: 'Points', value: points(lidar.pointCount) },
      { label: 'Spacing', value: `${lidar.averageSpacing.toFixed(2)} m` },
      { label: 'Ground', value: ground > 0 ? `${(ground * 100).toFixed(1)}%` : 'not found',
        tone: ground > 0 ? '' : 'warn' },
      ...(labelled ? [{ label: 'Labelled', value: labelled.toLocaleString() }] : []),
    ]
  }

  const offline = summary && summary.imageCount !== summary.onlineCount
  const rows = [{
    label: { drone: 'Photos', satellite: 'Scenes' }[workflow] || 'Photographs',
    value: summary ? (offline ? `${summary.onlineCount}/${summary.imageCount}` : summary.onlineCount) : 0,
    tone: offline ? 'warn' : '',
  }]
  const gcps = summary?.gcpCount ?? 0
  const checks = summary?.checkPointCount ?? 0
  if (workflow !== 'drone' || gcps || checks) {
    rows.push({ label: 'Control', value: `${gcps} GCP · ${checks} check` })
  }
  rows.push({ label: 'Tie points', value: (summary?.tiePointCount ?? 0).toLocaleString() })
  const gsd = project?.droneBlock?.gsdM
  if (workflow === 'drone' && gsd) rows.push({ label: 'Ground pixel', value: `${(gsd * 100).toFixed(1)} cm` })
  if (model) {
    rows.push({ label: 'Check RMS', value: model.rmsCheckM ? `${model.rmsCheckM.toFixed(2)} m` : 'no check points' })
  }
  return rows
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
  const lidar = useStore((s) => s.lidarSummary)
  const labelled = useStore((s) => s.lidarLabelCount)
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

  // The spine runs from the top of the rail to the mark of the last finished
  // step in the workflow, so the line joins the ticks rather than standing for
  // a fraction of the rail's height.
  const railRef = useRef(null)
  const spineRef = useRef(null)
  const stepRefs = useRef({})
  const [spineTo, setSpineTo] = useState(0)
  const lastDone = primary.map((step) => statuses[step.id]).lastIndexOf('done')
  const lastDoneId = lastDone >= 0 ? primary[lastDone].id : null
  useLayoutEffect(() => {
    function measure() {
      const mark = lastDoneId && stepRefs.current[lastDoneId]?.querySelector('.mark')
      const spine = spineRef.current
      if (!mark || !spine) { setSpineTo(0); return }
      const m = mark.getBoundingClientRect()
      setSpineTo(Math.max(0, m.top + m.height / 2 - spine.getBoundingClientRect().top))
    }
    measure()
    const observer = new ResizeObserver(measure)
    if (railRef.current) observer.observe(railRef.current)
    return () => observer.disconnect()
  }, [lastDoneId, primary.length])

  function renderStep(step) {
    const status = statuses[step.id]
    const locked = status === 'locked'
    return (
      <button
        key={step.id}
        ref={(el) => { if (el) stepRefs.current[step.id] = el }}
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
    <nav className="rail" ref={railRef}>
      <div className="rail__head">
        <PanelToggle side="left" open onToggle={toggleRail} label="steps" shortcut="Ctrl B" />
        <span className="rail__head-label">Steps</span>
      </div>

      {/* Progress down the spine of the rail, as far as the last finished step. */}
      <div className="rail__spine" aria-hidden="true" ref={spineRef}>
        <div className="rail__spine-fill" style={{ height: `${spineTo}px` }} />
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
        {readoutsFor({ workflow: workflowOf(project).id, summary, model, project, lidar, labelled })
          .map((row) => <Readout key={row.label} label={row.label} value={row.value} tone={row.tone} />)}
      </div>
    </nav>
  )
}
