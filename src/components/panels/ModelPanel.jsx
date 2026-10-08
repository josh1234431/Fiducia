import { useEffect, useState } from 'react'
import { useStore } from '../../state/store'
import { api } from '../../lib/api'
import InfoTip from '../InfoTip'

/**
 * Bundle adjustment, plus the diagnostics that make a failed one fixable.
 *
 * Three things here that the original does not do: a readiness check that says
 * exactly what is missing before you run anything, automatic blunder detection
 * that names the suspect point, and a residual table sortable in either pixels
 * or ground units without regenerating a report.
 */
export default function ModelPanel() {
  const project = useStore((s) => s.project)
  const readiness = useStore((s) => s.readiness)
  const refreshReadiness = useStore((s) => s.refreshReadiness)
  const call = useStore((s) => s.call)
  const setSelectedPoint = useStore((s) => s.setSelectedPoint)

  const [units, setUnits] = useState('ground')
  const [show, setShow] = useState('all')
  const [rows, setRows] = useState([])
  const [sort, setSort] = useState({ key: 'magnitude', direction: 'desc' })

  const model = project?.model

  useEffect(() => { refreshReadiness() }, [project?.modified, refreshReadiness])

  useEffect(() => {
    if (!model) { setRows([]); return }
    api.model.residuals(units, show).then((r) => setRows(r.rows)).catch(() => setRows([]))
  }, [model, units, show])

  const sorted = [...rows].sort((a, b) => {
    const dir = sort.direction === 'asc' ? 1 : -1
    const av = a[sort.key]
    const bv = b[sort.key]
    if (typeof av === 'string') return av.localeCompare(bv) * dir
    return ((av ?? 0) - (bv ?? 0)) * dir
  })

  function toggleSort(key) {
    setSort((s) => s.key === key
      ? { key, direction: s.direction === 'asc' ? 'desc' : 'asc' }
      : { key, direction: 'desc' })
  }

  const tolerance = units === 'ground' ? 2.0 : 1.0
  const allClear = readiness && !readiness.blockers.length && !readiness.warnings.length

  return (
    <>
      <section className="section">
        <div className="section__head">
          <span className="section__title">Readiness</span>
          <span className="section__rule" />
          {readiness && allClear && <span className="chip chip--good">ready</span>}
        </div>

        {/* Nothing to say takes no room: the all-clear is the chip above. */}
        {readiness && !allClear && (
          <div className={`readiness ${readiness.ready ? 'readiness--ready' : 'readiness--blocked'}`}>
            {readiness.blockers.map((item, i) => (
              <div key={`b${i}`}>
                <div className="readiness__item readiness__item--blocker">
                  <span className="dot" /><span>{item.message}</span>
                </div>
                {(item.fixes || (item.fix ? [{ fix: item.fix, label: item.fixLabel }] : []))
                  .map((option) => (
                    <button key={option.label} className="btn btn--sm"
                            style={{ margin: '4px 0 6px 16px', display: 'block' }}
                            onClick={() => call(() => api.camera.set({ ...(project?.camera || {}),
                                                                       ...option.fix }),
                              { successMessage: 'Camera updated' })}>
                      {option.label || 'Apply fix'}
                    </button>
                  ))}
              </div>
            ))}
            {readiness.warnings.map((item, i) => (
              <div className="readiness__item readiness__item--warning" key={`w${i}`}>
                <span className="dot" /><span>{item.message}</span>
              </div>
            ))}
          </div>
        )}

        <AdjustmentSettings project={project} call={call} />

        <button
          className="btn btn--primary btn--block"
          disabled={!readiness?.ready}
          onClick={() => call(() => api.model.compute())}
        >
          Compute model
        </button>

        {readiness && (
          <div className="field__hint" style={{ marginTop: 6 }}>
            {readiness.counts.images > 1
              ? `Bundle adjustment, ${readiness.counts.online} images: `
              : 'Space resection: '}
            {readiness.counts.completeGcps} control,
            {' '}{readiness.counts.tiePoints} tie,
            {' '}{readiness.counts.observations} measurements
          </div>
        )}
      </section>

      {model && (
        <>
          <section className="section">
            <div className="section__head">
              <span className="section__title">Solution</span>
          <span className="section__rule" />
              <span className={`chip ${!model.converged ? 'chip--bad'
                : model.quality === 'poor' ? 'chip--warn' : 'chip--good'}`}>
                {!model.converged ? (model.quality === 'failed' ? 'failed' : 'not converged')
                  : model.quality === 'poor' ? 'converged, poor fit' : 'converged'}
              </span>
            </div>

            {(model.qualityProblems || []).length > 0 && (
              <div className="readiness" style={{
                borderLeftColor: model.converged ? 'var(--signal-warn)' : 'var(--signal-bad)',
                marginBottom: 'var(--step-3)' }}>
                {model.qualityProblems.map((problem, index) => (
                  <div key={index} className={`readiness__item ${model.converged
                    ? 'readiness__item--warning' : 'readiness__item--blocker'}`}>
                    <span className="dot" /><span>{problem}</span>
                  </div>
                ))}
              </div>
            )}

            <div className="measures">
              <div className="measure">
                <div className="measure__name">Sigma-0</div>
                <div className="measure__value">{model.sigma0?.toFixed(4) ?? '—'}</div>
              </div>
              <div className="measure">
                <div className="measure__name">Image RMS</div>
                <div className="measure__value">
                  {model.rmsImageMm != null
                    ? `${(model.rmsImageMm * 1000).toFixed(1)}µm` : '—'}
                </div>
                {model.rmsImageMm != null && project?.camera?.kind === 'digital'
                  && project.camera.pixelPitchMm > 0 && (
                  <div className="field__hint">
                    {(model.rmsImageMm / project.camera.pixelPitchMm).toFixed(2)} px
                  </div>
                )}
              </div>
              <div className="measure">
                <div className="measure__name">Control RMS</div>
                {(() => {
                  // No control at all (a GPS-only block) is not a perfect 0.00 m.
                  const rms = Object.keys(model.controlResidualsM || {}).length
                    ? model.rmsControlM : null
                  return (
                    <div className={`measure__value ${
                      rms == null ? '' : rms < 2 ? 'measure__value--good' : 'measure__value--warn'}`}>
                      {rms != null ? `${rms.toFixed(2)}m` : '—'}
                    </div>
                  )
                })()}
              </div>
              <div className="measure">
                <div className="measure__name">Check RMS</div>
                <div className={`measure__value ${
                  model.rmsCheckM == null || model.rmsCheckM === 0 ? ''
                    : model.rmsCheckM < 2 ? 'measure__value--good' : 'measure__value--warn'}`}>
                  {model.rmsCheckM ? `${model.rmsCheckM.toFixed(2)}m` : '—'}
                </div>
              </div>
            </div>

            <div className="field__hint" style={{ marginTop: 6 }}>
              {model.method === 'bundle' ? 'Bundle block adjustment' : 'Space resection'}
              {' '}on {model.degreesOfFreedom} degrees of freedom.
              {' '}Solved {model.solvedAt}.
              {model.meanRedundancy != null
                && ` Mean redundancy ${model.meanRedundancy.toFixed(2)}.`}
            </div>
            {(model.notes || []).map((note) => (
              <div key={note} className="field__hint">{note}</div>
            ))}
            {(() => {
              const unused = (model.singleRayPoints || [])
                .filter((id) => !(model.horizontalChecks || []).includes(id))
              return unused.length > 0 && (
                <div className="field__hint">
                  {unused.length} point{unused.length === 1 ? ' is' : 's are'} measured on one
                  photo only and could not be used: {unused.slice(0, 8).join(', ')}
                  {unused.length > 8 ? '…' : ''}.
                </div>
              )
            })()}
          </section>

          {model.accuracy?.checkPoints > 0 && <AccuracySection accuracy={model.accuracy} />}

          {model.selfCalibration?.values && (
            <SelfCalibrationSection calibration={model.selfCalibration} />
          )}

          {model.precision && Object.keys(model.precision).length > 0 && (
            <section className="section">
              <div className="section__head">
                <span className="section__title">
                  Precision
                  <InfoTip>
                    <p>One standard deviation, from the adjustment covariance scaled
                    by sigma-0.</p>
                    <p>Corners is the ground uncertainty at each image corner, which
                    the orthophoto inherits. Values above 5 px are highlighted and
                    indicate insufficient or poorly distributed control.</p>
                  </InfoTip>
                </span>
                <span className="section__rule" />
              </div>
              <table className="table">
                <thead>
                  <tr>
                    <th>Image</th>
                    <th className="num" title="Perspective centre, horizontal">± XY m</th>
                    <th className="num" title="Perspective centre, height">± Z m</th>
                    <th className="num" title="Largest of omega, phi and kappa">± angle ″</th>
                    <th className="num" title="Ground uncertainty at the image corners">corners</th>
                  </tr>
                </thead>
                <tbody>
                  {Object.entries(model.precision).map(([imageId, entry]) => {
                    if (!entry) return null
                    const name = project?.images?.find((i) => i.id === imageId)?.name || imageId
                    const [sx, sy, sz, so, sp, sk] = entry.sigma
                    const weak = entry.cornerPx != null && entry.cornerPx > 5
                    return (
                      <tr key={imageId}>
                        <td>{name}</td>
                        <td className="num">{Math.hypot(sx, sy).toFixed(2)}</td>
                        <td className="num">{sz.toFixed(2)}</td>
                        <td className="num">
                          {(Math.max(so, sp, sk) * 206264.8).toFixed(0)}
                        </td>
                        <td className={`num ${weak ? 'over' : ''}`}>
                          {entry.cornerM.toFixed(2)} m
                          {entry.cornerPx != null && ` (${entry.cornerPx.toFixed(1)} px)`}
                        </td>
                      </tr>
                    )
                  })}
                </tbody>
              </table>
            </section>
          )}

          {model.suspects?.length > 0 && (
            <section className="section">
              <div className="section__head">
                <span className="section__title">Probable blunders</span>
          <span className="section__rule" />
                <span className="chip chip--warn">{model.suspects.length}</span>
              </div>
              {model.suspects.slice(0, 5).map((suspect) => (
                <div
                  key={suspect.pointId}
                  className="readiness"
                  style={{ borderColor: 'color-mix(in srgb, var(--signal-warn) 40%, transparent)',
                           marginBottom: 6 }}
                  onMouseEnter={() => setSelectedPoint(suspect.pointId)}
                  onMouseLeave={() => setSelectedPoint(null)}
                >
                  <div className="row__name" style={{ marginBottom: 3 }}>
                    {suspect.pointId}
                    <span className="chip chip--warn"
                          title="Standardised residual (data snooping). Above 3.29 fails the test.">
                      {suspect.standardised.toFixed(1)}
                    </span>
                  </div>
                  <div className="field__hint">{suspect.note}</div>
                  {suspect.kind === 'control' && (
                    <button
                      className="btn btn--sm"
                      style={{ marginTop: 6 }}
                      onClick={() => call(async () => {
                        await api.model.setCheckPoint(suspect.pointId, true)
                        return api.model.compute()
                      }, { successMessage: `${suspect.pointId} excluded. Re-solving.` })}
                    >
                      Exclude and re-solve
                    </button>
                  )}
                </div>
              ))}
            </section>
          )}

          <section className="section">
            <div className="section__head">
              <span className="section__title">Residuals</span>
          <span className="section__rule" />
            </div>

            <div className="grid-2" style={{ marginBottom: 'var(--step-2)' }}>
              <select value={units} onChange={(event) => setUnits(event.target.value)}>
                <option value="ground">Ground units (m)</option>
                <option value="pixels">Pixels</option>
              </select>
              <select value={show} onChange={(event) => setShow(event.target.value)}>
                <option value="all">All points</option>
                <option value="gcp">Control only</option>
                <option value="check">Check only</option>
                <option value="tie">Tie only</option>
              </select>
            </div>

            {sorted.length === 0 ? (
              <div className="empty-note">No residuals for this selection.</div>
            ) : (
              <div className="scroll-y" style={{ maxHeight: 320 }}>
                <table className="table">
                  <thead>
                    <tr>
                      <th onClick={() => toggleSort('pointId')}>Point</th>
                      <th onClick={() => toggleSort('imageName')}>Image</th>
                      <th className="num" onClick={() => toggleSort('dx')}>dX</th>
                      <th className="num" onClick={() => toggleSort('dy')}>dY</th>
                      <th className="num" onClick={() => toggleSort('magnitude')}>Res</th>
                    </tr>
                  </thead>
                  <tbody>
                    {sorted.slice(0, 250).map((row, index) => (
                      <tr
                        key={`${row.imageId}-${row.pointId}-${index}`}
                        onMouseEnter={() => setSelectedPoint(row.pointId)}
                        onMouseLeave={() => setSelectedPoint(null)}
                      >
                        <td>
                          {row.pointId}
                          {row.kind === 'check' && <span className="dim"> ck</span>}
                        </td>
                        <td className="truncate" style={{ maxWidth: 90 }}>{row.imageName}</td>
                        <td className="num">{row.dx?.toFixed(3)}</td>
                        <td className="num">{row.dy?.toFixed(3)}</td>
                        <td className={`num ${row.magnitude > tolerance ? 'over'
                          : row.magnitude > tolerance * 0.7 ? 'near' : ''}`}>
                          {row.magnitude?.toFixed(3)}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}

            <div className="field__hint" style={{ marginTop: 6 }}>
              Residuals above {tolerance}
              {units === 'ground' ? ' m' : ' px'} are highlighted.
            </div>
          </section>
        </>
      )}
    </>
  )
}

// Must match ADJUSTMENT_DEFAULTS in the engine.
const ADJUSTMENT_DEFAULTS = {
  controlSigmaXY: 0.5,
  controlSigmaZ: null,
  imageSigmaPx: 0.5,
  tieSigmaPx: 0.3,
  selfCalibration: 'none',
  useExteriorObservations: true,
  eoSigmaXY: 0.1,
  eoSigmaZ: 0.15,
  eoSigmaAngleDeg: null,
  gnssShift: false,
  robust: true,
  autoRejectTies: true,
}

const SELF_CALIBRATION = [
  ['none', 'None (certificate as given)'],
  ['interior', 'Interior: f, x₀, y₀'],
  ['radial', 'Interior and radial: + k1, k2'],
  ['brown', 'Brown: + k3, p1, p2'],
  ['lens', 'Brown, focal length held (drone, no control)'],
  ['full', 'Brown and affinity: + b1, b2'],
]

/**
 * How the block is weighted and what it may estimate. Saved with the project,
 * so the block is solved the same way every time until someone changes it.
 */
function AdjustmentSettings({ project, call }) {
  const saved = { ...ADJUSTMENT_DEFAULTS, ...(project?.adjustment || {}) }
  const [draft, setDraft] = useState(saved)
  useEffect(() => { setDraft({ ...ADJUSTMENT_DEFAULTS, ...(project?.adjustment || {}) }) },
    [project?.id, JSON.stringify(project?.adjustment || {})])

  const hasGnss = (project?.images || []).some((image) => image.exteriorObserved)

  function save(patch) {
    const next = { ...draft, ...patch }
    setDraft(next)
    call(() => api.project.patch({ adjustment: next }))
  }

  function numberField(key, label, unit, hint,
    { optional = false, step = '0.01', blank = 'not used' } = {}) {
    return (
      <div className="field">
        <label className="field__label">
          <span>{label}{hint && <InfoTip>{hint}</InfoTip>}</span>
          <span className="dim">{unit}</span>
        </label>
        <input
          type="number" className="numeric" step={step} min="0"
          placeholder={optional ? blank : undefined}
          value={draft[key] ?? ''}
          onChange={(event) => setDraft({ ...draft, [key]: event.target.value })}
          onBlur={(event) => {
            const value = event.target.value === '' ? null : Number(event.target.value)
            if (!optional && (value == null || !(value > 0))) {
              setDraft({ ...draft, [key]: saved[key] })
              return
            }
            save({ [key]: value })
          }}
        />
      </div>
    )
  }

  return (
    <details className="advanced" style={{ marginBottom: 'var(--step-2)' }}>
      <summary>Adjustment settings</summary>
      <div className="advanced__body">
        <div className="field__hint" style={{ marginBottom: 6 }}>
          A-priori standard deviations. With realistic values sigma-0 comes out near 1.
        </div>
        <div className="grid-2">
          {numberField('controlSigmaXY', 'Control, horizontal', 'm',
            'Precision of the control coordinates in X and Y. A survey file with a precision column overrides this per point.')}
          {numberField('controlSigmaZ', 'Control, vertical', 'm',
            'Precision of control heights. Heights read from a DEM are usually poorer than positions. Blank: as horizontal.',
            { optional: true, blank: 'as horizontal' })}
          {numberField('imageSigmaPx', 'Control measurements', 'px',
            'Precision of a control or check point clicked on the photo.', { step: '0.05' })}
          {numberField('tieSigmaPx', 'Tie point measurements', 'px',
            'Precision of automatic tie points. Least-squares matching reaches 0.1 to 0.3 px.',
            { step: '0.05' })}
        </div>

        <div className="field">
          <label className="field__label">
            <span>
              Self-calibration
              <InfoTip>
                <p>Estimates corrections to the camera alongside the orientations:
                focal length and principal point, then radial (k) and decentring (p)
                distortion, then affinity (b).</p>
                <p>Needs strong geometry: many well-spread tie points, control in
                height, ideally several strips. On a single short strip leave it off.
                Each parameter is reported with its standard deviation; one smaller than
                three of those is not significant.</p>
              </InfoTip>
            </span>
          </label>
          <select value={draft.selfCalibration}
                  onChange={(event) => save({ selfCalibration: event.target.value })}>
            {SELF_CALIBRATION.map(([value, label]) => (
              <option key={value} value={value}>{label}</option>
            ))}
          </select>
        </div>

        {hasGnss && (
          <>
            <Toggle checked={draft.useExteriorObservations}
                    onChange={(value) => save({ useExteriorObservations: value })}>
              Use imported camera positions (GNSS/IMU) as observations
            </Toggle>
            {draft.useExteriorObservations && (
              <>
                <div className="grid-3">
                  {numberField('eoSigmaXY', 'Position, horiz.', 'm')}
                  {numberField('eoSigmaZ', 'Position, vert.', 'm')}
                  {numberField('eoSigmaAngleDeg', 'Attitude', '°',
                    'Leave blank to use positions only.', { optional: true, step: '0.001' })}
                </div>
                <Toggle checked={draft.gnssShift} onChange={(value) => save({ gnssShift: value })}>
                  Solve a constant GNSS shift
                  <InfoTip>One offset for the whole flight, for a datum or antenna
                  offset error common to every position. Needs some ground control.</InfoTip>
                </Toggle>
              </>
            )}
          </>
        )}

        <Toggle checked={draft.robust} onChange={(value) => save({ robust: value })}>
          Robust estimation
          <InfoTip>Limits the pull of gross errors so they stand out instead of
          spreading through the block. Statistics are always those of ordinary least
          squares at the solution.</InfoTip>
        </Toggle>
        <Toggle checked={draft.autoRejectTies} onChange={(value) => save({ autoRejectTies: value })}>
          Remove tie points that fail data snooping
          <InfoTip>Automatic tie points whose standardised residual exceeds 3.29 are
          removed and the block solved again. Control and check points are only flagged:
          they are survey data.</InfoTip>
        </Toggle>
      </div>
    </details>
  )
}

function Toggle({ checked, onChange, children }) {
  return (
    <label className="field field--row">
      <span className="field__label"><span>{children}</span></span>
      <input type="checkbox" checked={!!checked}
             onChange={(event) => onChange(event.target.checked)} />
    </label>
  )
}

function metres(value, digits = 3) {
  return value == null || Number.isNaN(value) ? '—' : `${value.toFixed(digits)}`
}

/** Positional accuracy from the check points, as the standards state it. */
function AccuracySection({ accuracy }) {
  return (
    <section className="section">
      <div className="section__head">
        <span className="section__title">
          Accuracy
          <InfoTip>
            <p>From independent check points only: control points are part of the
            solution, so they measure its fit, not its accuracy.</p>
            <p>RMSE horizontal and vertical as in the ASPRS Positional Accuracy Standards
            (2023); 95% confidence as in the NSSDA (1.7308 × RMSE<sub>r</sub>, 1.96 ×
            RMSE<sub>z</sub>). A mean far from zero shows a systematic offset.</p>
          </InfoTip>
        </span>
        <span className="section__rule" />
        <span className="chip">{accuracy.checkPoints} check</span>
      </div>
      <table className="table">
        <thead>
          <tr><th />
            <th className="num">X</th><th className="num">Y</th><th className="num">Z</th>
          </tr>
        </thead>
        <tbody>
          <tr><td>RMSE m</td>
            <td className="num">{metres(accuracy.rmseX)}</td>
            <td className="num">{metres(accuracy.rmseY)}</td>
            <td className="num">{metres(accuracy.rmseZ)}</td></tr>
          <tr><td>Mean m</td>
            <td className="num">{metres(accuracy.meanX)}</td>
            <td className="num">{metres(accuracy.meanY)}</td>
            <td className="num">{metres(accuracy.meanZ)}</td></tr>
        </tbody>
      </table>
      <div className="measures" style={{ marginTop: 'var(--step-2)' }}>
        <div className="measure">
          <div className="measure__name">RMSE horizontal</div>
          <div className="measure__value">{metres(accuracy.rmseHorizontal, 2)}m</div>
        </div>
        <div className="measure">
          <div className="measure__name">RMSE vertical</div>
          <div className="measure__value">{metres(accuracy.rmseVertical, 2)}m</div>
        </div>
        <div className="measure">
          <div className="measure__name">95% horizontal</div>
          <div className="measure__value">{metres(accuracy.nssdaHorizontal95, 2)}m</div>
        </div>
        <div className="measure">
          <div className="measure__name">95% vertical</div>
          <div className="measure__value">{metres(accuracy.nssdaVertical95, 2)}m</div>
        </div>
      </div>
      {accuracy.checkPointsVertical < accuracy.checkPoints && (
        <div className="field__hint" style={{ marginTop: 6 }}>
          {accuracy.checkPoints - accuracy.checkPointsVertical} of the check points are measured
          on one photo only, so they check position at their surveyed height but not height.
          Measure them on a second photo to check height too.
        </div>
      )}
      {!accuracy.enoughCheckPoints && (
        <div className="field__hint" style={{ marginTop: 6 }}>
          The standards recommend at least 20 check points; with {accuracy.checkPoints} these
          figures are indicative.
        </div>
      )}
    </section>
  )
}

/** Camera corrections the adjustment estimated, with their significance. */
function SelfCalibrationSection({ calibration }) {
  return (
    <section className="section">
      <div className="section__head">
        <span className="section__title">
          Self-calibration
          <InfoTip>
            <p>Corrections to the certificate, estimated in the adjustment and used for
            every product from now on. f, x₀ and y₀ in millimetres; k, p and b relative to
            the format radius ({calibration.radiusMm?.toFixed(1)} mm).</p>
            <p>A value smaller than three standard deviations is not significant: the block
            cannot tell it from zero, and the certificate value stands.</p>
          </InfoTip>
        </span>
        <span className="section__rule" />
      </div>
      <table className="table">
        <thead>
          <tr><th>Parameter</th><th className="num">Value</th><th className="num">± 1σ</th><th /></tr>
        </thead>
        <tbody>
          {Object.entries(calibration.values).map(([name, value]) => {
            const sigma = calibration.sigmas?.[name]
            const significant = sigma ? Math.abs(value) > 3 * sigma : null
            return (
              <tr key={name}>
                <td>{name === 'df' ? 'Δf' : name === 'dx0' ? 'Δx₀' : name === 'dy0' ? 'Δy₀' : name}</td>
                <td className="num">{value.toExponential(3)}</td>
                <td className="num">{sigma != null ? sigma.toExponential(2) : '—'}</td>
                <td>{significant == null ? '' : significant
                  ? <span className="chip chip--good">significant</span>
                  : <span className="dim">not significant</span>}</td>
              </tr>
            )
          })}
        </tbody>
      </table>
    </section>
  )
}
