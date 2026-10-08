import { useEffect, useState } from 'react'
import { useStore } from '../../state/store'
import { api } from '../../lib/api'
import InfoTip from '../InfoTip'
import OutputRows from '../OutputRows'
import { useReveal } from '../../lib/reveal'

/**
 * Stereo DEM extraction.
 *
 * The pair list leads with base-to-height ratio and a plain verdict on it,
 * because B/H predicts DEM quality better than any other single number and
 * knowing it up front saves extracting a surface that was never going to work.
 */
export default function DemPanel() {
  const project = useStore((s) => s.project)
  const call = useStore((s) => s.call)

  const [pairs, setPairs] = useState([])
  const [selected, setSelected] = useState([])
  const [loading, setLoading] = useState(false)
  const [options, setOptions] = useState({
    method: 'sgm',
    detail: 'high',
    terrain: 'rolling',
    smoothing: 'medium',
    applyWallis: false,
    pixelSampling: 4,
    minElevation: '',
    maxElevation: '',
  })

  const stereoDems = project?.stereoDems || []
  const stereoPair = useStore((s) => s.stereoPair)
  const setStereoPair = useStore((s) => s.setStereoPair)
  const outputView = useStore((s) => s.outputView)
  const showOutput = useStore((s) => s.showOutput)
  const closeOutput = useStore((s) => s.closeOutput)
  const popOutOutput = useStore((s) => s.popOutOutput)
  const generatedRef = useReveal('dem')

  useEffect(() => {
    setLoading(true)
    api.dem
      .pairs()
      .then((result) => {
        setPairs(result.pairs)
        setSelected(result.pairs.filter((p) => p.quality === 'good')
          .map((p) => `${p.leftId}|${p.rightId}`))
        // Show a pair on the canvas straight away: the one already shown if
        // it is still a pair, otherwise the first good one.
        const current = useStore.getState().stereoPair
        const stillPair = current && result.pairs.some(
          (p) => p.leftId === current.leftId && p.rightId === current.rightId)
        if (!stillPair && result.pairs.length) {
          const first = result.pairs.find((p) => p.quality === 'good') || result.pairs[0]
          setStereoPair({ leftId: first.leftId, rightId: first.rightId })
        }
      })
      .catch(() => setPairs([]))
      .finally(() => setLoading(false))
  }, [project?.model?.solvedAt])

  if (!project?.model) {
    return (
      <div className="empty-note">
        A solved sensor model is required.
      </div>
    )
  }

  function toggle(key) {
    setSelected((current) =>
      current.includes(key) ? current.filter((k) => k !== key) : [...current, key])
  }

  const chosen = pairs.filter((p) => selected.includes(`${p.leftId}|${p.rightId}`))

  return (
    <>
      {project.droneBlock?.surface && (
        <DroneDense block={project.droneBlock} call={call} />
      )}

      <section className="section">
        <div className="section__head">
          <span className="section__title">Stereo pairs</span>
          <span className="section__rule" />
          {loading && <span className="chip">scanning</span>}
        </div>

        {pairs.length === 0 ? (
          <div className="empty-note">
            {loading ? 'Identifying stereo pairs…'
              : 'No overlapping pairs with solved orientation.'}
          </div>
        ) : (
          <div className="rows">
            {pairs.map((pair) => {
              const key = `${pair.leftId}|${pair.rightId}`
              const viewing = stereoPair?.leftId === pair.leftId
                && stereoPair?.rightId === pair.rightId
              return (
                // The row shows the pair side by side; the checkbox decides
                // whether it is extracted.
                <div
                  key={key}
                  className={`row ${viewing ? 'row--active' : ''}`}
                  style={{ cursor: 'pointer' }}
                  title={`${pair.note}. Click to show this pair side by side.`}
                  onClick={() => {
                    setStereoPair({ leftId: pair.leftId, rightId: pair.rightId })
                    if (outputView?.step === 'dem') closeOutput()
                  }}
                >
                  <input
                    type="checkbox"
                    checked={selected.includes(key)}
                    onChange={() => toggle(key)}
                    onClick={(event) => event.stopPropagation()}
                    aria-label={`Extract ${pair.leftName} and ${pair.rightName}`}
                    title="Include in extraction"
                    style={{ width: 14, flex: 'none' }}
                  />
                  <div className="row__main">
                    <div className="row__name truncate">
                      {pair.leftName} → {pair.rightName}
                      <span className={`chip ${pair.quality === 'good' ? 'chip--good'
                        : pair.quality === 'fair' ? 'chip--warn' : 'chip--bad'}`}>
                        B/H {pair.baseHeightRatio.toFixed(2)}
                      </span>
                    </div>
                    <div className="row__meta truncate">
                      base {pair.baselineM.toFixed(0)} m, {pair.note}
                    </div>
                  </div>
                  {viewing && <span className="chip">on canvas</span>}
                </div>
              )
            })}
          </div>
        )}
      </section>

      <section className="section">
        <div className="section__head"><span className="section__title">Matching</span>
          <span className="section__rule" /></div>

        <div className="field">
          <label className="field__label">
            <span>
              Method
              <InfoTip>
                <p>Semi-global matching enforces smoothness along several paths and
                performs better over low-texture ground.</p>
                <p>Normalised cross-correlation is faster but noisier in shadow and
                over water.</p>
              </InfoTip>
            </span>
          </label>
          <select
            value={options.method}
            onChange={(event) => setOptions({ ...options, method: event.target.value })}
          >
            <option value="sgm">Semi-global matching</option>
            <option value="ncc">Normalised cross-correlation</option>
          </select>
        </div>

        <div className="grid-2">
          <div className="field">
            <label className="field__label">
              <span>
                Detail
                <InfoTip>
                  The resolution the photos are matched at, as the long side of the
                  image. Height precision is proportional to the matched pixel's size,
                  so finer detail gives a finer, more precise surface, and takes
                  longer: on a large-format frame extra high takes about ten times as
                  long as high. Worth it only if the reference you check against can
                  show the difference.
                </InfoTip>
              </span>
            </label>
            <select
              value={options.detail}
              onChange={(event) => setOptions({ ...options, detail: event.target.value })}
            >
              <option value="low">Low (2,000 px)</option>
              <option value="medium">Medium (3,000 px)</option>
              <option value="high">High (4,000 px)</option>
              <option value="extra_high">Extra high (8,000 px)</option>
            </select>
          </div>
          <div className="field">
            <label className="field__label">Terrain</label>
            <select
              value={options.terrain}
              onChange={(event) => setOptions({ ...options, terrain: event.target.value })}
            >
              <option value="flat">Flat</option>
              <option value="rolling">Rolling</option>
              <option value="mountainous">Mountainous</option>
            </select>
          </div>
        </div>

        <details className="advanced">
          <summary>Options</summary>
          <div className="advanced__body">
        <div className="grid-2">
          <div className="field">
            <label className="field__label">Smoothing</label>
            <select
              value={options.smoothing}
              onChange={(event) => setOptions({ ...options, smoothing: event.target.value })}
            >
              <option value="none">None</option>
              <option value="low">Low</option>
              <option value="medium">Medium</option>
              <option value="high">High</option>
            </select>
          </div>
          <div className="field">
            <label className="field__label">Pixel sampling</label>
            <input
              type="number" min="1" max="16" className="numeric"
              value={options.pixelSampling}
              onChange={(event) => setOptions({
                ...options, pixelSampling: Number(event.target.value),
              })}
            />
          </div>
        </div>

        <label className="field field--row">
          <span className="field__label">
            <span>
              Wallis filter
              <InfoTip>
                Equalises local contrast to recover texture in shadow and over
                water, at the cost of additional noise elsewhere.
              </InfoTip>
            </span>
          </span>
          <input
            type="checkbox" checked={options.applyWallis}
            onChange={(event) => setOptions({ ...options, applyWallis: event.target.checked })}
          />
        </label>

        <div className="grid-2" style={{ marginTop: 'var(--step-3)' }}>
          <div className="field">
            <label className="field__label">Min elevation</label>
            <input
              className="numeric" placeholder="Automatic" value={options.minElevation}
              onChange={(event) => setOptions({ ...options, minElevation: event.target.value })}
            />
          </div>
          <div className="field">
            <label className="field__label">Max elevation</label>
            <input
              className="numeric" placeholder="Automatic" value={options.maxElevation}
              onChange={(event) => setOptions({ ...options, maxElevation: event.target.value })}
            />
          </div>
        </div>
          </div>
        </details>
      </section>

      <section className="section">
        <button
          className="btn btn--primary btn--block"
          disabled={!chosen.length}
          onClick={() => call(() => api.dem.extract({
            pairs: chosen.map((p) => ({ leftId: p.leftId, rightId: p.rightId })),
            options: {
              ...options,
              minElevation: options.minElevation === '' ? null : Number(options.minElevation),
              maxElevation: options.maxElevation === '' ? null : Number(options.maxElevation),
            },
          }))}
        >
          Extract from {chosen.length} pair{chosen.length === 1 ? '' : 's'}
        </button>
      </section>

      {stereoDems.length > 0 && (
        <section className="section" ref={generatedRef}>
          <div className="section__head"><span className="section__title">Extracted</span>
          <span className="section__rule" /></div>
          <OutputRows
            items={stereoDems.map((entry, index) => ({
              key: `${index}:${entry.path}`,
              path: entry.path,
              name: (entry.merged ? 'Merged: ' : '') + entry.path.split(/[\\/]/).pop(),
              meta: `${entry.resolution?.toFixed(2)} m, ${((entry.coverage || 0) * 100).toFixed(0)}% coverage`,
            }))}
            step="dem"
            current={outputView?.step === 'dem' ? outputView.path : null}
            onView={showOutput}
            onPopOut={popOutOutput}
          />
        </section>
      )}
    </>
  )
}


/**
 * The whole drone block's surface in one step: every well-overlapping pair
 * matched densely and the results fused by vote. The pair list below stays
 * for working pair by pair.
 */
function DroneDense({ block, call }) {
  const [detail, setDetail] = useState('medium')
  const job = useStore((s) => s.jobs.find((j) => j.kind === 'drone_dense'
    && (j.status === 'running' || j.status === 'queued')))
  const dense = block.dense
  const measured = dense ? Math.round(dense.measuredFraction * 100) : 0
  return (
    <section className="section">
      <div className="section__head">
        <span className="section__title">Dense surface</span>
        <span className="section__rule" />
        {dense?.stale && <span className="chip chip--warn">out of date</span>}
        <InfoTip>
          Matches every well-overlapping pair of photos pixel by pixel and fuses the
          results, keeping heights that at least two pairs agree on. Where none agree
          (moving trees, water, deep shadow) the surface from the tie points fills in.
          Orthophotos are then made on it.
        </InfoTip>
      </div>

      {dense && (
        <>
          <div className="measures">
            <div className="measure">
              <div className="measure__name">Cell size</div>
              <div className="measure__value">{dense.resolution} m</div>
            </div>
            <div className="measure">
              <div className="measure__name">Pairs</div>
              <div className="measure__value">{dense.pairs}</div>
            </div>
          </div>
          <div className="field__hint" style={{ marginTop: 'var(--step-2)' }}>
            {measured}% of the ground the pairs cover is measured by two or more of
            them; the rest comes from the tie points. Heights {dense.elevationRange?.[0]?.toFixed(1)} to{' '}
            {dense.elevationRange?.[1]?.toFixed(1)} m. Built {dense.builtAt}.
            {dense.stale && ' The block has been solved again since: rebuild it.'}
          </div>
        </>
      )}

      <div className="field" style={{ marginTop: 'var(--step-3)' }}>
        <label className="field__label">Detail</label>
        <select value={detail} onChange={(event) => setDetail(event.target.value)}>
          <option value="low">Low (fastest)</option>
          <option value="medium">Medium</option>
          <option value="high">High</option>
        </select>
      </div>
      <button
        className={`btn btn--block ${dense && !dense.stale ? '' : 'btn--primary'}`}
        disabled={!!job}
        onClick={() => call(() => api.drone.dense({ detail }), { refresh: false })}
      >
        {job ? (job.message || 'Building…') : dense ? 'Rebuild dense surface' : 'Build dense surface'}
      </button>
    </section>
  )
}
