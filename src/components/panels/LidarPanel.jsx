import { useEffect, useRef, useState } from 'react'
import { useStore } from '../../state/store'
import { api } from '../../lib/api'
import InfoTip from '../InfoTip'

/**
 * LiDAR to terrain model.
 *
 * The class histogram is shown before any filtering choice is made, because
 * "my DTM has holes" is almost always "class 2 was 4% of the returns" — and
 * that is knowable in advance rather than after a long rasterise.
 *
 * Noise can be found and the ground classified here; each writes a new cloud
 * and opens it, and the source file is never changed. Points can be labelled
 * in the side view on the canvas, and a classifier trained on those labels
 * labels the rest. Heights above the ground make a canopy height model over
 * vegetation and a normalised DSM over buildings.
 */

// Classes offered for labelling, most used first.
const LABEL_CLASSES = [2, 3, 4, 5, 6, 7, 18, 9, 1, 13, 14, 15, 17, 10, 11]
export default function LidarPanel() {
  const project = useStore((s) => s.project)
  const call = useStore((s) => s.call)
  const toast = useStore((s) => s.toast)

  const source = useStore((s) => s.lidarCloud)
  const setLidarCloud = useStore((s) => s.setLidarCloud)
  const brush = useStore((s) => s.lidarBrush)
  const setBrush = useStore((s) => s.setLidarBrush)
  const labelsVersion = useStore((s) => s.lidarLabelsVersion)
  const bumpLabels = useStore((s) => s.bumpLidarLabels)
  const reference = useStore((s) => s.reference)
  const [summary, setSummary] = useState(null)
  const [labelCounts, setLabelCounts] = useState(null)
  const [noise, setNoise] = useState({
    method: 'statistical', neighbours: 8, stdRatio: 2.5, radius: 2.0, minNeighbours: 3,
  })
  const [learn, setLearn] = useState({ onlyClasses: [], minConfidence: 0 })
  const [comparison, setComparison] = useState(null)
  const [options, setOptions] = useState({
    cellSize: 1.0,
    classes: [2],
    returns: 'last',
    cellAssignment: 'idw',
    voidFill: 'natural_neighbor',
  })

  const [ground, setGround] = useState({
    cellSize: 1.0, slope: 0.15, window: 18, elevationThreshold: 0.5, lowNoise: true,
  })
  const [height, setHeight] = useState({
    cellSize: 0.5, returns: 'first', method: 'pit_free', dtmPath: '', maxHeight: '',
    tickedOnly: false,
  })

  const outputs = project?.lidar || []
  const clouds = project?.lidarClouds || []

  async function open(path, { keepSection = false } = {}) {
    setLidarCloud(path, { keepSection })
    setSummary(null)
    const result = await call(() => api.lidar.inspect(path), { refresh: false })
    if (result) setSummary(result)
  }

  // Coming back to the step with a cloud still open.
  useEffect(() => {
    if (source && !summary) open(source)
  }, []) // eslint-disable-line react-hooks/exhaustive-deps

  useEffect(() => {
    if (!source || !project) { setLabelCounts(null); return }
    api.lidar.labels(source).then(setLabelCounts).catch(() => setLabelCounts(null))
  }, [source, labelsVersion, project?.lidarClouds?.length]) // eslint-disable-line react-hooks/exhaustive-deps

  const classColours = reference?.lidarClassColours || {}
  const className = (code) => reference?.lidarClasses?.[String(code)] || `Class ${code}`
  const learning = project?.lidarLearning
  const labelledClasses = Object.keys(labelCounts?.byClass || {})
  const readyToTrain = labelledClasses.length >= 2
    && labelledClasses.every((code) => labelCounts.byClass[code] >= 10)

  async function clearLabels() {
    await call(() => api.lidar.setLabels({ path: source, clearAll: true }), { refresh: false })
    bumpLabels()
  }

  function describe(entry) {
    if (entry.kind === 'noise') {
      return `${(entry.noisePoints || 0).toLocaleString()} noise points (${entry.method})`
    }
    if (entry.kind === 'learned') {
      return `classes from labels, ${(entry.pointsChanged || 0).toLocaleString()} points changed`
    }
    return `${(entry.groundFraction * 100).toFixed(1)}% ground`
      + (entry.lowNoisePoints ? `, ${entry.lowNoisePoints.toLocaleString()} low noise` : '')
  }

  async function choose() {
    if (!window.fiducia?.isDesktop) {
      toast('Point clouds can only be opened in the desktop app', 'warn')
      return
    }
    const paths = await window.fiducia.dialog.openFiles({
      title: 'Open a point cloud', multiple: false,
      filters: [{ name: 'LiDAR', extensions: ['las', 'laz'] }],
    })
    if (paths?.length) open(paths[0])
  }

  // When work on the open cloud writes a new cloud, carry on with it.
  const seenClouds = useRef(clouds.length)
  useEffect(() => {
    if (clouds.length > seenClouds.current) {
      const latest = clouds[clouds.length - 1]
      if (latest.source === source) {
        open(latest.outputPath, { keepSection: true })
        toast(`Opened the new cloud: ${describe(latest)}`, 'good')
      }
    }
    seenClouds.current = clouds.length
  }, [clouds.length]) // eslint-disable-line react-hooks/exhaustive-deps

  async function chooseDtm() {
    const paths = await window.fiducia?.dialog.openFiles({
      title: 'Choose a terrain model', multiple: false,
      filters: [{ name: 'Raster', extensions: ['tif', 'tiff', 'img', 'pix'] }],
    })
    if (paths?.length) setHeight({ ...height, dtmPath: paths[0] })
  }

  const groundShare = summary
    ? (summary.classHistogram['2']?.count || 0) / Math.max(summary.pointCount, 1)
    : 0
  const fileName = (path) => path.split(/[\\/]/).pop()

  function toggleClass(code) {
    setOptions((current) => ({
      ...current,
      classes: current.classes.includes(code)
        ? current.classes.filter((c) => c !== code)
        : [...current.classes, code].sort((a, b) => a - b),
    }))
  }

  async function compare(derivedPath) {
    const paths = await window.fiducia?.dialog.openFiles({
      title: 'Choose a reference surface', multiple: false,
      filters: [{ name: 'Raster', extensions: ['tif', 'tiff', 'img', 'pix'] }],
    })
    if (!paths?.length) return
    const result = await call(() => api.lidar.compare(derivedPath, paths[0]),
      { refresh: false })
    if (result) setComparison(result)
  }

  const selectedCount = summary
    ? options.classes.reduce(
        (total, code) => total + (summary.classHistogram[String(code)]?.count || 0), 0)
    : 0

  return (
    <>
      <section className="section">
        <div className="section__head">
          <span className="section__title">Point cloud</span>
          <span className="section__rule" />
          <button className="btn btn--primary btn--sm" onClick={choose}>Open…</button>
        </div>

        {!source ? (
          <div className="empty-note">
            No point cloud open. LAS and LAZ are supported.
          </div>
        ) : (
          <>
            <div className="field__hint truncate" style={{ marginBottom: 6 }}>{source}</div>
            {summary && (
              <>
                <div className="measures">
                  <div className="measure">
                    <div className="measure__name">Points</div>
                    <div className="measure__value" style={{ fontSize: 13 }}>
                      {(summary.pointCount / 1e6).toFixed(2)}M
                    </div>
                  </div>
                  <div className="measure">
                    <div className="measure__name">Spacing</div>
                    <div className="measure__value" style={{ fontSize: 13 }}>
                      {summary.averageSpacing.toFixed(2)} m
                    </div>
                  </div>
                  <div className="measure">
                    <div className="measure__name">Elevation</div>
                    <div className="measure__value" style={{ fontSize: 13 }}>
                      {summary.elevationRange[0].toFixed(0)}–
                      {summary.elevationRange[1].toFixed(0)}
                    </div>
                  </div>
                  <div className="measure">
                    <div className="measure__name">Format</div>
                    <div className="measure__value" style={{ fontSize: 13 }}>
                      LAS {summary.version}
                    </div>
                  </div>
                </div>

                {!summary.crs && (
                  <div className="field__hint"
                       style={{ color: 'var(--signal-warn)', marginTop: 6 }}>
                    No coordinate system is declared. The project output projection
                    is assumed.
                  </div>
                )}
              </>
            )}
          </>
        )}

        {clouds.length > 0 && (
          <div className="rows" style={{ marginTop: 'var(--step-3)' }}>
            {clouds.map((entry, index) => (
              <div key={index} className={`row ${entry.outputPath === source ? 'row--active' : ''}`}>
                <div className="row__main">
                  <div className="row__name truncate">{fileName(entry.outputPath)}</div>
                  <div className="row__meta">{describe(entry)}</div>
                </div>
                <div className="row__actions">
                  {entry.outputPath !== source && (
                    <button className="btn btn--ghost btn--sm" onClick={() => open(entry.outputPath)}>
                      Open
                    </button>
                  )}
                </div>
              </div>
            ))}
          </div>
        )}
      </section>

      {summary && (
        <>
          <section className="section">
            <div className="section__head">
              <span className="section__title">
                Noise
                <InfoTip>
                  Labels points that belong to no surface, such as birds, multipath
                  and sensor spikes, as low noise (class 7) below the ground or high
                  noise (class 18) above it. Nothing is deleted, and every other tool
                  leaves noise out. Statistical: points far from their neighbours
                  compared with the cloud's usual spacing. Isolated: points with too
                  few neighbours within a radius.
                </InfoTip>
              </span>
              <span className="section__rule" />
            </div>
            <div className="grid-2">
              <div className="field">
                <label className="field__label">Method</label>
                <select value={noise.method} onChange={(e) => setNoise({ ...noise, method: e.target.value })}>
                  <option value="statistical">Statistical</option>
                  <option value="isolated">Isolated points</option>
                </select>
              </div>
              {noise.method === 'statistical' ? (
                <div className="field">
                  <label className="field__label">
                    Strictness <span className="dim">σ</span>
                    <InfoTip>How many standard deviations beyond the usual spacing counts as noise. Lower finds more.</InfoTip>
                  </label>
                  <input type="number" step="0.5" min="0.5" className="numeric" value={noise.stdRatio}
                    onChange={(e) => setNoise({ ...noise, stdRatio: Number(e.target.value) })} />
                </div>
              ) : (
                <div className="field">
                  <label className="field__label">Radius <span className="dim">m</span></label>
                  <input type="number" step="0.5" min="0.1" className="numeric" value={noise.radius}
                    onChange={(e) => setNoise({ ...noise, radius: Number(e.target.value) })} />
                </div>
              )}
            </div>
            <details className="advanced">
              <summary>Options</summary>
              <div className="advanced__body">
                {noise.method === 'statistical' ? (
                  <div className="field">
                    <label className="field__label">Neighbours</label>
                    <input type="number" step="1" min="2" className="numeric" value={noise.neighbours}
                      onChange={(e) => setNoise({ ...noise, neighbours: Number(e.target.value) })} />
                  </div>
                ) : (
                  <div className="field">
                    <label className="field__label">Fewest neighbours</label>
                    <input type="number" step="1" min="1" className="numeric" value={noise.minNeighbours}
                      onChange={(e) => setNoise({ ...noise, minNeighbours: Number(e.target.value) })} />
                  </div>
                )}
              </div>
            </details>
            <button className="btn btn--block"
              onClick={() => call(() => api.lidar.noise({ path: source, ...noise }))}>
              Find noise
            </button>
          </section>

          <section className="section">
            <div className="section__head">
              <span className="section__title">
                Ground
                <InfoTip>
                  Finds the ground with the Simple Morphological Filter (SMRF) and
                  writes a new cloud with it labelled as class 2. Other classes are
                  kept, and points far below the ground are labelled as low noise
                  (class 7). The opened file is not changed.
                </InfoTip>
              </span>
              <span className="section__rule" />
              <span className="chip">
                {groundShare > 0 ? `${(groundShare * 100).toFixed(1)}% ground` : 'no ground class'}
              </span>
            </div>

            <details className="advanced">
              <summary>Options</summary>
              <div className="advanced__body">
                <div className="grid-2">
                  <div className="field">
                    <label className="field__label">Cell size <span className="dim">m</span></label>
                    <input type="number" step="0.25" min="0.1" className="numeric"
                      value={ground.cellSize}
                      onChange={(e) => setGround({ ...ground, cellSize: Number(e.target.value) })} />
                  </div>
                  <div className="field">
                    <label className="field__label">
                      Largest object <span className="dim">m</span>
                      <InfoTip>The width of the largest building or object to remove.</InfoTip>
                    </label>
                    <input type="number" step="1" min="1" className="numeric"
                      value={ground.window}
                      onChange={(e) => setGround({ ...ground, window: Number(e.target.value) })} />
                  </div>
                  <div className="field">
                    <label className="field__label">
                      Slope <span className="dim">rise/run</span>
                      <InfoTip>The steepest ground expected. Raise it for steep terrain.</InfoTip>
                    </label>
                    <input type="number" step="0.05" min="0.01" className="numeric"
                      value={ground.slope}
                      onChange={(e) => setGround({ ...ground, slope: Number(e.target.value) })} />
                  </div>
                  <div className="field">
                    <label className="field__label">
                      Tolerance <span className="dim">m</span>
                      <InfoTip>How far off the ground surface a point can be and still count as ground.</InfoTip>
                    </label>
                    <input type="number" step="0.05" min="0.05" className="numeric"
                      value={ground.elevationThreshold}
                      onChange={(e) => setGround({ ...ground, elevationThreshold: Number(e.target.value) })} />
                  </div>
                </div>
                <label className="field field--row">
                  <span className="field__label">Label points far below the ground as low noise</span>
                  <input type="checkbox" checked={ground.lowNoise}
                    onChange={(e) => setGround({ ...ground, lowNoise: e.target.checked })} />
                </label>
              </div>
            </details>

            <button
              className={`btn btn--block ${groundShare > 0 ? '' : 'btn--primary'}`}
              onClick={() => call(() => api.lidar.classifyGround({ path: source, ...ground }))}
            >
              {groundShare > 0 ? 'Find the ground again' : 'Find the ground'}
            </button>

          </section>


          <section className="section">
            <div className="section__head">
              <span className="section__title">
                Classes from labels
                <InfoTip>
                  Label a few points of each class in the side view, then train: a
                  random forest learns from each point's height above the ground,
                  its return, and the shape of its neighbourhood, and labels the
                  rest of the cloud. Labelled points keep their labels. Places it is
                  least sure of are marked on the plan; label there and train again.
                </InfoTip>
              </span>
              <span className="section__rule" />
              {labelCounts?.count > 0 && (
                <span className="chip">{labelCounts.count.toLocaleString()} labelled</span>
              )}
            </div>

            <div className="field">
              <label className="field__label">Label as</label>
              <div className="label-picker">
                <button className={`label-picker__item ${brush === null ? 'label-picker__item--on' : ''}`}
                  onClick={() => setBrush(null)} title="Pan the side view instead of labelling">
                  Pan
                </button>
                {LABEL_CLASSES.map((code) => (
                  <button key={code}
                    className={`label-picker__item ${brush === code ? 'label-picker__item--on' : ''}`}
                    onClick={() => setBrush(code)}
                    title={`Label points as ${className(code)} (class ${code})`}>
                    <i className="swatch-dot" style={{ background: classColours[String(code)] }} />
                    {className(code)}
                    {labelCounts?.byClass?.[String(code)] ? (
                      <span className="dim">{labelCounts.byClass[String(code)].toLocaleString()}</span>
                    ) : null}
                  </button>
                ))}
              </div>
              <div className="field__hint">
                Drag across the plan to draw a section, then drag a box over points
                in the side view. Shift-drag clears labels.
              </div>
            </div>

            <details className="advanced">
              <summary>Options</summary>
              <div className="advanced__body">
                <div className="field">
                  <label className="field__label">
                    Only change points now in
                    <InfoTip>Leave all unticked to let any point change. Points labelled noise stay noise unless noise is among your labels.</InfoTip>
                  </label>
                  <div className="label-picker">
                    {Object.keys(summary.classHistogram).map(Number).sort((a, b) => a - b).map((code) => (
                      <label key={code} className={`label-picker__item ${learn.onlyClasses.includes(code) ? 'label-picker__item--on' : ''}`}>
                        <input type="checkbox" checked={learn.onlyClasses.includes(code)}
                          onChange={() => setLearn({
                            ...learn,
                            onlyClasses: learn.onlyClasses.includes(code)
                              ? learn.onlyClasses.filter((c) => c !== code)
                              : [...learn.onlyClasses, code],
                          })} />
                        {className(code)}
                      </label>
                    ))}
                  </div>
                </div>
                <div className="field">
                  <label className="field__label">
                    Minimum confidence
                    <InfoTip>A point the forest is less sure of than this keeps its class.</InfoTip>
                  </label>
                  <input type="number" step="0.05" min="0" max="0.95" className="numeric"
                    value={learn.minConfidence}
                    onChange={(e) => setLearn({ ...learn, minConfidence: Number(e.target.value) })} />
                </div>
              </div>
            </details>

            <div className="grid-2">
              <button className="btn btn--primary" disabled={!readyToTrain}
                title={readyToTrain ? 'Train on the labels and label the whole cloud'
                  : 'Label at least 10 points of at least two classes first'}
                onClick={() => call(() => api.lidar.learn({
                  path: source,
                  changeClasses: learn.onlyClasses.length ? learn.onlyClasses : null,
                  minConfidence: learn.minConfidence,
                }))}>
                Train and apply
              </button>
              <button className="btn" disabled={!labelCounts?.count} onClick={clearLabels}>
                Clear labels
              </button>
            </div>

            {learning && (
              <div style={{ marginTop: 'var(--step-3)' }}>
                <div className="field__hint" style={{ marginBottom: 6 }}>
                  Last training: {learning.pointsLabelled.toLocaleString()} labels,
                  {' '}{learning.pointsChanged.toLocaleString()} points changed. Recall is the
                  share of each class the forest finds, judged on labels each tree never saw.
                </div>
                <div className="rows">
                  {learning.classes.map((entry) => (
                    <div key={entry.class} className="row">
                      <i className="swatch-dot" style={{ background: classColours[String(entry.class)] }} />
                      <div className="row__main">
                        <div className="row__name">{entry.label}</div>
                        <div className="row__meta">
                          {entry.labelled.toLocaleString()} labelled, {entry.result.toLocaleString()} in the result
                        </div>
                      </div>
                      <div className={`row__value ${entry.recall >= 0.9 ? 'measure__value--good'
                        : entry.recall >= 0.7 ? 'measure__value--warn' : 'measure__value--bad'}`}
                        title={`Recall ${(entry.recall * 100).toFixed(1)}%, precision ${(entry.precision * 100).toFixed(1)}%`}>
                        {(entry.recall * 100).toFixed(0)}%
                      </div>
                    </div>
                  ))}
                </div>
                <div className="field__hint" style={{ marginTop: 6 }}>
                  Most telling: {learning.importance.slice(0, 3).map((item) => item.feature).join(', ')}.
                  {learning.uncertainSpots?.length
                    ? ` ${learning.uncertainSpots.length} uncertain places are marked ? on the plan.` : ''}
                </div>
              </div>
            )}
          </section>

          <section className="section">
            <div className="section__head">
              <span className="section__title">
                Classification filter
                <InfoTip>
                  Ground (class 2) alone produces a terrain model. Include
                  vegetation and buildings for a surface model.
                </InfoTip>
              </span>
          <span className="section__rule" />
              <span className="chip">
                {(selectedCount / 1e6).toFixed(2)}M selected
              </span>
            </div>

            <div className="rows">
              {Object.entries(summary.classHistogram)
                .sort((a, b) => b[1].count - a[1].count)
                .map(([code, info]) => (
                  <label
                    key={code}
                    className={`row ${options.classes.includes(Number(code)) ? 'row--active' : ''}`}
                    style={{ cursor: 'pointer' }}
                  >
                    <input
                      type="checkbox"
                      checked={options.classes.includes(Number(code))}
                      onChange={() => toggleClass(Number(code))}
                      style={{ width: 14, flex: 'none' }}
                    />
                    <div className="row__main">
                      <div className="row__name">{info.label}</div>
                      <div className="row__meta">
                        class {code}, {(info.count / 1e6).toFixed(2)}M,
                        {' '}{((info.count / summary.pointCount) * 100).toFixed(1)}%
                      </div>
                    </div>
                  </label>
                ))}
            </div>

          </section>

          <section className="section">
            <div className="section__head"><span className="section__title">Rasterising</span>
          <span className="section__rule" /></div>

            <div className="grid-2">
              <div className="field">
                <label className="field__label">Cell size <span className="dim">m</span></label>
                <input
                  type="number" step="0.25" className="numeric"
                  value={options.cellSize}
                  onChange={(event) => setOptions({
                    ...options, cellSize: Number(event.target.value),
                  })}
                />
              </div>
              <div className="field">
                <label className="field__label">Returns</label>
                <select
                  value={options.returns}
                  onChange={(event) => setOptions({ ...options, returns: event.target.value })}
                >
                  <option value="all">All</option>
                  <option value="first">First</option>
                  <option value="last">Last</option>
                  <option value="single">Single only</option>
                </select>
              </div>
            </div>

            <div className="field__hint" style={{ marginTop: -4, marginBottom: 'var(--step-3)' }}>
              Average point spacing: {summary.averageSpacing.toFixed(2)} m.
            </div>

            <details className="advanced">
              <summary>Options</summary>
              <div className="advanced__body">

            <div className="grid-2">
              <div className="field">
                <label className="field__label">Cell assignment</label>
                <select
                  value={options.cellAssignment}
                  onChange={(event) => setOptions({
                    ...options, cellAssignment: event.target.value,
                  })}
                >
                  <option value="idw">Inverse distance</option>
                  <option value="mean">Mean</option>
                  <option value="minimum">Minimum</option>
                  <option value="maximum">Maximum</option>
                </select>
              </div>
              <div className="field">
                <label className="field__label">Void fill</label>
                <select
                  value={options.voidFill}
                  onChange={(event) => setOptions({ ...options, voidFill: event.target.value })}
                >
                  <option value="natural_neighbor">Natural neighbour</option>
                  <option value="linear">Linear</option>
                  <option value="nearest">Nearest</option>
                  <option value="none">None</option>
                </select>
              </div>
            </div>
              </div>
            </details>

            <button
              className="btn btn--primary btn--block"
              disabled={!options.classes.length}
              onClick={() => call(() => api.lidar.rasterize({ path: source, ...options }))}
            >
              Create elevation raster
            </button>
          </section>

          <section className="section">
            <div className="section__head">
              <span className="section__title">
                Height above ground
                <InfoTip>
                  Each point's height above the ground, rasterised: a canopy height
                  model over vegetation, a normalised DSM over buildings. Highest
                  point takes the top return in each cell. Pit-free (Khosravipour
                  et al., 2014) layers triangulated surfaces so that returns which
                  slip deep into a crown cannot punch holes in it.
                </InfoTip>
              </span>
              <span className="section__rule" />
            </div>

            <div className="grid-2">
              <div className="field">
                <label className="field__label">Cell size <span className="dim">m</span></label>
                <input type="number" step="0.25" min="0.1" className="numeric"
                  value={height.cellSize}
                  onChange={(e) => setHeight({ ...height, cellSize: Number(e.target.value) })} />
              </div>
              <div className="field">
                <label className="field__label">Method</label>
                <select value={height.method}
                  onChange={(e) => setHeight({ ...height, method: e.target.value })}>
                  <option value="pit_free">Pit-free</option>
                  <option value="highest">Highest point</option>
                </select>
              </div>
            </div>

            <div className="field">
              <label className="field__label">Ground from</label>
              <select value={height.dtmPath ? 'dtm' : 'cloud'}
                onChange={(e) => (e.target.value === 'dtm' ? chooseDtm()
                  : setHeight({ ...height, dtmPath: '' }))}>
                <option value="cloud">Ground points in this cloud</option>
                <option value="dtm">{height.dtmPath ? fileName(height.dtmPath) : 'A terrain model…'}</option>
              </select>
              {!height.dtmPath && groundShare === 0 && (
                <div className="field__hint" style={{ color: 'var(--signal-warn)' }}>
                  This cloud has no ground class. Find the ground first, or choose a terrain model.
                </div>
              )}
            </div>

            <details className="advanced">
              <summary>Options</summary>
              <div className="advanced__body">
                <div className="grid-2">
                  <div className="field">
                    <label className="field__label">Returns</label>
                    <select value={height.returns}
                      onChange={(e) => setHeight({ ...height, returns: e.target.value })}>
                      <option value="first">First</option>
                      <option value="all">All</option>
                      <option value="last">Last</option>
                      <option value="single">Single only</option>
                    </select>
                  </div>
                  <div className="field">
                    <label className="field__label">
                      Maximum height <span className="dim">m</span>
                      <InfoTip>Points higher than this above the ground are left out, such as birds or cloud returns.</InfoTip>
                    </label>
                    <input type="number" step="1" min="0" className="numeric" placeholder="none"
                      value={height.maxHeight}
                      onChange={(e) => setHeight({ ...height, maxHeight: e.target.value })} />
                  </div>
                </div>
                <label className="field field--row">
                  <span className="field__label">Only the classes ticked above</span>
                  <input type="checkbox" checked={height.tickedOnly}
                    onChange={(e) => setHeight({ ...height, tickedOnly: e.target.checked })} />
                </label>
              </div>
            </details>

            <button
              className="btn btn--primary btn--block"
              disabled={!height.dtmPath && groundShare === 0}
              onClick={() => call(() => api.lidar.height({
                path: source,
                cellSize: height.cellSize,
                returns: height.returns,
                method: height.method,
                dtmPath: height.dtmPath || null,
                maxHeight: height.maxHeight === '' ? null : Number(height.maxHeight),
                classes: height.tickedOnly ? options.classes : [],
              }))}
            >
              Create height model
            </button>
          </section>
        </>
      )}

      {outputs.length > 0 && (
        <section className="section">
          <div className="section__head"><span className="section__title">Surfaces</span>
          <span className="section__rule" /></div>
          <div className="rows">
            {outputs.map((entry, index) => (
              <div key={index} className="row">
                <div className="row__main">
                  <div className="row__name truncate">
                    {fileName(entry.outputPath)}
                  </div>
                  <div className="row__meta">
                    {entry.kind === 'height'
                      ? `height above ground, ${entry.method === 'pit_free' ? 'pit-free' : 'highest point'}, `
                      : ''}
                    {entry.cellSize} m, {(entry.pointsUsed / 1e6).toFixed(2)}M points
                  </div>
                </div>
                <div className="row__actions">
                  <button
                    className="btn btn--ghost btn--sm"
                    onClick={() => compare(entry.outputPath)}
                    title="Compare with a reference surface"
                  >
                    Validate
                  </button>
                  <button
                    className="btn btn--ghost btn--sm"
                    onClick={() => window.fiducia?.shell.showItem(entry.outputPath)}
                  >
                    ↗
                  </button>
                </div>
              </div>
            ))}
          </div>
        </section>
      )}

      {comparison && (
        <section className="section">
          <div className="section__head">
            <span className="section__title">Validation</span>
          <span className="section__rule" />
            <button className="btn btn--ghost btn--sm" onClick={() => setComparison(null)}>
              ✕
            </button>
          </div>
          <div className="measures">
            <div className="measure">
              <div className="measure__name">RMSE</div>
              <div className="measure__value">{comparison.rmse.toFixed(2)} m</div>
            </div>
            <div className="measure">
              <div className="measure__name">R²</div>
              <div className={`measure__value ${comparison.rSquared > 0.9 ? 'measure__value--good'
                : comparison.rSquared > 0.7 ? 'measure__value--warn' : 'measure__value--bad'}`}>
                {comparison.rSquared.toFixed(4)}
              </div>
            </div>
            <div className="measure">
              <div className="measure__name">Mean error</div>
              <div className="measure__value" style={{ fontSize: 13 }}>
                {comparison.meanError.toFixed(2)} m
              </div>
            </div>
            <div className="measure">
              <div className="measure__name">Samples</div>
              <div className="measure__value" style={{ fontSize: 13 }}>
                {comparison.sampleCount.toLocaleString()}
              </div>
            </div>
          </div>
          <div className="field__hint" style={{ marginTop: 6 }}>
            Regression: derived = {comparison.slope.toFixed(4)} × reference
            {comparison.intercept >= 0 ? ' + ' : ' − '}
            {Math.abs(comparison.intercept).toFixed(2)}
          </div>
        </section>
      )}
    </>
  )
}
