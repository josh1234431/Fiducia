import { useState } from 'react'
import { useStore } from '../../state/store'
import { api } from '../../lib/api'
import InfoTip from '../InfoTip'

/**
 * LiDAR to terrain model.
 *
 * The class histogram is shown before any filtering choice is made, because
 * "my DTM has holes" is almost always "class 2 was 4% of the returns" — and
 * that is knowable in advance rather than after a long rasterise.
 */
export default function LidarPanel() {
  const project = useStore((s) => s.project)
  const call = useStore((s) => s.call)
  const toast = useStore((s) => s.toast)

  const [source, setSource] = useState(null)
  const [summary, setSummary] = useState(null)
  const [comparison, setComparison] = useState(null)
  const [options, setOptions] = useState({
    cellSize: 1.0,
    classes: [2],
    returns: 'last',
    cellAssignment: 'idw',
    voidFill: 'natural_neighbor',
  })

  const outputs = project?.lidar || []

  async function choose() {
    if (!window.fiducia?.isDesktop) {
      toast('Point clouds can only be opened in the desktop app', 'warn')
      return
    }
    const paths = await window.fiducia.dialog.openFiles({
      title: 'Open a point cloud', multiple: false,
      filters: [{ name: 'LiDAR', extensions: ['las', 'laz'] }],
    })
    if (!paths?.length) return
    setSource(paths[0])
    setSummary(null)
    const result = await call(() => api.lidar.inspect(paths[0]), { refresh: false })
    if (result) setSummary(result)
  }

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
      </section>

      {summary && (
        <>
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
                    {entry.outputPath.split(/[\\/]/).pop()}
                  </div>
                  <div className="row__meta">
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
