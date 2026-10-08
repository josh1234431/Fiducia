import { useEffect, useMemo, useState } from 'react'
import { useStore } from '../../state/store'
import { api } from '../../lib/api'
import InfoTip from '../InfoTip'
import OutputRows from '../OutputRows'
import { useReveal } from '../../lib/reveal'

/**
 * Terrain — everything that happens to an elevation model after extraction.
 *
 * Stereo hands back a surface, not a product: holes over water, spikes on
 * roofs, one DEM per pair rather than one for the block. This is the stage
 * between that and something a client can use, and leaving it to a GIS package
 * is how a photogrammetric workflow ends up half-finished in someone else's
 * software.
 */
export default function TerrainPanel() {
  const project = useStore((s) => s.project)
  const call = useStore((s) => s.call)
  const jobs = useStore((s) => s.jobs)
  const outputView = useStore((s) => s.outputView)
  const showOutput = useStore((s) => s.showOutput)
  const popOutOutput = useStore((s) => s.popOutOutput)
  const generatedRef = useReveal('terrain')

  const [selected, setSelected] = useState(null)
  const [stats, setStats] = useState(null)
  const [volume, setVolume] = useState(null)
  const [contourOptions, setContourOptions] = useState({ interval: 5, indexEvery: 5 })
  const [shadeOptions, setShadeOptions] = useState({ azimuth: 315, altitude: 45, zFactor: 1 })
  const [mergeChosen, setMergeChosen] = useState([])

  // Every elevation surface the project knows about, from any source.
  const surfaces = useMemo(() => {
    const out = []
    const seen = new Set()
    const add = (path, role) => {
      if (!path || seen.has(path)) return
      seen.add(path)
      out.push({ path, role, name: path.split(/[\\/]/).pop() })
    }
    add(project?.dem?.referencePath, 'reference')
    ;(project?.stereoDems || []).forEach((d) => add(d.path, 'stereo'))
    ;(project?.lidar || []).forEach((d) => add(d.outputPath, 'lidar'))
    ;(project?.surfaces || []).forEach((d) => add(d.path, d.role))
    return out
  }, [project?.dem, project?.stereoDems, project?.lidar, project?.surfaces])

  useEffect(() => {
    if (!selected && surfaces.length) setSelected(surfaces[0].path)
  }, [surfaces, selected])

  useEffect(() => {
    if (!selected) { setStats(null); return }
    api.terrain.statistics(selected).then(setStats).catch(() => setStats(null))
  }, [selected, project?.modified])

  const busy = jobs.some(
    (j) => j.kind === 'terrain' && (j.status === 'running' || j.status === 'queued'),
  )

  function run(fn, label) {
    return call(fn, { successMessage: label })
  }

  if (!surfaces.length) {
    return (
      <div className="empty-note">
        No elevation models. Set a reference DEM, extract one from stereo, or
        rasterise a LiDAR point cloud.
      </div>
    )
  }

  return (
    <>
      <section className="section">
        <div className="section__head">
          <span className="section__title">Surface</span>
          <span className="section__rule" />
        </div>

        <div className="field">
          <select value={selected || ''} onChange={(event) => setSelected(event.target.value)}>
            {surfaces.map((surface) => (
              <option key={surface.path} value={surface.path}>
                {surface.name} — {surface.role}
              </option>
            ))}
          </select>
        </div>

        {stats && (
          <>
            <div className="measures">
              <div className="measure">
                <div className="measure__value" style={{ fontSize: 14 }}>
                  {stats.min.toFixed(1)}–{stats.max.toFixed(1)}
                </div>
                <div className="measure__name">Elevation range, m</div>
              </div>
              <div className="measure">
                <div className={`measure__value ${
                  stats.coverage > 0.95 ? 'measure__value--good'
                    : stats.coverage > 0.8 ? 'measure__value--warn' : 'measure__value--bad'}`}
                     style={{ fontSize: 14 }}>
                  {(stats.coverage * 100).toFixed(1)}%
                </div>
                <div className="measure__name">Coverage</div>
              </div>
              <div className="measure">
                <div className="measure__value" style={{ fontSize: 14 }}>
                  {stats.cellSize.toFixed(2)} m
                </div>
                <div className="measure__name">Cell size</div>
              </div>
              <div className="measure">
                <div className="measure__value" style={{ fontSize: 14 }}>
                  {stats.voidCells.toLocaleString()}
                </div>
                <div className="measure__name">Void cells</div>
              </div>
            </div>

            <button
              className="btn btn--sm btn--block"
              style={{ marginTop: 'var(--step-2)' }}
              onClick={() => showOutput({
                path: selected,
                label: surfaces.find((s) => s.path === selected)?.name,
                step: 'terrain',
              })}
            >
              Show on the canvas
            </button>
          </>
        )}
      </section>

      <section className="section">
        <div className="section__head">
          <span className="section__title">
            Repair
            <InfoTip>
              <p>Void filling is limited to gaps adjacent to measured ground.
              Large voids, such as open water, are left unfilled.</p>
              <p>Median smoothing removes spikes while preserving breaklines.</p>
            </InfoTip>
          </span>
          <span className="section__rule" />
        </div>

        <div className="grid-2">
          <button
            className="btn" disabled={busy || !selected}
            onClick={() => run(() => api.terrain.fill({ path: selected }), 'Filling voids')}
          >
            Fill voids
          </button>
          <button
            className="btn" disabled={busy || !selected}
            onClick={() => run(() => api.terrain.smooth({ path: selected, method: 'median', size: 5 }),
              'Smoothing')}
          >
            Smooth
          </button>
        </div>

        {surfaces.length > 1 && (
          <>
            <div className="field" style={{ marginTop: 'var(--step-3)' }}>
              <label className="field__label">Merge surfaces</label>
              <div className="rows">
                {surfaces.map((surface) => (
                  <label key={surface.path} className="row" style={{ cursor: 'pointer' }}>
                    <input
                      type="checkbox" style={{ width: 13, flex: 'none' }}
                      checked={mergeChosen.includes(surface.path)}
                      onChange={() => setMergeChosen((current) =>
                        current.includes(surface.path)
                          ? current.filter((p) => p !== surface.path)
                          : [...current, surface.path])}
                    />
                    <div className="row__main">
                      <div className="row__name truncate">{surface.name}</div>
                    </div>
                  </label>
                ))}
              </div>
            </div>
            <button
              className="btn btn--block" disabled={busy || mergeChosen.length < 2}
              onClick={() => run(() => api.terrain.merge({ inputs: mergeChosen, method: 'feather' }),
                'Merging surfaces')}
            >
              Merge {mergeChosen.length || ''} into one surface
            </button>
          </>
        )}
      </section>

      <section className="section">
        <div className="section__head">
          <span className="section__title">
            Shaded relief
            <InfoTip>
              A low sun angle reveals matching blunders that a colour ramp
              conceals.
            </InfoTip>
          </span>
          <span className="section__rule" />
        </div>

        <div className="grid-3">
          <div className="field">
            <label className="field__label">Azimuth</label>
            <input type="number" className="numeric" value={shadeOptions.azimuth}
                   onChange={(e) => setShadeOptions({ ...shadeOptions, azimuth: Number(e.target.value) })} />
          </div>
          <div className="field">
            <label className="field__label">Altitude</label>
            <input type="number" className="numeric" value={shadeOptions.altitude}
                   onChange={(e) => setShadeOptions({ ...shadeOptions, altitude: Number(e.target.value) })} />
          </div>
          <div className="field">
            <label className="field__label">Z factor</label>
            <input type="number" step="0.5" className="numeric" value={shadeOptions.zFactor}
                   onChange={(e) => setShadeOptions({ ...shadeOptions, zFactor: Number(e.target.value) })} />
          </div>
        </div>

        <button
          className="btn btn--primary btn--block" disabled={busy || !selected}
          onClick={() => run(() => api.terrain.hillshade({ path: selected, ...shadeOptions }),
            'Hillshade written')}
        >
          Generate hillshade
        </button>
      </section>

      <section className="section">
        <div className="section__head">
          <span className="section__title">Contours</span>
          <span className="section__rule" />
        </div>

        <div className="grid-2">
          <div className="field">
            <label className="field__label">Interval, m</label>
            <input type="number" step="0.5" className="numeric" value={contourOptions.interval}
                   onChange={(e) => setContourOptions({ ...contourOptions, interval: Number(e.target.value) })} />
          </div>
          <div className="field">
            <label className="field__label">Index every</label>
            <input type="number" className="numeric" value={contourOptions.indexEvery}
                   onChange={(e) => setContourOptions({ ...contourOptions, indexEvery: Number(e.target.value) })} />
          </div>
        </div>

        {stats && (
          <div className="field__hint" style={{ marginBottom: 6 }}>
            Approximately {Math.max(0, Math.floor((stats.max - stats.min) / Math.max(contourOptions.interval, 0.01)))}
            {' '}contours over a {(stats.max - stats.min).toFixed(0)} m range.
          </div>
        )}

        <button
          className="btn btn--primary btn--block" disabled={busy || !selected}
          onClick={() => run(() => api.terrain.contours({ path: selected, ...contourOptions }),
            'Contours written')}
        >
          Generate contours
        </button>
      </section>

      <section className="section">
        <div className="section__head">
          <span className="section__title">
            Bare earth
            <InfoTip>
              Removes vegetation and structures from a surface model using a
              progressive morphological filter. Verify the result against a
              hillshade; flat roofs and terraces can be misclassified.
            </InfoTip>
          </span>
          <span className="section__rule" />
        </div>
        <button
          className="btn btn--block" disabled={busy || !selected}
          onClick={() => run(() => api.terrain.bareEarth({ path: selected }),
            'Filtered to bare earth')}
        >
          Filter to terrain model
        </button>
      </section>

      <section className="section">
        <div className="section__head">
          <span className="section__title">Volume</span>
          <span className="section__rule" />
        </div>

        {surfaces.length < 2 ? (
          <div className="field__hint">
            Two surfaces are required: a measured surface and a base surface.
          </div>
        ) : (
          <>
            <div className="field">
              <label className="field__label">Against</label>
              <select
                onChange={async (event) => {
                  if (!event.target.value) { setVolume(null); return }
                  const result = await call(
                    () => api.terrain.volume(selected, event.target.value),
                    { refresh: false },
                  )
                  setVolume(result)
                }}
                defaultValue=""
              >
                <option value="">Choose a base surface…</option>
                {surfaces.filter((s) => s.path !== selected).map((surface) => (
                  <option key={surface.path} value={surface.path}>{surface.name}</option>
                ))}
              </select>
            </div>

            {volume && (
              <div className="measures">
                <div className="measure">
                  <div className="measure__value" style={{ fontSize: 14 }}>
                    {Math.round(volume.cutVolume).toLocaleString()}
                  </div>
                  <div className="measure__name">Cut, m³</div>
                </div>
                <div className="measure">
                  <div className="measure__value" style={{ fontSize: 14 }}>
                    {Math.round(volume.fillVolume).toLocaleString()}
                  </div>
                  <div className="measure__name">Fill, m³</div>
                </div>
                <div className="measure">
                  <div className={`measure__value ${
                    volume.netVolume >= 0 ? 'measure__value--good' : 'measure__value--warn'}`}
                       style={{ fontSize: 14 }}>
                    {Math.round(volume.netVolume).toLocaleString()}
                  </div>
                  <div className="measure__name">Net, m³</div>
                </div>
                <div className="measure">
                  <div className="measure__value" style={{ fontSize: 14 }}>
                    {Math.round(volume.comparedArea).toLocaleString()}
                  </div>
                  <div className="measure__name">Area, m²</div>
                </div>
              </div>
            )}
          </>
        )}
      </section>

      {(project?.terrainProducts || []).length > 0 && (
        <section className="section" ref={generatedRef}>
          <div className="section__head"><span className="section__title">Generated</span>
          <span className="section__rule" /></div>
          <OutputRows
            items={project.terrainProducts.slice().reverse().map((product) => ({
              key: product.path,
              path: product.path,
              name: product.path.split(/[\\/]/).pop(),
              meta: `${product.kind === 'hillshade' ? 'Shaded relief' : 'Contours'} of `
                + `${(product.source || '').split(/[\\/]/).pop()}, ${product.createdAt}`,
            }))}
            step="terrain"
            current={outputView?.step === 'terrain' ? outputView.path : null}
            onView={showOutput}
            onPopOut={popOutOutput}
          />
        </section>
      )}
    </>
  )
}
