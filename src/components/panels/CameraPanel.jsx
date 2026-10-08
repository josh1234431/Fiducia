import { useMemo, useState } from 'react'
import { useStore } from '../../state/store'
import { api } from '../../lib/api'
import CertificateReader from '../CertificateReader'
import InfoTip from '../InfoTip'

const SLOTS = [
  'top_left', 'top_middle', 'top_right',
  'left_middle', null, 'right_middle',
  'bottom_left', 'bottom_middle', 'bottom_right',
]

const SLOT_LABEL = {
  top_left: 'TL', top_middle: 'TM', top_right: 'TR',
  left_middle: 'LM', right_middle: 'RM',
  bottom_left: 'BL', bottom_middle: 'BM', bottom_right: 'BR',
}

/**
 * Interior orientation.
 *
 * The two things that make this faster than the original: principal point
 * offset accepts PPA and PPS directly and does the addition (PPO = PPA + PPS,
 * otherwise worked out by hand for every camera), and
 * the distortion table fit shows its own residual so a mistyped figure is
 * caught here rather than surfacing later as an inexplicable GCP error.
 */
export default function CameraPanel() {
  const project = useStore((s) => s.project)
  const call = useStore((s) => s.call)
  const toast = useStore((s) => s.toast)

  const camera = project?.camera || {}
  const isFilm = (project?.mathModel?.kind || 'aerial_film') === 'aerial_film'

  const [ppMode, setPpMode] = useState('ppo')
  const [ppa, setPpa] = useState({ x: '', y: '' })
  const [pps, setPps] = useState({ x: '', y: '' })
  const [tableOpen, setTableOpen] = useState(false)
  const [tableRows, setTableRows] = useState([
    { radius: '', distortion: '' }, { radius: '', distortion: '' },
    { radius: '', distortion: '' }, { radius: '', distortion: '' },
  ])
  const [tableFit, setTableFit] = useState(null)
  const [library, setLibrary] = useState(null)

  function update(patch) {
    call(() => api.camera.set({ ...camera, kind: isFilm ? 'film' : 'digital', ...patch }))
  }

  const fiducialCount = Object.keys(camera.fiducialsMm || {}).length

  const gsd = useMemo(() => {
    if (isFilm && camera.imageScale) return null
    return null
  }, [camera, isFilm])

  async function fitTable() {
    const radii = []
    const distortions = []
    tableRows.forEach((row) => {
      const r = Number(row.radius)
      const d = Number(row.distortion)
      if (Number.isFinite(r) && row.radius !== '' && row.distortion !== '') {
        radii.push(r)
        distortions.push(d)
      }
    })
    if (radii.length < 2) {
      toast('At least two radius and distortion pairs are required', 'warn')
      return
    }
    const result = await call(() => api.camera.distortionTable({
      radii, distortions, distanceUnits: 'mm', distortionUnits: 'um',
    }), { refresh: false })
    if (result) {
      setTableFit(result)
      update({ k0: result.k0, k1: result.k1, k2: result.k2, k3: result.k3 })
    }
  }

  if (!project) return null

  return (
    <>
      <CertificateReader
        onApply={(proposed) => call(() => api.camera.set({ ...camera, ...proposed }))}
      />

      <section className="section">
        <div className="section__head"><span className="section__title">Camera</span>
          <span className="section__rule" /></div>

        <div className="field">
          <label className="field__label">Name</label>
          <input
            value={camera.name || ''}
            placeholder="e.g. Wild RC30 / 15-4 UAG-S 13352"
            onChange={(event) => update({ name: event.target.value })}
          />
        </div>

        <div className="field">
          <label className="field__label">
            Focal length
            <span className="dim">mm</span>
          </label>
          <input
            type="number" step="0.001" className="numeric"
            value={camera.focalMm ?? ''}
            placeholder="153.690"
            onChange={(event) => update({ focalMm: Number(event.target.value) })}
          />
        </div>
      </section>

      <section className="section">
        <div className="section__head">
          <span className="section__title">Principal point offset</span>
          <span className="section__rule" />
        </div>

        <div className="field">
          <select value={ppMode} onChange={(event) => setPpMode(event.target.value)}>
            <option value="ppo">PPO</option>
            <option value="ppa_pps">PPA + PPS</option>
          </select>
        </div>

        {ppMode === 'ppo' ? (
          <div className="grid-2">
            <div className="field">
              <label className="field__label">PPO x <span className="dim">mm</span></label>
              <input
                type="number" step="0.000001" className="numeric"
                value={camera.ppoXMm ?? 0}
                onChange={(event) => update({ ppoXMm: Number(event.target.value) })}
              />
            </div>
            <div className="field">
              <label className="field__label">PPO y <span className="dim">mm</span></label>
              <input
                type="number" step="0.000001" className="numeric"
                value={camera.ppoYMm ?? 0}
                onChange={(event) => update({ ppoYMm: Number(event.target.value) })}
              />
            </div>
          </div>
        ) : (
          <>
            <div className="grid-2">
              <div className="field">
                <label className="field__label">PPA x</label>
                <input type="number" step="0.000001" className="numeric" value={ppa.x}
                       onChange={(e) => setPpa({ ...ppa, x: e.target.value })} />
              </div>
              <div className="field">
                <label className="field__label">PPA y</label>
                <input type="number" step="0.000001" className="numeric" value={ppa.y}
                       onChange={(e) => setPpa({ ...ppa, y: e.target.value })} />
              </div>
            </div>
            <div className="grid-2">
              <div className="field">
                <label className="field__label">PPS x</label>
                <input type="number" step="0.000001" className="numeric" value={pps.x}
                       onChange={(e) => setPps({ ...pps, x: e.target.value })} />
              </div>
              <div className="field">
                <label className="field__label">PPS y</label>
                <input type="number" step="0.000001" className="numeric" value={pps.y}
                       onChange={(e) => setPps({ ...pps, y: e.target.value })} />
              </div>
            </div>
            <button
              className="btn btn--block"
              onClick={() => {
                const x = Number(ppa.x || 0) + Number(pps.x || 0)
                const y = Number(ppa.y || 0) + Number(pps.y || 0)
                update({ ppoXMm: x, ppoYMm: y })
                toast(`PPO set to ${x.toFixed(6)}, ${y.toFixed(6)} mm`, 'good')
              }}
            >
              Set PPO from PPA + PPS
            </button>
          </>
        )}
      </section>

      <section className="section">
        <div className="section__head">
          <span className="section__title">
            Radial lens distortion
            <InfoTip>
              Enter K₀–K₃ directly, or fit them from the distortion table on the
              calibration certificate.
            </InfoTip>
          </span>
          <span className="section__rule" />
          <button className="btn btn--ghost btn--sm" onClick={() => setTableOpen((v) => !v)}>
            {tableOpen ? 'Hide table' : 'From table…'}
          </button>
        </div>

        <div className="field__hint" style={{ marginBottom: 'var(--step-2)' }}>
          Δr = K₀r + K₁r³ + K₂r⁵ + K₃r⁷
        </div>

        <div className="grid-2">
          {['k0', 'k1', 'k2', 'k3'].map((key, i) => (
            <div className="field" key={key}>
              <label className="field__label">
                K{i} <span className="dim">R{[1, 3, 5, 7][i]}</span>
              </label>
              <input
                className="numeric"
                value={camera[key] ?? 0}
                onChange={(event) => update({ [key]: Number(event.target.value) })}
              />
            </div>
          ))}
        </div>

        {tableOpen && (
          <div style={{ borderTop: '1px solid var(--rule)',
                        paddingTop: 'var(--step-3)', marginTop: 'var(--step-2)' }}>
            {tableRows.map((row, index) => (
              <div className="grid-2" key={index} style={{ marginBottom: 4 }}>
                <input
                  className="numeric" placeholder={`r${index + 1} (mm)`}
                  value={row.radius}
                  onChange={(event) => {
                    const next = [...tableRows]
                    next[index] = { ...next[index], radius: event.target.value }
                    setTableRows(next)
                  }}
                />
                <input
                  className="numeric" placeholder="Δr (µm)"
                  value={row.distortion}
                  onChange={(event) => {
                    const next = [...tableRows]
                    next[index] = { ...next[index], distortion: event.target.value }
                    setTableRows(next)
                  }}
                />
              </div>
            ))}

            <div style={{ display: 'flex', gap: 6, marginTop: 6 }}>
              <button
                className="btn btn--sm"
                onClick={() => setTableRows([...tableRows, { radius: '', distortion: '' }])}
              >
                Add row
              </button>
              <button className="btn btn--primary btn--sm" onClick={fitTable}>
                Fit K₀–K₃
              </button>
            </div>

            {tableFit && (
              <div className="measures" style={{ marginTop: 'var(--step-3)' }}>
                <div className="measure">
                  <div className="measure__name">Fit RMS</div>
                  <div className={`measure__value ${tableFit.rmsUm < 1 ? 'measure__value--good'
                    : tableFit.rmsUm < 3 ? 'measure__value--warn' : 'measure__value--bad'}`}>
                    {tableFit.rmsUm.toFixed(3)} µm
                  </div>
                </div>
                <div className="measure">
                  <div className="measure__name">Worst</div>
                  <div className="measure__value">{tableFit.maxUm.toFixed(3)} µm</div>
                </div>
              </div>
            )}
          </div>
        )}
      </section>

      {isFilm && (
        <>
          <section className="section">
            <div className="section__head">
              <span className="section__title">
                Calibrated fiducials
                <InfoTip>Positions in millimetres, from the calibration certificate.</InfoTip>
              </span>
          <span className="section__rule" />
              <span className={`chip ${fiducialCount >= 4 ? 'chip--good' : 'chip--warn'}`}>
                {fiducialCount} of 8
              </span>
            </div>

            <div className="field">
              <label className="field__label">Position</label>
              <select
                value={camera.fiducialPosition || 'edge_and_corner'}
                onChange={(event) => update({ fiducialPosition: event.target.value })}
              >
                <option value="edge_and_corner">Edge and corner</option>
                <option value="corner">Corner only</option>
                <option value="edge">Edge only</option>
              </select>
            </div>

            <div className="field">
              <label className="field__label">
                <span>
                  Transformation
                  <InfoTip>
                    How scan pixels are mapped to film millimetres from the measured
                    fiducials. Affine (six parameters) also absorbs a scanner's differing
                    scale in x and y and slight skew, and needs four marks; conformal
                    (four) keeps the shape and suits a square, orthogonal scanner.
                    Automatic uses affine with four or more marks.
                  </InfoTip>
                </span>
              </label>
              <select
                value={camera.fiducialTransform || 'auto'}
                onChange={(event) => update({ fiducialTransform: event.target.value })}
              >
                <option value="auto">Automatic</option>
                <option value="affine">Affine</option>
                <option value="conformal">Conformal</option>
              </select>
            </div>

            {SLOTS.filter(Boolean).map((slot) => {
              const value = (camera.fiducialsMm || {})[slot] || ['', '']
              return (
                <div className="grid-3" key={slot} style={{ marginBottom: 4,
                     alignItems: 'center' }}>
                  <span style={{ fontSize: 11, color: 'var(--ink-soft)' }}>
                    {slot.replace(/_/g, ' ')}
                  </span>
                  <input
                    className="numeric" placeholder="x" value={value[0]}
                    onChange={(event) => update({
                      fiducialsMm: {
                        ...(camera.fiducialsMm || {}),
                        [slot]: [Number(event.target.value), Number(value[1]) || 0],
                      },
                    })}
                  />
                  <input
                    className="numeric" placeholder="y" value={value[1]}
                    onChange={(event) => update({
                      fiducialsMm: {
                        ...(camera.fiducialsMm || {}),
                        [slot]: [Number(value[0]) || 0, Number(event.target.value)],
                      },
                    })}
                  />
                </div>
              )
            })}
          </section>

          <section className="section">
            <div className="section__head"><span className="section__title">Photo scale</span>
          <span className="section__rule" /></div>
            <div className="field">
              <label className="field__label">
                <span>
                  Image scale 1 :
                  <InfoTip>Used to estimate the initial flying height for the adjustment.</InfoTip>
                </span>
              </label>
              <input
                type="number" className="numeric"
                value={camera.imageScale ?? ''}
                placeholder="20000"
                onChange={(event) => update({ imageScale: Number(event.target.value) })}
              />
            </div>
          </section>
        </>
      )}

      {!isFilm && (
        <section className="section">
          <div className="section__head"><span className="section__title">Sensor geometry</span>
          <span className="section__rule" /></div>

          <div className="field">
            <label className="field__label">Pixel pitch <span className="dim">mm</span></label>
            <input
              type="number" step="0.0001" className="numeric"
              value={camera.pixelPitchMm ?? ''}
              placeholder="0.0060"
              onChange={(event) => update({ pixelPitchMm: Number(event.target.value) })}
            />
          </div>

          <div className="grid-2">
            <div className="field">
              <label className="field__label">Columns</label>
              <input
                type="number" className="numeric" value={camera.columns ?? ''}
                onChange={(event) => update({ columns: Number(event.target.value) })}
              />
            </div>
            <div className="field">
              <label className="field__label">Rows</label>
              <input
                type="number" className="numeric" value={camera.rows ?? ''}
                onChange={(event) => update({ rows: Number(event.target.value) })}
              />
            </div>
          </div>

          <div className="grid-2">
            <div className="field">
              <label className="field__label">
                <span>
                  Long-track
                  <InfoTip>Along the flight direction, not the long side of the image.</InfoTip>
                </span>
                <span className="dim">mm</span>
              </label>
              <input
                type="number" step="0.000001" className="numeric"
                value={camera.longTrackOffsetMm ?? 0}
                onChange={(event) => update({ longTrackOffsetMm: Number(event.target.value) })}
              />
            </div>
            <div className="field">
              <label className="field__label">Cross-track <span className="dim">mm</span></label>
              <input
                type="number" step="0.000001" className="numeric"
                value={camera.crossTrackOffsetMm ?? 0}
                onChange={(event) => update({ crossTrackOffsetMm: Number(event.target.value) })}
              />
            </div>
          </div>
        </section>
      )}

      <section className="section">
        <div className="section__head">
          <span className="section__title">
            Camera library
            <InfoTip>Stores a calibrated camera for reuse in later projects.</InfoTip>
          </span>
          <span className="section__rule" />
        </div>
        <div className="grid-2">
          <button
            className="btn"
            onClick={async () => {
              const target = await window.fiducia?.dialog.saveFile({
                title: 'Save to a camera library', defaultPath: 'cameras.json',
                filters: [{ name: 'Camera library', extensions: ['json'] }],
              })
              if (!target) return
              call(() => api.exchange.saveCamera(target, camera.name || 'Camera'), {
                successMessage: 'Camera saved to the library', refresh: false,
              })
            }}
            disabled={!camera.focalMm}
          >
            Save to library…
          </button>
          <button
            className="btn"
            onClick={async () => {
              const paths = await window.fiducia?.dialog.openFiles({
                title: 'Open a camera library', multiple: false,
                filters: [{ name: 'Camera library', extensions: ['json'] }],
              })
              if (!paths?.length) return
              const library = await call(() => api.exchange.loadCamera(paths[0]),
                { refresh: false })
              if (!library?.cameras?.length) return
              setLibrary({ path: paths[0], cameras: library.cameras })
            }}
          >
            Load from library…
          </button>
        </div>

        {library && (
          <div className="rows" style={{ marginTop: 'var(--step-2)' }}>
            {library.cameras.map((entry) => (
              <button
                key={entry.label}
                className="row"
                onClick={() => {
                  call(() => api.exchange.loadCamera(library.path, entry.label), {
                    successMessage: entry.label + ' loaded',
                  })
                  setLibrary(null)
                }}
              >
                <div className="row__main">
                  <div className="row__name">{entry.label}</div>
                  <div className="row__meta">
                    {entry.camera.kind === 'film' ? 'Scanned film' : 'Digital'}
                    {entry.camera.focalMm ? ', ' + entry.camera.focalMm + ' mm' : ''}
                  </div>
                </div>
              </button>
            ))}
          </div>
        )}

      </section>

      <section className="section">
        <div className="section__head">
          <span className="section__title">
            Corrections
            <InfoTip>
              <p>At the frame corners, earth curvature displaces image points by
              approximately 13 µm at 1,500 m and 36 µm at 4,000 m. Refraction is
              approximately one third of that.</p>
              <p>Applied in the adjustment and orthorectification. Not yet applied
              to stereo DEM extraction or automatic control.</p>
            </InfoTip>
          </span>
          <span className="section__rule" /></div>
        <label className="field field--row">
          <span className="field__label">Atmospheric refraction</span>
          <input
            type="checkbox" checked={!!camera.applyAtmospheric}
            onChange={(event) => update({ applyAtmospheric: event.target.checked })}
          />
        </label>
        <label className="field field--row">
          <span className="field__label">Earth curvature</span>
          <input
            type="checkbox" checked={!!camera.applyEarthCurvature}
            onChange={(event) => update({ applyEarthCurvature: event.target.checked })}
          />
        </label>
      </section>
    </>
  )
}
