import { useEffect, useState } from 'react'
import { useStore, WORKFLOWS, workflowOf } from '../../state/store'
import { api } from '../../lib/api'
import InfoTip from '../InfoTip'
import ProjectionPicker, { rememberProjection } from '../ProjectionPicker'

const MATH_MODELS = [
  { id: 'aerial_film', name: 'Aerial — scanned film',
    hint: 'Fiducial marks and a calibration certificate' },
  { id: 'aerial_digital', name: 'Aerial — digital / UAV',
    hint: 'Frame sensor, no fiducials' },
  { id: 'satellite_rpc', name: 'Satellite — rational polynomial',
    hint: 'Scene RPCs, refined with control' },
]

export default function ProjectPanel() {
  const project = useStore((s) => s.project)
  const projections = useStore((s) => s.projections)
  const patchProject = useStore((s) => s.patchProject)
  const call = useStore((s) => s.call)
  const toast = useStore((s) => s.toast)

  const [snapshots, setSnapshots] = useState([])
  const [described, setDescribed] = useState(null)

  const projection = project?.projection || {}
  const mathModel = project?.mathModel || {}
  const workflow = workflowOf(project)

  useEffect(() => {
    api.project.snapshots().then((r) => setSnapshots(r.snapshots)).catch(() => {})
  }, [project?.modified])

  useEffect(() => {
    if (!projection.output) { setDescribed(null); return }
    api.projections
      .describe(projection.output)
      .then(setDescribed)
      .catch(() => setDescribed(null))
  }, [projection.output])

  if (!project) return null

  return (
    <>
      <section className="section">
        <div className="section__head"><span className="section__title">Identity</span>
          <span className="section__rule" />
          <button
            className="btn btn--ghost btn--sm"
            onClick={() => useStore.getState().setSettingsOpen(true)}
            title="Your name, certificate reading, measuring and appearance"
          >
            Settings…
          </button>
        </div>

        <div className="field">
          <label className="field__label">Name</label>
          <input
            value={project.name || ''}
            onChange={(event) => patchProject({ name: event.target.value })}
          />
        </div>

        <div className="field">
          <label className="field__label" htmlFor="project-author">
            <span>
              Author
              <InfoTip>
                Stored in the project file. Orthophotos record the name of
                whoever generates them, set under Set author name in the
                command palette.
              </InfoTip>
            </span>
          </label>
          <input
            id="project-author"
            value={project.author || ''}
            placeholder="Not recorded"
            onChange={(event) => patchProject({ author: event.target.value })}
          />
        </div>

        <div className="field">
          <label className="field__label">Description</label>
          <textarea
            rows={2}
            value={project.description || ''}
            onChange={(event) => patchProject({ description: event.target.value })}
          />
        </div>
      </section>

      <section className="section">
        <div className="section__head">
          <span className="section__title">Processing</span>
          <span className="section__rule" />
          <InfoTip>
            Decides which steps lead the rail and in what order. Changing it keeps
            everything already done; every other step stays under More tools.
          </InfoTip>
        </div>

        {WORKFLOWS.map((option) => (
          <button
            key={option.id}
            className={`option ${workflow.id === option.id ? 'option--on' : ''}`}
            onClick={() => patchProject({
              workflow: option.id,
              ...(option.mathModel ? { mathModel: { ...mathModel, kind: option.mathModel } } : {}),
            })}
          >
            <span className="option__name">{option.name}</span>
            <span className="option__hint">{option.hint}</span>
          </button>
        ))}
      </section>

      {workflow.id === 'all' && (
        <section className="section">
          <div className="section__head"><span className="section__title">Math model</span>
            <span className="section__rule" /></div>

          {MATH_MODELS.map((option) => (
            <button
              key={option.id}
              className={`option ${mathModel.kind === option.id ? 'option--on' : ''}`}
              onClick={() => patchProject({ mathModel: { ...mathModel, kind: option.id } })}
            >
              <span className="option__name">{option.name}</span>
              <span className="option__hint">{option.hint}</span>
            </button>
          ))}
        </section>
      )}

      <section className="section">
        <div className="section__head">
          <span className="section__title">Projection</span>
          <span className="section__rule" />
        </div>

        {workflow.id === 'drone' && !projection.output && (
          <div className="field__hint" style={{ marginBottom: 'var(--step-2)' }}>
            Nothing to set yet: aligning the photos picks the UTM zone their GPS is in,
            and an output pixel twice their ground resolution. Change either here after.
          </div>
        )}

        <div className="field">
          <label className="field__label">Output projection</label>
          <ProjectionPicker
            id="output-projection"
            value={projection.output || ''}
            presets={projections}
            onChange={(value) => {
              rememberProjection(value)
              patchProject({
                projection: {
                  ...projection,
                  output: value,
                  // The GCP projection is the same as the output in almost
                  // every real project, so it follows automatically rather
                  // than being a separate step you can forget.
                  gcpSource: projection.gcpSource || value,
                },
              })
            }}
          />
          {described && (
            <div className="field__hint">
              <span className="data">
                {described.epsg ? `EPSG:${described.epsg}` : 'Custom definition, no EPSG code'}
              </span>
              <InfoTip>
                <p>
                  {described.epsg
                    ? `Output files are labelled EPSG:${described.epsg}, which GIS software recognises directly.`
                    : 'This system has no official EPSG code. Fiducia defines it itself and writes the full definition into every output file, so GIS software places the data correctly, but it will not match an EPSG code by name.'}
                </p>
                <p className="data" style={{ wordBreak: 'break-all' }}>{described.definition}</p>
              </InfoTip>
              <br />
              {described.datum}, {described.unit}
              {described.notes && <><br />{described.notes}</>}
            </div>
          )}
        </div>

        <div className="field">
          <label className="field__label">
            GCP projection
            <button
              className="btn btn--ghost btn--sm"
              onClick={() => patchProject({
                projection: { ...projection, gcpSource: projection.output },
              })}
              title="Match the output projection"
            >
              match output
            </button>
          </label>
          <ProjectionPicker
            id="gcp-projection"
            value={projection.gcpSource || ''}
            presets={projections}
            onChange={(value) => patchProject({
              projection: { ...projection, gcpSource: value },
            })}
          />
        </div>

        <div className="grid-2">
          <div className="field">
            <label className="field__label">Pixel spacing X</label>
            <input
              type="number" step="0.01" className="numeric"
              value={projection.pixelSpacingX ?? 0.5}
              onChange={(event) => patchProject({
                projection: { ...projection, pixelSpacingX: Number(event.target.value) },
              })}
            />
          </div>
          <div className="field">
            <label className="field__label">Pixel spacing Y</label>
            <input
              type="number" step="0.01" className="numeric"
              value={projection.pixelSpacingY ?? 0.5}
              onChange={(event) => patchProject({
                projection: { ...projection, pixelSpacingY: Number(event.target.value) },
              })}
            />
          </div>
        </div>

        <div className="field">
          <label className="field__label">Elevation reference</label>
          <select
            value={projection.elevationReference || 'mean_sea_level'}
            onChange={(event) => patchProject({
              projection: { ...projection, elevationReference: event.target.value },
            })}
          >
            <option value="mean_sea_level">Mean sea level</option>
            <option value="ellipsoid">Ellipsoid</option>
          </select>
        </div>
      </section>

      <section className="section">
        <div className="section__head">
          <span className="section__title">
            History
            <InfoTip>
              Named restore points. Autosave runs continuously; take a snapshot
              before any change you may wish to reverse in bulk.
            </InfoTip>
          </span>
          <span className="section__rule" />
          <button
            className="btn btn--ghost btn--sm"
            onClick={() => call(() => api.project.snapshot('manual'),
              { successMessage: 'Snapshot taken' })}
          >
            Snapshot now
          </button>
        </div>

        {snapshots.length === 0 ? (
          <div className="empty-note">No snapshots yet</div>
        ) : (
          <div className="rows">
            {snapshots.slice(0, 8).map((snapshot) => (
              <div key={snapshot.path} className="row">
                <div className="row__main">
                  <div className="row__name truncate">{snapshot.name}</div>
                  <div className="row__meta">
                    {snapshot.modified.replace('T', ' ')},
                    {' '}{(snapshot.sizeBytes / 1024).toFixed(0)} KB
                  </div>
                </div>
                <div className="row__actions">
                  <button
                    className="btn btn--ghost btn--sm"
                    onClick={() => {
                      if (!window.confirm(
                        'Restore this snapshot? The current state will be saved as a ' +
                        'snapshot first.',
                      )) return
                      call(() => api.project.restore(snapshot.path),
                        { successMessage: 'Snapshot restored' })
                    }}
                  >
                    Restore
                  </button>
                </div>
              </div>
            ))}
          </div>
        )}
      </section>

      <section className="section">
        <div className="section__head">
          <span className="section__title">
            Handover
            <InfoTip>
              Packages the project with its journal and snapshots. Generated
              outputs are excluded, and image links remain valid on any machine.
            </InfoTip>
          </span>
          <span className="section__rule" /></div>
        <button
          className="btn btn--block"
          onClick={async () => {
            const desktop = window.fiducia?.isDesktop
            if (!desktop) { toast('Archiving is only available in the desktop app', 'warn'); return }
            const target = await window.fiducia.dialog.saveFile({
              title: 'Save project archive',
              defaultPath: `${project.name}.zip`,
              filters: [{ name: 'Zip archive', extensions: ['zip'] }],
            })
            if (!target) return
            call(() => api.project.archive(target, false),
              { successMessage: 'Archive written' })
          }}
        >
          Archive project…
        </button>
      </section>
    </>
  )
}
