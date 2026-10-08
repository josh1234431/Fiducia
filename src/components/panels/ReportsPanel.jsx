import { useState } from 'react'
import { useStore } from '../../state/store'
import { api } from '../../lib/api'
import InfoTip from '../InfoTip'

const SECTIONS = [
  ['projectInfo', 'Project and math model'],
  ['cameraInfo', 'Camera calibration'],
  ['projection', 'Projection'],
  ['imageInfo', 'Images'],
  ['fiducials', 'Fiducial measurements'],
  ['exteriorOrientation', 'Exterior orientation'],
  ['gcps', 'Ground control points'],
  ['tiePoints', 'Tie points'],
  ['modelStatistics', 'Model statistics'],
  ['outputs', 'Outputs'],
]

/**
 * Reports.
 *
 * Everything is ticked by default. An incomplete export is easy to miss, and
 * the legacy default of unticked is a trap, not a preference, so it is
 * inverted here.
 */
export default function ReportsPanel() {
  const project = useStore((s) => s.project)
  const call = useStore((s) => s.call)
  const toast = useStore((s) => s.toast)

  const [include, setInclude] = useState(
    Object.fromEntries(SECTIONS.map(([key]) => [key, true])),
  )
  const [imageIds, setImageIds] = useState(null)   // null means every image
  const [units, setUnits] = useState('ground')
  const [show, setShow] = useState('all')
  const [text, setText] = useState('')
  // Classic reproduces the long-standing text layout, column for column, so
  // marking templates and parsing scripts built around it keep working.
  const [layout, setLayoutState] = useState(() => {
    try { return localStorage.getItem('fiducia.reportLayout') || 'standard' } catch { return 'standard' }
  })
  const classic = layout === 'classic'
  function setLayout(next) {
    setLayoutState(next)
    setText('')
    try { localStorage.setItem('fiducia.reportLayout', next) } catch { /* preference only */ }
  }
  const [which, setWhich] = useState(null)

  const images = project?.images || []
  const selectedImages = imageIds ?? images.map((i) => i.id)

  async function preview(kind) {
    const result = kind === 'project'
      ? await call(() => api.reports.project({ include, imageIds: selectedImages, layout }),
          { refresh: false })
      : await call(() => api.reports.residual({ units, show, layout }), { refresh: false })
    if (result != null) {
      setText(typeof result === 'string' ? result : JSON.stringify(result, null, 2))
      setWhich(kind)
    }
  }

  async function save(kind) {
    if (!window.fiducia?.isDesktop) {
      toast('Reports can only be saved in the desktop app', 'warn')
      return
    }
    const name = project?.name?.replace(/[^\w\-.]/g, '_') || 'project'
    const target = await window.fiducia.dialog.saveFile({
      title: `Save ${kind} report`,
      defaultPath: `${name}_${kind}.txt`,
      filters: [{ name: 'Text', extensions: ['txt'] }],
    })
    if (!target) return

    await call(() => kind === 'project'
      ? api.reports.project({ include, imageIds: selectedImages, savePath: target, layout })
      : api.reports.residual({ units, show, savePath: target, layout }),
      { successMessage: `${kind} report saved`, refresh: false })
  }

  function toggleImage(id) {
    const current = selectedImages
    setImageIds(current.includes(id)
      ? current.filter((x) => x !== id)
      : [...current, id])
  }

  if (!project) return null

  return (
    <>
      <section className="section">
        <div className="section__head">
          <span className="section__title">Layout</span>
          <span className="section__rule" />
        </div>
        <button
          className={`option ${!classic ? 'option--on' : ''}`}
          onClick={() => setLayout('standard')}
        >
          <span className="option__name">Standard</span>
          <span className="option__hint">
            Sectioned, with model statistics and blunder detection.
          </span>
        </button>
        <button
          className={`option ${classic ? 'option--on' : ''}`}
          onClick={() => setLayout('classic')}
        >
          <span className="option__name">Classic text layout</span>
          <span className="option__hint">
            The established fixed-column format, compatible with existing templates
            and parsing scripts.
          </span>
        </button>
      </section>

      <section className="section">
        <div className="section__head">
          <span className="section__title">Project report</span>
          <span className="section__rule" />
        </div>

        {classic && (
          <div className="field__hint" style={{ marginBottom: 'var(--step-2)' }}>
            The classic layout includes every section.
          </div>
        )}

        {!classic && (
          <details className="advanced">
            <summary>Sections</summary>
            <div className="advanced__body">
              {SECTIONS.map(([key, label]) => (
                <label className="field field--row" key={key} style={{ marginBottom: 4 }}>
                  <span className="field__label">{label}</span>
                  <input
                    type="checkbox" checked={include[key]}
                    onChange={(event) => setInclude({ ...include, [key]: event.target.checked })}
                  />
                </label>
              ))}
            </div>
          </details>
        )}

        <div className="section__head" style={{ marginTop: 'var(--step-3)' }}>
          <span className="section__title">Images included</span>
          <span className="section__rule" />
          <button
            className="btn btn--ghost btn--sm"
            onClick={() => setImageIds(
              selectedImages.length === images.length ? [] : images.map((i) => i.id))}
          >
            {selectedImages.length === images.length ? 'None' : 'All'}
          </button>
        </div>

        {images.length === 0 ? (
          <div className="empty-note">No images in the project.</div>
        ) : (
          <div className="rows">
            {images.map((image) => (
              <label
                key={image.id}
                className={`row ${selectedImages.includes(image.id) ? 'row--active' : ''}`}
                style={{ cursor: 'pointer' }}
              >
                <input
                  type="checkbox"
                  checked={selectedImages.includes(image.id)}
                  onChange={() => toggleImage(image.id)}
                  style={{ width: 14, flex: 'none' }}
                />
                <div className="row__main">
                  <div className="row__name truncate">{image.name}</div>
                </div>
              </label>
            ))}
          </div>
        )}

        <div style={{ display: 'flex', gap: 6, marginTop: 'var(--step-3)' }}>
          <button className="btn btn--sm" onClick={() => preview('project')}>Preview</button>
          <button className="btn btn--primary btn--sm" onClick={() => save('project')}>
            Save as .txt…
          </button>
        </div>
      </section>

      <section className="section">
        <div className="section__head">
          <span className="section__title">
            Residual report
            {classic && (
              <InfoTip>
                DS columns are standardised residuals. Values above approximately
                3 warrant inspection.
              </InfoTip>
            )}
          </span>
          <span className="section__rule" />
        </div>

        {classic ? (
          <div className="field__hint" style={{ marginBottom: 'var(--step-2)' }}>
            Ground units, all points, largest residual first.
          </div>
        ) : (
        <div className="grid-2">
          <div className="field">
            <label className="field__label">Units</label>
            <select value={units} onChange={(event) => setUnits(event.target.value)}>
              <option value="ground">Ground units</option>
              <option value="pixels">Pixels</option>
            </select>
          </div>
          <div className="field">
            <label className="field__label">Show</label>
            <select value={show} onChange={(event) => setShow(event.target.value)}>
              <option value="all">All points</option>
              <option value="gcp">Control</option>
              <option value="check">Check</option>
              <option value="tie">Tie</option>
            </select>
          </div>
        </div>
        )}

        {!project.model && (
          <div className="field__hint" style={{ color: 'var(--signal-warn)' }}>
            A solved sensor model is required.
          </div>
        )}

        <div style={{ display: 'flex', gap: 6 }}>
          <button
            className="btn btn--sm" disabled={!project.model}
            onClick={() => preview('residual')}
          >
            Preview
          </button>
          <button
            className="btn btn--primary btn--sm" disabled={!project.model}
            onClick={() => save('residual')}
          >
            Save as .txt…
          </button>
        </div>
      </section>

      {text && (
        <section className="section">
          <div className="section__head">
            <span className="section__title">
              {which === 'project' ? 'Project report' : 'Residual report'} preview
            </span>
          <span className="section__rule" />
            <button className="btn btn--ghost btn--sm" onClick={() => setText('')}>✕</button>
          </div>
          <div className="report-view">{text}</div>
          <button
            className="btn btn--sm btn--block"
            style={{ marginTop: 6 }}
            onClick={() => {
              navigator.clipboard.writeText(text)
              toast('Report copied', 'good')
            }}
          >
            Copy to clipboard
          </button>
        </section>
      )}
    </>
  )
}
