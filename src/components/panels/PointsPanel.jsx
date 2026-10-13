import { useEffect, useState } from 'react'
import { useStore, selectActiveImage } from '../../state/store'
import { api } from '../../lib/api'
import { axisLabels, isLoEastingNorthing } from '../../lib/axes'
import InfoTip from '../InfoTip'
import { accuracyTone, fitRule, fitToneM, fitTonePx } from '../../lib/quality'

/**
 * Ground control and tie points.
 *
 * The measurement flow is one gesture: arm a pick, click the feature on the
 * photo, type or extract the ground coordinate. On a legacy workstation the same action
 * spans a viewer window, a collection dialog, a reference-image window and a
 * separate DEM browse — four windows that all have to be open at once.
 */

/**
 * Split pasted text into coordinate values, normalised to a decimal point.
 *
 * Commas are ambiguous: "-3742938,908851 -43169,619897" is two values with
 * decimal commas, while "-43169.62,-3742938.91" is two values separated by a
 * comma. A comma is read as a decimal mark when the values are already
 * separated by spaces, tabs or semicolons, or when it is the only comma in a
 * single run of digits; otherwise it separates values.
 */
export function splitCoordinates(text) {
  const tokens = String(text ?? '').trim().split(/[\s;]+/).filter(Boolean)
  const decimalComma = (token) => /^[+-]?\d+,\d+$/.test(token)
  if (tokens.length && tokens.every(decimalComma)) {
    return tokens.map((token) => token.replace(',', '.'))
  }
  return tokens.flatMap((token) => token.split(',')).filter(Boolean)
}

export default function PointsPanel() {
  const project = useStore((s) => s.project)
  const image = useStore(selectActiveImage)
  const armPick = useStore((s) => s.armPick)
  const call = useStore((s) => s.call)
  const toast = useStore((s) => s.toast)
  const setSelectedPoint = useStore((s) => s.setSelectedPoint)
  const selectedPointId = useStore((s) => s.selectedPointId)

  const [draft, setDraft] = useState(null)
  const [crs, setCrs] = useState(null)
  const [swapHint, setSwapHint] = useState(null)   // { z } when the pair looks reversed
  const [preview, setPreview] = useState(null)
  const [autoJobId, setAutoJobId] = useState(null)
  const [proposals, setProposals] = useState(null)
  const [rejected, setRejected] = useState(new Set())
  const [autoOptions, setAutoOptions] = useState({
    targetCount: 20, searchM: 90, minScore: 0.55,
  })
  const jobs = useStore((s) => s.jobs)
  const autoJob = jobs.find((j) => j.id === autoJobId)

  useEffect(() => {
    if (autoJob?.status === 'done' && autoJob.result) {
      setProposals(autoJob.result.results)
      setRejected(new Set())
      setAutoJobId(null)
    } else if (autoJob?.status === 'failed') {
      setAutoJobId(null)
    }
  }, [autoJob?.status])

  // What the two horizontal coordinates are called here. Taken from the
  // projection control is collected in, because on the Lo zones they are a
  // westing and a southing, not an easting and a northing.
  const crsId = project?.projection?.gcpSource || project?.projection?.output || ''
  useEffect(() => {
    if (!crsId) { setCrs(null); return }
    let current = true
    api.projections.describe(crsId)
      .then((described) => { if (current) setCrs(described) })
      .catch(() => { if (current) setCrs(null) })
    return () => { current = false }
  }, [crsId])

  const loAsYX = useStore((s) => s.loAsYX)
  const setLoAsYX = useStore((s) => s.setLoAsYX)
  const axes = axisLabels(crs, { loAsYX })
  // Shown and typed in the axes' own sense; stored as the projection's.
  const shown = (value) => (value === '' || value == null ? value : Number(value) * axes.sign)

  // Live residuals for the photo on the canvas. Recomputed whenever control
  // on it changes; nothing is saved until the model is computed properly.
  const controlKey = (project?.gcps || [])
    .map((g) => `${g.id}:${g.x}:${g.y}:${g.z}:${!!g.isCheckPoint}`).join('|')
    + '#' + (project?.observations || []).filter((o) => o.imageId === image?.id)
      .map((o) => `${o.pointId}:${o.col}:${o.row}`).join('|')
    + '#' + JSON.stringify(project?.camera || {})
  useEffect(() => {
    if (!image?.id) { setPreview(null); return undefined }
    let current = true
    const timer = setTimeout(() => {
      api.model.preview(image.id)
        .then((result) => { if (current) setPreview(result) })
        .catch(() => { if (current) setPreview(null) })
    }, 250)
    return () => { current = false; clearTimeout(timer) }
  }, [image?.id, controlKey])

  const residualOf = Object.fromEntries((preview?.points || []).map((p) => [p.pointId, p]))

  /**
   * Read a typed or pasted coordinate. A pair or triple pasted into one box
   * (as ArcGIS copies them) is split across the boxes instead of becoming a
   * value nobody can compute with.
   */
  function enterCoordinate(axis, text) {
    const parts = splitCoordinates(text)
    const numeric = parts.length > 1 && parts.every((part) => Number.isFinite(Number(part)))
    setSwapHint(null)
    if (numeric && axis !== 'z') {
      const [first, second, third] = parts
      setDraft({ ...draft, x: first, y: second, ...(third !== undefined ? { z: third } : {}) })
      toast(`Pasted values split as ${axes.first} ${first}, ${axes.second} ${second}`
        + (third !== undefined ? `, ${axes.elevation} ${third}` : '')
        + '. Verify the order.', 'accent', 7000)
      return
    }
    setDraft({ ...draft, [axis]: text })
  }

  function numberOf(text) {
    const trimmed = String(text ?? '').trim().replace(/^(-?\d+),(\d+)$/, '$1.$2')
    if (!trimmed) return null
    const value = Number(trimmed)
    return Number.isFinite(value) ? value : null
  }

  // Without a DEM to test against, magnitude is still telling: an easting or
  // westing is within a few hundred kilometres of its meridian, while a
  // northing or southing in southern Africa is in the millions.
  function looksReversed(x, y) {
    if (crs?.isGeographic) return Math.abs(x) <= 90 && Math.abs(y) > 90
    return Math.abs(x) >= 1000000 && Math.abs(y) < 1000000
  }

  function swapDraft() {
    setDraft({ ...draft, x: draft.y, y: draft.x,
               ...(swapHint?.z != null ? { z: swapHint.z.toFixed(3) } : {}) })
    setSwapHint(null)
  }

  async function chooseReference() {
    const paths = await window.fiducia?.dialog.openFiles({
      title: 'Choose the geocoded reference (select every tile)', multiple: true,
      filters: [{ name: 'Imagery', extensions: ['tif', 'tiff', 'pix', 'img', 'jpg', 'vrt'] }],
    })
    if (!paths?.length) return
    await call(() => api.auto.setReferenceImage(paths), {
      successMessage: paths.length > 1 ? `${paths.length} reference tiles joined` : 'Reference image set',
    })
  }

  async function findControl() {
    if (!dem.referenceImage) {
      toast('A geocoded reference image is required', 'warn')
      return
    }
    setProposals(null)
    const response = await call(() => api.auto.gcps({
      referencePath: dem.referenceImage,
      demPath: dem.referencePath,
      ...autoOptions,
    }), { refresh: false })
    if (response?.job) setAutoJobId(response.job.id)
  }

  // Automatic points already on the photos being proposed for: after the
  // model is solved a second pass finds the same features more precisely.
  const proposalImages = new Set((proposals || []).map((entry) => entry.imageId))
  const earlierAutomatic = new Set(
    (project?.observations || [])
      .filter((o) => proposalImages.has(o.imageId))
      .map((o) => o.pointId)
      .filter((id) => (project?.gcps || []).some((g) => g.id === id && g.source === 'automatic')),
  ).size
  const [replaceEarlier, setReplaceEarlier] = useState(true)

  const accepted = (proposals || []).flatMap((entry) =>
    (entry.proposals || [])
      .map((proposal, index) => ({
        ...proposal, imageId: entry.imageId, key: entry.imageId + ':' + index,
      }))
      .filter((proposal) => !rejected.has(proposal.key)),
  )
  const [tieOptions, setTieOptions] = useState({
    method: 'ncc', targetCount: 60, minScore: 0.75, refine: 'lsm', replaceAutomatic: true,
  })

  const gcps = project?.gcps || []
  const observations = project?.observations || []
  const dem = project?.dem || {}
  const images = project?.images || []

  // Automatic control measured on one photo only.
  const photosPerPoint = {}
  for (const o of observations) {
    (photosPerPoint[o.pointId] ||= new Set()).add(o.imageId)
  }
  const singlePhotoControl = gcps
    .filter((g) => g.source === 'automatic' && !g.transferTried
      && (photosPerPoint[g.id]?.size || 0) === 1)
    .map((g) => g.id)

  function observationsFor(pointId) {
    return observations.filter((o) => o.pointId === pointId)
  }

  // The clicked position is drawn on the photo until it is saved or dropped,
  // so the operator sees where the point will go before accepting it.
  useEffect(() => {
    const setPendingPoint = useStore.getState().setPendingPoint
    setPendingPoint(draft && draft.imageId != null
      ? { imageId: draft.imageId, col: draft.col, row: draft.row, id: draft.id }
      : null)
  }, [draft?.imageId, draft?.col, draft?.row, draft?.id])
  useEffect(() => () => useStore.getState().setPendingPoint(null), [])

  function startMeasure(existing) {
    if (!image) { toast('Select an image first', 'warn'); return }
    armPick({
      label: existing ? `Measure ${existing.id}` : 'New control point',
      loupe: true,
      // The clicked image, not the one active when the pick was armed: with
      // images popped out onto other monitors, those are often different.
      onPick: (col, row, picked) => {
        setDraft({
          id: existing?.id || null,
          col, row,
          imageId: (picked || image).id,
          x: existing?.x != null ? shown(existing.x) : '',
          y: existing?.y != null ? shown(existing.y) : '',
          z: existing?.z ?? '',
          isCheckPoint: existing?.isCheckPoint ?? false,
        })
      },
    })
  }

  async function extractElevation() {
    const x = numberOf(draft?.x) == null ? null : numberOf(draft.x) * axes.sign
    const y = numberOf(draft?.y) == null ? null : numberOf(draft.y) * axes.sign
    if (x == null || y == null) {
      toast(`Enter numeric ${axes.first.toLowerCase()} and ${axes.second.toLowerCase()} values first`, 'warn')
      return
    }
    const demPath = dem.referencePath
    if (!demPath) {
      toast('No reference DEM is set', 'warn')
      return
    }
    const result = await call(() => api.raster.sample(demPath, x, y), { refresh: false })
    if (!result) return
    if (result.value == null) {
      if (result.swappedValue != null) {
        setSwapHint({ z: result.swappedValue })
      } else {
        toast(`${x}, ${y} falls outside the DEM in either order. Check the coordinates `
          + 'and the control projection.', 'warn', 9000)
      }
      return
    }
    setSwapHint(null)
    setDraft({ ...draft, z: result.value.toFixed(3) })
  }

  async function saveDraft() {
    if (!draft) return
    const x = numberOf(draft.x) == null ? null : numberOf(draft.x) * axes.sign
    const y = numberOf(draft.y) == null ? null : numberOf(draft.y) * axes.sign
    const z = numberOf(draft.z)
    if (x == null || y == null || z == null) {
      toast(`Numeric ${axes.first.toLowerCase()}, ${axes.second.toLowerCase()} and ${axes.elevation.toLowerCase()} values are required`, 'warn')
      return
    }
    // Caught here rather than after the adjustment, where a reversed pair
    // shows up only as a residual of kilometres.
    if (!swapHint?.accepted) {
      if (dem.referencePath) {
        const test = await api.raster.sample(dem.referencePath, x, y).catch(() => null)
        if (test && test.value == null && test.swappedValue != null) {
          setSwapHint({ z: test.swappedValue })
          return
        }
      } else if (looksReversed(x, y)) {
        setSwapHint({ z: null })
        return
      }
    }
    await call(() => api.points.upsertGcp({
      id: draft.id || undefined,
      x, y, z,
      isCheckPoint: draft.isCheckPoint,
      measurements: [{ imageId: draft.imageId, col: draft.col, row: draft.row }],
    }), { successMessage: draft.id ? `${draft.id} updated` : 'Control point added' })
    setDraft(null)
    setSwapHint(null)
  }

  async function chooseDem() {
    const paths = await window.fiducia?.dialog.openFiles({
      title: 'Choose a reference DEM', multiple: false,
      filters: [{ name: 'Elevation', extensions: ['tif', 'tiff', 'pix', 'img', 'asc'] }],
    })
    if (!paths?.length) return
    await call(() => api.project.patch({ dem: { ...dem, referencePath: paths[0] } }),
      { successMessage: 'Reference DEM set' })
  }

  async function importControl() {
    const paths = await window.fiducia?.dialog.openFiles({
      title: 'Open a survey file', multiple: false,
      filters: [{ name: 'Survey', extensions: ['csv', 'txt', 'dat', 'asc'] },
                { name: 'All files', extensions: ['*'] }],
    })
    if (!paths?.length) return

    const preview = await call(() => api.exchange.importControl({ path: paths[0] }),
      { refresh: false })
    if (!preview) return

    const summary = preview.points.length + ' point(s) read'
      + (preview.problems.length ? ', ' + preview.problems.length + ' row(s) unreadable' : '')
      + '.\n\nColumns: ' + Object.keys(preview.columns).join(', ')
      + (preview.notes.length ? '\n\n' + preview.notes.join('\n') : '')
      + '\n\nImport these points?'

    if (!window.confirm(summary)) return
    await call(() => api.exchange.importControl({ path: paths[0], apply: true }), {
      successMessage: 'Control points imported',
    })
  }

  async function exportControl(format) {
    const target = await window.fiducia?.dialog.saveFile({
      title: 'Export control points',
      defaultPath: format === 'geojson' ? 'control.geojson' : 'control.csv',
      filters: [{ name: format === 'geojson' ? 'GeoJSON' : 'CSV',
                  extensions: [format === 'geojson' ? 'geojson' : 'csv'] }],
    })
    if (!target) return
    await call(() => api.exchange.exportControl({ path: target, format }), {
      successMessage: 'Control points exported', refresh: false,
    })
  }

  return (
    <>
      <section className="section">
        <div className="section__head">
          <span className="section__title">
            Reference DEM
            <InfoTip>Supplies control point elevations and drives orthorectification.</InfoTip>
          </span>
          <span className="section__rule" />
        </div>
        {dem.referencePath ? (
          <div className="row" style={{ borderTop: '1px solid var(--rule)' }}>
            <div className="row__main">
              <div className="row__meta truncate">{dem.referencePath}</div>
            </div>
            <button className="btn btn--ghost btn--sm" onClick={chooseDem}>Change</button>
          </div>
        ) : (
          <button className="btn btn--block" onClick={chooseDem}>Choose a DEM…</button>
        )}
      </section>

      <section className="section">
        <div className="section__head">
          <span className="section__title">Ground control</span>
          <span className="section__rule" />
          <button className="btn btn--ghost btn--sm" onClick={importControl}>
            Import…
          </button>
          <button
            className="btn btn--ghost btn--sm"
            onClick={() => exportControl('csv')}
            disabled={!gcps.length}
          >
            Export…
          </button>
          <button
            className="btn btn--primary btn--sm"
            onClick={() => startMeasure(null)}
            disabled={!image}
          >
            Measure…
          </button>
        </div>

        {draft && (
          <div style={{ borderLeft: '2px solid var(--accent)',
                        padding: '2px 0 2px var(--step-3)',
                        marginBottom: 'var(--step-4)' }}>
            <div className="section__title" style={{ marginBottom: 6 }}>
              {draft.id ? `Editing ${draft.id}` : 'New point'}
            </div>

            <div className="field__hint" style={{ marginBottom: 6 }}>
              Image position <span className="numeric">
                {draft.col.toFixed(1)}, {draft.row.toFixed(1)}
              </span>
            </div>

            <div className="grid-3">
              {[['x', axes.first], ['y', axes.second], ['z', axes.elevation]].map(
                ([axis, label]) => (
                  <div className="field" key={axis}>
                    <label className="field__label">
                      {label}
                      {axis === 'z' && <span className="dim">m</span>}
                    </label>
                    <input
                      className="numeric"
                      value={draft[axis]}
                      onChange={(event) => enterCoordinate(axis, event.target.value)}
                    />
                  </div>
                ))}
            </div>

            <div className="field__hint" style={{ marginBottom: 6 }}>
              {crs ? crs.label : 'No control projection set'}
              {axes.note ? `. ${axes.note}` : ''}
            </div>
            {isLoEastingNorthing(crs) && (
              <label className="field field--row" style={{ marginBottom: 6 }}>
                <span className="field__label">Show as the surveyor's Y and X</span>
                <input type="checkbox" checked={loAsYX} onChange={(event) => setLoAsYX(event.target.checked)} />
              </label>
            )}

            {swapHint && !swapHint.accepted && (
              <div className="readiness" style={{ borderLeftColor: 'var(--signal-warn)',
                                                  marginBottom: 6 }}>
                <div className="readiness__item readiness__item--warning">
                  <span className="dot" />
                  <span>
                    {swapHint.z != null
                      ? <>The coordinates appear to be reversed. As entered, the point
                          falls outside the DEM; swapped, it falls inside at{' '}
                          {swapHint.z.toFixed(2)} m.</>
                      : <>The coordinates appear to be reversed. The{' '}
                          {axes.first.toLowerCase()} is entered first.</>}
                  </span>
                </div>
                <div style={{ display: 'flex', gap: 6, marginTop: 6 }}>
                  <button className="btn btn--primary btn--sm" onClick={swapDraft}>
                    Swap
                  </button>
                  <button className="btn btn--ghost btn--sm"
                          onClick={() => setSwapHint({ ...swapHint, accepted: true })}>
                    Keep as entered
                  </button>
                </div>
              </div>
            )}

            <label className="field field--row">
              <span className="field__label">Check point</span>
              <input
                type="checkbox" checked={draft.isCheckPoint}
                onChange={(event) => setDraft({ ...draft, isCheckPoint: event.target.checked })}
              />
            </label>

            <div style={{ display: 'flex', gap: 6 }}>
              <button className="btn btn--sm" onClick={extractElevation}
                      disabled={!dem.referencePath}>
                Extract Z from DEM
              </button>
              <button className="btn btn--primary btn--sm" onClick={saveDraft}>Accept</button>
              <button className="btn btn--ghost btn--sm" onClick={() => setDraft(null)}>
                Cancel
              </button>
            </div>
          </div>
        )}

        <LivePreview
          preview={preview}
          image={image}
          onFix={(fix) => call(() => api.camera.set({ ...(project?.camera || {}), ...fix }),
            { successMessage: 'Camera updated' })}
        />

        {gcps.length === 0 ? (
          <div className="empty-note">
            No control points. A minimum of three is required; four or more
            provide redundancy.
          </div>
        ) : (
          <ControlTable
            gcps={gcps}
            project={project}
            image={image}
            preview={preview}
            residualOf={residualOf}
            selectedPointId={selectedPointId}
            setSelectedPoint={setSelectedPoint}
            observationsFor={observationsFor}
            axes={axes}
            onMeasure={startMeasure}
            onToggleCheck={(gcp) => call(() => api.model.setCheckPoint(gcp.id, !gcp.isCheckPoint))}
            onDelete={(gcp) => call(() => api.points.deleteGcp(gcp.id))}
          />
        )}
      </section>

      <section className="section">
        <div className="section__head">
          <span className="section__title">
            Automatic control
            <InfoTip>
              <p>Matches each image against an orthorectified reference mosaic.
              Image patches are rectified before correlation.</p>
              <p>No points need measuring first: each photo is placed on the
              reference automatically. After computing the model, run it again
              for more precise points; they replace the first pass.</p>
            </InfoTip>
          </span>
          <span className="section__rule" />
          <button
            className="btn btn--primary btn--sm"
            onClick={findControl}
            disabled={!!autoJobId || !dem.referenceImage}
          >
            {autoJobId ? 'Matching…' : 'Find control'}
          </button>
        </div>

        <div className="field">
          <label className="field__label">Reference image</label>
          {dem.referenceImage ? (
            <div className="row" style={{ borderTop: '1px solid var(--rule)' }}>
              <div className="row__main">
                {(dem.referenceImages || []).length > 1 ? (
                  <div className="row__meta truncate"
                       title={dem.referenceImages.join('\n')}>
                    {dem.referenceImages.length} tiles joined:{' '}
                    {dem.referenceImages.map((p) => p.split(/[\\/]/).pop()).join(', ')}
                  </div>
                ) : (
                  <div className="row__meta truncate">{dem.referenceImage}</div>
                )}
              </div>
              <button className="btn btn--ghost btn--sm" onClick={chooseReference}>
                Change
              </button>
            </div>
          ) : (
            <button className="btn btn--block" onClick={chooseReference}>
              Choose a geocoded image…
            </button>
          )}
        </div>

        {singlePhotoControl.length > 0 && images.filter((i) => i.online).length > 1 && (
          <div className="field">
            <button
              className="btn btn--block"
              onClick={() => call(() => api.auto.transferGcps(singlePhotoControl), {
                successMessage: 'Control measured on the overlapping photos',
              })}
            >
              Measure {singlePhotoControl.length} control point
              {singlePhotoControl.length === 1 ? '' : 's'} on all photos
            </button>
            <div className="field__hint">
              These are measured on one photo only. Measured on every photo that sees
              them, they tie the photos together and check height as well as position.
            </div>
          </div>
        )}

        <details className="advanced">
          <summary>Options</summary>
          <div className="advanced__body">
        <div className="grid-3">
          <div className="field">
            <label className="field__label">Target count</label>
            <input
              type="number" className="numeric" value={autoOptions.targetCount}
              onChange={(event) => setAutoOptions({
                ...autoOptions, targetCount: Number(event.target.value),
              })}
            />
          </div>
          <div className="field">
            <label className="field__label">Search <span className="dim">m</span></label>
            <input
              type="number" className="numeric" value={autoOptions.searchM}
              onChange={(event) => setAutoOptions({
                ...autoOptions, searchM: Number(event.target.value),
              })}
            />
          </div>
          <div className="field">
            <label className="field__label">Min score</label>
            <input
              type="number" step="0.05" className="numeric" value={autoOptions.minScore}
              onChange={(event) => setAutoOptions({
                ...autoOptions, minScore: Number(event.target.value),
              })}
            />
          </div>
        </div>
          </div>
        </details>

        {autoJob && (autoJob.status === 'running' || autoJob.status === 'queued') && (
          <div className="field__hint">{autoJob.message}</div>
        )}

        {proposals && (
          <div style={{ marginTop: 'var(--step-3)' }}>
            {proposals.map((entry) => (
              <div key={entry.imageId} style={{ marginBottom: 'var(--step-3)' }}>
                <div className="row__name" style={{ marginBottom: 4 }}>
                  {entry.name}
                  <span className={entry.error ? 'chip chip--bad' : 'chip'}>
                    {entry.error ? 'failed' : (entry.proposals || []).length + ' found'}
                  </span>
                </div>
                <div className="field__hint" style={{ marginBottom: 4 }}>
                  {entry.error || entry.message}
                </div>
                {(entry.warnings || []).map((warning, index) => (
                  <div key={index} className="field__hint"
                       style={{ color: 'var(--signal-warn)' }}>
                    {warning}
                  </div>
                ))}
                {(entry.proposals || []).map((proposal, index) => {
                  const key = entry.imageId + ':' + index
                  const off = rejected.has(key)
                  return (
                    <label
                      key={key}
                      className={proposal.score < 0.7 ? 'proposal proposal--weak' : 'proposal'}
                      style={{ opacity: off ? 0.4 : 1, cursor: 'pointer' }}
                    >
                      <input
                        type="checkbox" checked={!off} style={{ width: 13, flex: 'none' }}
                        onChange={() => setRejected((current) => {
                          const next = new Set(current)
                          if (next.has(key)) next.delete(key)
                          else next.add(key)
                          return next
                        })}
                      />
                      <span className="proposal__main">
                        <span className="proposal__coords">
                          {shown(proposal.x).toFixed(2)} {axes.firstShort},{' '}
                          {shown(proposal.y).toFixed(2)} {axes.secondShort},{' '}
                          {proposal.z.toFixed(2)}
                        </span>
                        <span className="proposal__meta">
                          pixel {proposal.col.toFixed(0)}, {proposal.row.toFixed(0)};
                          {' '}score {proposal.score.toFixed(3)}; shift{' '}
                          {proposal.shiftM.toFixed(1)} m
                        </span>
                      </span>
                    </label>
                  )
                })}
              </div>
            ))}

            {earlierAutomatic > 0 && (
              <label className="field field--row" style={{ marginBottom: 6 }}>
                <span className="field__label">
                  Replace the {earlierAutomatic} earlier automatic point
                  {earlierAutomatic === 1 ? '' : 's'} on these photos
                </span>
                <input
                  type="checkbox" checked={replaceEarlier}
                  onChange={(event) => setReplaceEarlier(event.target.checked)}
                />
              </label>
            )}

            <div style={{ display: 'flex', gap: 6 }}>
              <button
                className="btn btn--primary btn--sm"
                disabled={!accepted.length}
                onClick={async () => {
                  const replacing = replaceEarlier && earlierAutomatic > 0
                  await call(() => api.auto.applyGcps(accepted, replacing), {
                    successMessage: accepted.length + ' control point'
                      + (accepted.length === 1 ? '' : 's')
                      + (replacing ? ` added, ${earlierAutomatic} earlier replaced` : ' added'),
                  })
                  setProposals(null)
                }}
              >
                Accept {accepted.length}
              </button>
              <button className="btn btn--ghost btn--sm" onClick={() => setProposals(null)}>
                Discard
              </button>
            </div>
          </div>
        )}
      </section>

      <section className="section">
        <div className="section__head">
          <span className="section__title">
            Tie points
            <InfoTip>
              <p>Collected in the background across all overlapping photos. Candidates
              are found on reduced images, filtered by a RANSAC two-view geometry check,
              then measured again at full resolution by least-squares matching.</p>
              <p>A feature seen on three or more photos becomes one multi-ray point,
              which ties the strip together far better than separate pairs.</p>
            </InfoTip>
          </span>
          <span className="section__rule" />
          <span className="chip">{(project?.tiePoints || []).length}</span>
        </div>

        {images.filter((i) => i.online).length < 2 ? (
          <div className="empty-note">
            At least two overlapping images are required.
          </div>
        ) : (
          <>
            <details className="advanced">
              <summary>Options</summary>
              <div className="advanced__body">
            <div className="field">
              <label className="field__label">Matching method</label>
              <select
                value={tieOptions.method}
                onChange={(event) => setTieOptions({ ...tieOptions, method: event.target.value })}
              >
                <option value="ncc">Normalised cross-correlation</option>
                <option value="fbm">Feature-based (AKAZE)</option>
              </select>
              <div className="field__hint">
                {tieOptions.method === 'ncc'
                  ? 'Suited to imagery flown in a single session.'
                  : 'Scale and rotation invariant. Suited to mixed orientations.'}
              </div>
            </div>

            <div className="grid-2">
              <div className="field">
                <label className="field__label">Target per photo pair</label>
                <input
                  type="number" className="numeric" value={tieOptions.targetCount}
                  onChange={(event) => setTieOptions({
                    ...tieOptions, targetCount: Number(event.target.value),
                  })}
                />
              </div>
              <div className="field">
                <label className="field__label">Min score</label>
                <input
                  type="number" step="0.05" min="0" max="1" className="numeric"
                  value={tieOptions.minScore}
                  onChange={(event) => setTieOptions({
                    ...tieOptions, minScore: Number(event.target.value),
                  })}
                />
              </div>
            </div>

            <div className="field">
              <label className="field__label">
                <span>
                  Full-resolution refinement
                  <InfoTip>
                    Least-squares matching fits an affine and a brightness correction
                    between the patches and reaches about a tenth of a pixel, with a
                    precision estimate for every point. Correlation alone reaches about
                    a third of a pixel.
                  </InfoTip>
                </span>
              </label>
              <select
                value={tieOptions.refine}
                onChange={(event) => setTieOptions({ ...tieOptions, refine: event.target.value })}
              >
                <option value="lsm">Least-squares matching</option>
                <option value="ncc">Correlation only</option>
                <option value="none">None (reduced image only, fastest)</option>
              </select>
            </div>

            <label className="field field--row">
              <span className="field__label">Replace earlier automatic tie points</span>
              <input
                type="checkbox" checked={tieOptions.replaceAutomatic}
                onChange={(event) => setTieOptions({
                  ...tieOptions, replaceAutomatic: event.target.checked,
                })}
              />
            </label>
              </div>
            </details>

            <button
              className="btn btn--primary btn--block"
              onClick={() => call(() => api.points.autoTie(tieOptions))}
            >
              Collect tie points
            </button>
          </>
        )}
      </section>
    </>
  )
}


/**
 * Residuals as control is collected, for the photo on the canvas.
 *
 * From the fourth point on, every change re-solves this one photo and shows
 * how well the control agrees with it, so a mistyped coordinate or a
 * misplaced click shows up while the point is still fresh in mind -- not an
 * hour later, as a bad adjustment.
 */
/**
 * The control points as a table, worst first: what a check of the control
 * needs to see at once. Errors are the block solution's (surveyed minus
 * adjusted); "This photo" is the live fit of the photo on screen. Control
 * points are coloured by the fit rule, check points by the accuracy target.
 */
function ControlTable({
  gcps, project, image, preview, residualOf, selectedPointId, setSelectedPoint,
  observationsFor, axes, onMeasure, onToggleCheck, onDelete,
}) {
  const [sort, setSort] = useState({ key: 'error', descending: true })
  const residuals = project?.model?.controlResidualsM || {}
  const target = project?.accuracyTarget || {}

  const rows = gcps.map((gcp) => {
    const r = residuals[gcp.id]
    const dz = r && r[2] != null && Number.isFinite(r[2]) ? r[2] : null
    const horizontal = r && Number.isFinite(r[0]) ? Math.hypot(r[0], r[1]) : null
    const rays = observationsFor(gcp.id).length
    const live = preview?.ready ? residualOf[gcp.id] : null
    return {
      gcp, rays, r, horizontal, dz, live,
      complete: gcp.x != null && gcp.y != null && gcp.z != null,
      error: horizontal == null ? -1 : Math.hypot(horizontal, dz ?? 0),
    }
  })
  const value = {
    id: (row) => row.gcp.id,
    rays: (row) => row.rays,
    error: (row) => row.error,
    horizontal: (row) => row.horizontal ?? -1,
    dz: (row) => (row.dz == null ? -1 : Math.abs(row.dz)),
    px: (row) => row.live?.px ?? -1,
  }[sort.key]
  rows.sort((a, b) => {
    const x = value(a)
    const y = value(b)
    const order = typeof x === 'string' ? x.localeCompare(y, undefined, { numeric: true }) : x - y
    return sort.descending ? -order : order
  })

  const header = (key, label, title, numeric = true) => (
    <th
      className={numeric ? 'num' : ''}
      title={title}
      aria-sort={sort.key === key ? (sort.descending ? 'descending' : 'ascending') : 'none'}
      onClick={() => setSort({ key, descending: sort.key === key ? !sort.descending : key !== 'id' })}
      style={{ cursor: 'pointer' }}
    >
      {label}{sort.key === key ? (sort.descending ? ' ↓' : ' ↑') : ''}
    </th>
  )
  const toneClass = (tone) => (tone ? `tone--${tone}` : '')
  // Two decimals, without the "-0.00" of a tiny negative.
  const metres = (v) => (Math.abs(v) < 0.005 ? 0 : v).toFixed(2)

  return (
    <div className="table-scroll">
      <table className="table table--control">
        <thead>
          <tr>
            {header('id', 'Point', 'Control holds the solution; a check point only tests it', false)}
            {header('rays', 'Rays', 'Photos the point is measured on')}
            {header('horizontal', 'XY m', 'Horizontal error in the block solution, surveyed minus adjusted')}
            {header('dz', 'Z m', 'Height error in the block solution, surveyed minus adjusted')}
            {header('px', 'Photo px', `Fit on ${image?.name || 'the photo on screen'} on its own, in pixels`)}
            <th aria-label="Actions" />
          </tr>
        </thead>
        <tbody>
          {rows.map(({ gcp, rays, r, horizontal, dz, live, complete }) => {
            const check = gcp.isCheckPoint
            const onePhoto = check && rays === 1
            const xyTone = horizontal == null ? ''
              : check ? accuracyTone(horizontal, target.horizontalM) : fitToneM(horizontal, project)
            const zTone = dz == null || !check ? '' : accuracyTone(Math.abs(dz), target.verticalM)
            const where = complete
              ? `${(Number(gcp.x) * axes.sign).toFixed(2)} ${axes.firstShort}, ${(Number(gcp.y) * axes.sign).toFixed(2)} ${axes.secondShort}, ${Number(gcp.z).toFixed(2)} m`
              : 'Coordinates incomplete'
            return (
              <tr
                key={gcp.id}
                className={selectedPointId === gcp.id ? 'is-selected' : ''}
                onMouseEnter={() => setSelectedPoint(gcp.id)}
                onMouseLeave={() => setSelectedPoint(null)}
              >
                <td title={where}>
                  <strong>{gcp.id}</strong>
                  {check && <span className="dim"> check</span>}
                  {!complete && <span className="chip chip--warn" style={{ marginLeft: 4 }}>incomplete</span>}
                  {onePhoto && (
                    <span className="chip chip--warn" style={{ marginLeft: 4 }}
                          title="Measured on one photo only: it checks position at its surveyed height, not height. Measure it on a second photo.">
                      1 photo
                    </span>
                  )}
                </td>
                <td className="num">{rays}</td>
                <td className={`num ${toneClass(xyTone)}`}
                    title={r ? `dX ${r[0].toFixed(3)} m, dY ${r[1].toFixed(3)} m` : 'Not in the solution yet'}>
                  {horizontal == null ? '—' : metres(horizontal)}
                </td>
                <td className={`num ${toneClass(zTone)}`}
                    title={onePhoto ? 'Height not checked: measured on one photo' : ''}>
                  {dz == null ? (onePhoto ? 'n/c' : '—') : metres(dz)}
                </td>
                <td className={`num ${preview?.worst === gcp.id ? 'tone--bad'
                  : toneClass(live?.px != null ? fitTonePx(live.px) : '')}`}
                    title={live ? `${live.groundM.toFixed(2)} m on the ground. ${fitRule(project)}` : ''}>
                  {live?.px != null ? live.px.toFixed(1) : '—'}
                </td>
                <td className="table__actions">
                  <button className="btn btn--ghost btn--sm" title="Measure on the current image"
                          aria-label="Measure on the current image"
                          onClick={() => onMeasure(gcp)} disabled={!image}>+</button>
                  <button className="btn btn--ghost btn--sm"
                          title={check ? 'Use as control: in the solution' : 'Use as a check point: out of the solution, testing it'}
                          onClick={() => onToggleCheck(gcp)}>
                    {check ? '↺' : '⊘'}
                  </button>
                  <button className="btn btn--ghost btn--sm btn--danger" title="Delete the point"
                          onClick={() => onDelete(gcp)}>✕</button>
                </td>
              </tr>
            )
          })}
        </tbody>
      </table>
    </div>
  )
}

function LivePreview({ preview, image, onFix }) {
  const project = useStore((s) => s.project)
  if (!image || !preview) return null

  const setup = preview.setupProblems || []
  if (setup.length) {
    return (
      <div className="readiness readiness--blocked" style={{ marginBottom: 'var(--step-3)' }}>
        {setup.map((problem) => (
          <div key={problem.code}>
            <div className="readiness__item readiness__item--blocker">
              <span className="dot" />
              <span>{problem.message}</span>
            </div>
            {(problem.fixes || (problem.fix ? [{ fix: problem.fix, label: problem.fixLabel }] : []))
              .map((option) => (
                <button key={option.label} className="btn btn--sm"
                        style={{ margin: '4px 0 6px 16px', display: 'block' }}
                        onClick={() => onFix(option.fix)}>
                  {option.label || 'Apply fix'}
                </button>
              ))}
          </div>
        ))}
      </div>
    )
  }

  if (!preview.ready) {
    return (
      <div className="field__hint" style={{ marginBottom: 'var(--step-3)' }}>
        Residuals: {preview.message}
      </div>
    )
  }

  const tone = preview.quality === 'good' ? 'var(--signal-good)'
    : preview.quality === 'poor' ? 'var(--signal-warn)' : 'var(--signal-bad)'
  const worst = preview.worst && preview.points.find((p) => p.pointId === preview.worst)

  return (
    <div className="readiness" style={{ borderLeftColor: tone, marginBottom: 'var(--step-3)' }}>
      <div className="readiness__item">
        <span className="dot" style={{ background: tone }} />
        <span>
          <strong>{image.name} on its own</strong>: RMS{' '}
          {preview.rmsPx != null ? (
            <><span className="numeric">{preview.rmsPx.toFixed(1)} px</span>
              {' '}({preview.rmsGroundM.toFixed(2)} m on the ground)</>
          ) : <span className="numeric">{preview.rmsGroundM.toFixed(2)} m</span>}
          {' '}from {preview.count} control points ({preview.degreesOfFreedom} redundant),
          before the block is adjusted. {fitRule(project)}
        </span>
      </div>
      {worst && (
        <div className="readiness__item readiness__item--warning">
          <span className="dot" />
          <span>
            <strong>{worst.pointId}</strong> is inconsistent with the other points
            (remaining fit {worst.restPx.toFixed(1)} px; misclosure{' '}
            {worst.withoutItM.toFixed(2)} m). Check its coordinates and image position.
          </span>
        </div>
      )}
      {(preview.problems || []).map((problem, index) => (
        <div key={index} className="readiness__item readiness__item--warning">
          <span className="dot" />
          <span>{problem}</span>
        </div>
      ))}
      {preview.note && (
        <div className="field__hint" style={{ marginTop: 4 }}>{preview.note}</div>
      )}
    </div>
  )
}
