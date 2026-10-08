import { useEffect, useState } from 'react'
import { useStore, selectActiveImage, workflowOf } from '../../state/store'
import { api } from '../../lib/api'
import InfoTip from '../InfoTip'

const PAD = [
  'top_left', 'top_middle', 'top_right',
  'left_middle', null, 'right_middle',
  'bottom_left', 'bottom_middle', 'bottom_right',
]

const SHORT = {
  top_left: 'TL', top_middle: 'TM', top_right: 'TR',
  left_middle: 'LM', right_middle: 'RM',
  bottom_left: 'BL', bottom_middle: 'BM', bottom_right: 'BR',
}

export default function ImagesPanel() {
  const project = useStore((s) => s.project)
  const image = useStore(selectActiveImage)
  const setActiveImage = useStore((s) => s.setActiveImage)
  const armPick = useStore((s) => s.armPick)
  const call = useStore((s) => s.call)
  const toast = useStore((s) => s.toast)
  const popOut = useStore((s) => s.popOut)
  const poppedOut = useStore((s) => s.poppedOut)
  const canPopOut = !!window.fiducia?.windows

  const images = project?.images || []
  const camera = project?.camera || {}
  const isFilm = (project?.mathModel?.kind || 'aerial_film') === 'aerial_film'
  const drone = workflowOf(project).id === 'drone'
  const calibrated = camera.fiducialsMm || {}
  const fit = image?.fiducialFit

  const [activeSlot, setActiveSlot] = useState(null)
  const [detectJobId, setDetectJobId] = useState(null)
  const [detected, setDetected] = useState(null)
  const jobs = useStore((s) => s.jobs)
  const detectJob = jobs.find((j) => j.id === detectJobId)
  const alignJob = jobs.find((j) => j.kind === 'drone'
    && (j.status === 'running' || j.status === 'queued'))

  useEffect(() => {
    if (detectJob?.status === 'done' && detectJob.result) {
      setDetected(detectJob.result.results)
      setDetectJobId(null)
    } else if (detectJob?.status === 'failed') {
      setDetectJobId(null)
    }
  }, [detectJob?.status])

  // A photo already measured by hand supplies the templates. Without one the
  // detector falls back to a generated cross, which is less certain.
  const templateImage = images.find(
    (i) => i.online && Object.keys(i.fiducials || {}).length >= 3,
  )

  async function detectFiducials() {
    const targets = images
      .filter((i) => i.online && i.id !== templateImage?.id)
      .map((i) => i.id)
    const list = targets.length ? targets : images.filter((i) => i.online).map((i) => i.id)
    if (!list.length) {
      toast('No online images', 'warn')
      return
    }
    setDetected(null)
    const response = await call(
      () => api.auto.fiducials({ imageIds: list, templateImageId: templateImage?.id }),
      { refresh: false },
    )
    if (response?.job) setDetectJobId(response.job.id)
  }

  async function addImages() {
    if (!window.fiducia?.isDesktop) {
      toast('Images can only be added in the desktop app', 'warn')
      return
    }
    const paths = await window.fiducia.dialog.openFiles({ title: 'Add imagery' })
    if (!paths?.length) return
    const result = await call(() => api.images.add(paths))
    if (result?.added?.length && !image) setActiveImage(result.added[0].id)
  }

  async function relinkOne(target) {
    const paths = await window.fiducia.dialog.openFiles({
      title: `Locate ${target.name}`, multiple: false,
    })
    if (!paths?.length) return
    await call(() => api.project.relink({ imageId: target.id, path: paths[0] }),
      { successMessage: `${target.name} relinked` })
  }

  function measureFiducial(slot) {
    if (!image) return
    setActiveSlot(slot)
    armPick({
      label: `Fiducial ${SHORT[slot]}`,
      onPick: (col, row, picked) => {
        call(() => api.images.setFiducial((picked || image).id, { slot, col, row }),
          { refresh: true })
        setActiveSlot(null)
      },
    })
  }

  // Two clicks, one per corner. Between them the viewer draws the region
  // from the first corner to the cursor, so the second click is aimed.
  function defineClip() {
    if (!image) return
    const setClipDraft = useStore.getState().setClipDraft
    toast('Click one corner of the clip region', 'accent')
    armPick({
      label: 'Clip — first corner',
      onPick: (c0, r0, first) => {
        const target = first || image
        setClipDraft({ imageId: target.id, col: c0, row: r0 })
        toast('Click the opposite corner', 'accent')
        armPick({
          label: 'Clip — opposite corner',
          onCancel: () => setClipDraft(null),
          onPick: (c1, r1, second) => {
            setClipDraft(null)
            if (second && second.id !== target.id) {
              toast(`Both corners must be on ${target.name}. Define the region again.`, 'warn')
              return
            }
            if (Math.abs(c1 - c0) < 2 || Math.abs(r1 - r0) < 2) {
              toast('The two corners are too close together. Define the region again.', 'warn')
              return
            }
            call(() => api.images.patch(target.id, {
              clipRegion: [
                Math.min(c0, c1), Math.min(r0, r1),
                Math.max(c0, c1), Math.max(r0, r1),
              ],
            }), { successMessage: 'Clip region set' })
          },
        })
      },
    })
  }

  const offline = images.filter((i) => !i.online)

  return (
    <>
      {!isFilm && images.filter((i) => i.online).length >= 2 && (
        <DroneAlignment block={project?.droneBlock} job={alignJob} call={call} />
      )}

      <section className="section">
        <div className="section__head">
          <span className="section__title">Imagery</span>
          <span className="section__rule" />
          {canPopOut && images.filter((i) => i.online).length > 1 && (
            <button
              className="btn btn--ghost btn--sm"
              title="Open each image in a separate window"
              onClick={async () => {
                for (const entry of images.filter((i) => i.online)) await popOut(entry)
              }}
            >
              Pop out all
            </button>
          )}
          <button className="btn btn--primary btn--sm" onClick={addImages}>Add…</button>
        </div>

        {offline.length > 0 && (
          <div className="readiness readiness--blocked">
            <div className="readiness__item readiness__item--blocker">
              <span className="dot" />
              <span>
                {offline.length} image{offline.length === 1 ? '' : 's'} not found at
                the stored location.
              </span>
            </div>
            <button
              className="btn btn--sm btn--block"
              style={{ marginTop: 6 }}
              onClick={() => call(() => api.project.relink(),
                { successMessage: 'Searched known folders' })}
            >
              Search known folders
            </button>
          </div>
        )}

        {images.length === 0 ? (
          drone ? (
            <div className="empty-note">
              Add the flight's photos (the JPEGs straight off the drone). Fiducia reads
              the camera and each photo's GPS position from them, then aligns them for
              you.
            </div>
          ) : (
            <div className="empty-note">
              No imagery. Supported formats: GeoTIFF, JPEG and PCIDSK{' '}
              <span className="numeric">.pix</span>.
            </div>
          )
        ) : (
          <div className="rows">
            {images.map((entry) => {
              const entryFit = entry.fiducialFit
              return (
                <button
                  key={entry.id}
                  className={`row ${image?.id === entry.id ? 'row--active' : ''}`}
                  onClick={() => setActiveImage(entry.id)}
                >
                  <div className="row__main">
                    <div className="row__name truncate">
                      {entry.name}
                      {!entry.online && <span className="chip chip--bad">offline</span>}
                      {isFilm && entry.online && entryFit && (
                        <span className={`chip ${entryFit.rmsPx < 2 ? 'chip--good' : 'chip--warn'}`}>
                          {entryFit.rmsPx.toFixed(2)} px
                        </span>
                      )}
                      {isFilm && entry.online && !entryFit && (
                        <span className="chip chip--warn">no IO</span>
                      )}
                      {entry.exterior && entry.exteriorSource === 'gnss' && (
                        <span className="chip">GPS</span>
                      )}
                      {entry.exterior && entry.exteriorSource !== 'gnss' && (
                        <span className="chip chip--accent">solved</span>
                      )}
                    </div>
                    <div className="row__meta truncate">
                      {entry.width} × {entry.height}, {entry.bandCount} band
                      {entry.clipRegion ? ', clipped' : ''}
                    </div>
                  </div>
                  <div className="row__actions">
                    {canPopOut && entry.online && (
                      <button
                        className={`btn btn--ghost btn--sm ${poppedOut.includes(entry.id) ? 'btn--on' : ''}`}
                        title={poppedOut.includes(entry.id)
                          ? 'Bring window to front'
                          : 'Open in a separate window'}
                        onClick={(event) => { event.stopPropagation(); popOut(entry) }}
                      >
                        ⧉
                      </button>
                    )}
                    {!entry.online && (
                      <button
                        className="btn btn--ghost btn--sm"
                        onClick={(event) => { event.stopPropagation(); relinkOne(entry) }}
                      >
                        Locate…
                      </button>
                    )}
                    <button
                      className="btn btn--ghost btn--sm btn--danger"
                      onClick={(event) => {
                        event.stopPropagation()
                        if (!window.confirm(
                          `Remove ${entry.name} and its measurements?`)) return
                        call(() => api.images.remove(entry.id))
                      }}
                    >
                      ✕
                    </button>
                  </div>
                </button>
              )
            })}
          </div>
        )}
      </section>

      {image && isFilm && (
        <section className="section">
          <div className="section__head">
            <span className="section__title">
              Interior orientation
              <InfoTip>
                <p>Select a mark in the grid, then select the fiducial on the
                image. The affine fit and residuals update after each mark.</p>
                <p>Detect all uses the first image with three or more measured
                marks as the template.</p>
              </InfoTip>
            </span>
          <span className="section__rule" />
            <button
              className="btn btn--sm"
              onClick={detectFiducials}
              disabled={!!detectJobId || Object.keys(calibrated).length < 3}
              title={templateImage
                ? `Detect marks on all images, using ${templateImage.name} as the template`
                : 'Detect marks on all images, using a generated template'}
            >
              {detectJobId ? 'Detecting…' : 'Detect all'}
            </button>
            {fit && (
              <span className={`chip ${fit.rmsPx < 2 ? 'chip--good' : 'chip--warn'}`}>
                RMS {fit.rmsPx.toFixed(3)} px
              </span>
            )}
          </div>

          {Object.keys(calibrated).length < 3 ? (
            <div className="empty-note">
              Calibrated fiducial positions are required. Enter them in the Camera step.
            </div>
          ) : (
            <>
              {detected && (
                <div className="readiness" style={{ borderLeftColor: 'var(--accent)' }}>
                  {detected.map((entry) => (
                    <div key={entry.imageId} className="readiness__item">
                      <span className="dot" />
                      <span>
                        <strong>{entry.name}</strong> — {entry.error || entry.message}
                      </span>
                    </div>
                  ))}
                  <div style={{ display: 'flex', gap: 6, marginTop: 6 }}>
                    <button
                      className="btn btn--primary btn--sm"
                      onClick={async () => {
                        const usable = detected.filter((d) => !d.error && d.accepted >= 3)
                        if (!usable.length) { toast('No detections met the confidence threshold', 'warn'); return }
                        await call(() => api.auto.applyFiducials(usable), {
                          successMessage: `Marks applied to ${usable.length} image${usable.length === 1 ? '' : 's'}`,
                        })
                        setDetected(null)
                      }}
                    >
                      Apply to {detected.filter((d) => !d.error && d.accepted >= 3).length} image(s)
                    </button>
                    <button className="btn btn--ghost btn--sm" onClick={() => setDetected(null)}>
                      Discard
                    </button>
                  </div>
                </div>
              )}

              <div className="fidpad">
                {PAD.map((slot, index) => {
                  if (!slot) {
                    return <div key={index} className="fidpad__slot fidpad__slot--empty" />
                  }
                  const measured = (image.fiducials || {})[slot]
                  const known = !!calibrated[slot]
                  const residualIndex = fit?.used?.indexOf(slot)
                  const residual = residualIndex >= 0
                    ? fit.residualsPx[residualIndex] : null

                  return (
                    <button
                      key={slot}
                      className={[
                        'fidpad__slot',
                        measured ? 'fidpad__slot--set' : '',
                        activeSlot === slot ? 'fidpad__slot--active' : '',
                      ].join(' ')}
                      disabled={!known}
                      title={known
                        ? `Measure ${slot.replace(/_/g, ' ')}`
                        : 'Not defined on the certificate'}
                      onClick={() => measureFiducial(slot)}
                    >
                      <span>{SHORT[slot]}</span>
                      {measured && (
                        <span className="fidpad__res">
                          {residual != null ? `${residual.toFixed(2)}px` : '✓'}
                        </span>
                      )}
                    </button>
                  )
                })}
              </div>

              {fit && (
                <div className="measures" style={{ marginTop: 'var(--step-3)' }}>
                  <div className="measure">
                    <div className="measure__name">RMS residual</div>
                    <div className={`measure__value ${fit.rmsPx < 2 ? 'measure__value--good'
                      : fit.rmsPx < 4 ? 'measure__value--warn' : 'measure__value--bad'}`}>
                      {fit.rmsPx.toFixed(3)}
                    </div>
                  </div>
                  <div className="measure">
                    <div className="measure__name">Worst mark</div>
                    <div className="measure__value">{fit.maxPx.toFixed(3)}</div>
                  </div>
                </div>
              )}

              {image.fiducialError && !fit && (
                <div className="field__hint" style={{ color: 'var(--signal-warn)' }}>
                  {image.fiducialError}
                </div>
              )}

              {Object.keys(image.fiducials || {}).length > 0 && (
                <button
                  className="btn btn--sm btn--block"
                  style={{ marginTop: 6 }}
                  onClick={() => call(() => api.images.setFiducial(image.id, { fiducials: {} }))}
                >
                  Clear all marks
                </button>
              )}
            </>
          )}
        </section>
      )}

      {image && (
        <section className="section">
          <div className="section__head">
            <span className="section__title">
              Clip region
              <InfoTip>
                Limits the output extent only. Points outside the region remain
                in the adjustment.
              </InfoTip>
            </span>
          <span className="section__rule" />
            {image.clipRegion && (
              <button
                className="btn btn--ghost btn--sm"
                onClick={() => call(() => api.images.patch(image.id, { clipRegion: null }))}
              >
                Clear
              </button>
            )}
          </div>

          <button className="btn btn--block" onClick={defineClip}>
            {image.clipRegion ? 'Redefine…' : 'Define…'}
          </button>

          {image.clipRegion && (
            <div className="field__hint" style={{ marginTop: 6 }}>
              <span className="numeric">
                {image.clipRegion.map((v) => Math.round(v)).join(', ')}
              </span>
            </div>
          )}
        </section>
      )}
    </>
  )
}


/**
 * Drone photos placed from their own GPS: camera, starting orientations and
 * tie points in one step, ready for the bundle on the Model step.
 */
function DroneAlignment({ block, job, call }) {
  const running = !!job
  const heading = (block?.headingSource || []).join(', ')
  return (
    <section className="section">
      <div className="section__head">
        <span className="section__title">Drone alignment</span>
        <span className="section__rule" />
        <InfoTip>
          Reads each photo's GPS position, sets up the camera from the photo metadata
          and matches features across overlapping photos. Where the photos do not
          record their heading or height above the ground, both are measured from how
          the features move between neighbours. Solve the block on the Model step
          afterwards.
        </InfoTip>
      </div>

      {block && (
        <>
          <div className="measures">
            <div className="measure">
              <div className="measure__name">Tie points</div>
              <div className="measure__value">{block.tiePoints?.toLocaleString()}</div>
            </div>
            <div className="measure">
              <div className="measure__name">On 3+ photos</div>
              <div className="measure__value">{block.multiRay?.toLocaleString()}</div>
            </div>
            <div className="measure">
              <div className="measure__name">Flying height</div>
              <div className="measure__value">{block.flyingHeightM?.toFixed(1)} m</div>
            </div>
            <div className="measure">
              <div className="measure__name">Ground pixel</div>
              <div className="measure__value">{(block.gsdM * 100).toFixed(1)} cm</div>
            </div>
          </div>
          <div className="field__hint" style={{ marginTop: 'var(--step-2)' }}>
            {block.photos} photos, {block.pairsMatched} of {block.pairsTried} overlapping
            pairs matched, in <span className="numeric">{block.crs}</span>.
            {' '}Height from {block.heightSource}{heading ? `, heading from ${heading}` : ''}.
            {block.withoutPosition?.length > 0 && (
              <> Without GPS, left out: {block.withoutPosition.join(', ')}.</>
            )}
          </div>
        </>
      )}

      <button
        className={`btn btn--block ${block ? '' : 'btn--primary'}`}
        style={{ marginTop: 'var(--step-3)' }}
        disabled={running}
        onClick={() => call(() => api.drone.align(), { refresh: false })}
      >
        {running ? (job.message || 'Aligning…') : block ? 'Align again' : 'Align photos'}
      </button>
    </section>
  )
}
