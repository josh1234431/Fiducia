import { useEffect, useState } from 'react'
import { useStore } from '../../state/store'
import { api } from '../../lib/api'
import InfoTip from '../InfoTip'
import OutputRows from '../OutputRows'
import { useReveal } from '../../lib/reveal'

// Mirrors engine/fiducia/export.py. The first compression listed is the default.
const FORMATS = [
  { id: 'GTiff', label: 'GeoTIFF (.tif)', compressions: ['DEFLATE', 'LZW', 'JPEG', 'NONE'] },
  { id: 'COG', label: 'Cloud-Optimised GeoTIFF (.tif)', compressions: ['DEFLATE', 'LZW', 'JPEG', 'NONE'] },
  { id: 'PCIDSK', label: 'PCIDSK (.pix)', compressions: [] },
  { id: 'HFA', label: 'ERDAS Imagine (.img)', compressions: [] },
  { id: 'JP2OpenJPEG', label: 'JPEG 2000, lossless (.jp2)', compressions: [] },
]

const COMPRESSION_LABELS = {
  DEFLATE: 'Deflate (lossless)',
  LZW: 'LZW (lossless)',
  JPEG: 'JPEG (lossy, smallest)',
  NONE: 'None',
  LOSSLESS: 'Lossless',
}

export default function OrthoPanel() {
  const project = useStore((s) => s.project)
  const call = useStore((s) => s.call)
  const toast = useStore((s) => s.toast)

  const [selected, setSelected] = useState([])
  const [resampling, setResampling] = useState('bilinear')
  const [background, setBackground] = useState('0')
  const [reserveBackground, setReserveBackground] = useState(true)
  const [format, setFormat] = useState('GTiff')
  const [compression, setCompression] = useState('DEFLATE')
  const [quality, setQuality] = useState(90)
  const formatInfo = FORMATS.find((f) => f.id === format) || FORMATS[0]
  const lossy = compression === 'JPEG' && formatInfo.compressions.includes('JPEG')
  const [footprint, setFootprint] = useState(null)
  const [footprintError, setFootprintError] = useState(null)

  const images = (project?.images || []).filter((i) => i.online && i.exterior)
  const projection = project?.projection || {}
  const dem = project?.dem || {}
  const orthos = project?.orthos || []
  const outputView = useStore((s) => s.outputView)
  const showOutput = useStore((s) => s.showOutput)
  const popOutOutput = useStore((s) => s.popOutOutput)
  const generatedRef = useReveal('ortho')

  // On a surface with trees and buildings in it (a drone block's own), leave
  // out the ground each photo cannot see; on a bare-earth DEM nothing hides.
  const droneSurface = !!dem.referencePath && [project?.droneBlock?.dense?.path,
    project?.droneBlock?.surface?.path].includes(dem.referencePath)
  const [hideUnseen, setHideUnseen] = useState(droneSurface)
  useEffect(() => { setHideUnseen(droneSurface) }, [droneSurface])

  useEffect(() => {
    setSelected(images.map((i) => i.id))
    // Only re-seed when the set of solvable images actually changes.
  }, [images.map((i) => i.id).join(',')])

  useEffect(() => {
    if (!selected.length) { setFootprint(null); setFootprintError(null); return }
    api.ortho
      .footprint({ imageId: selected[0] })
      .then((result) => { setFootprint(result); setFootprintError(null) })
      .catch((error) => { setFootprint(null); setFootprintError(error.message || null) })
  }, [selected.join(','), projection.pixelSpacingX, projection.output, dem.referencePath])

  function toggle(id) {
    setSelected((current) =>
      current.includes(id) ? current.filter((x) => x !== id) : [...current, id])
  }

  if (!project?.model) {
    return (
      <div className="empty-note">
        A solved sensor model is required.
      </div>
    )
  }

  return (
    <>
      <section className="section">
        <div className="section__head">
          <span className="section__title">Images to process</span>
          <span className="section__rule" />
          <button
            className="btn btn--ghost btn--sm"
            onClick={() => setSelected(
              selected.length === images.length ? [] : images.map((i) => i.id))}
          >
            {selected.length === images.length ? 'None' : 'All'}
          </button>
        </div>

        {images.length === 0 ? (
          <div className="empty-note">No images have a solved exterior orientation.</div>
        ) : (
          <div className="rows">
            {images.map((image) => {
              const done = orthos.find((o) => o.imageId === image.id)
              return (
                <label
                  key={image.id}
                  className={`row ${selected.includes(image.id) ? 'row--active' : ''}`}
                  style={{ cursor: 'pointer' }}
                >
                  <input
                    type="checkbox"
                    checked={selected.includes(image.id)}
                    onChange={() => toggle(image.id)}
                    style={{ width: 14, flex: 'none' }}
                  />
                  <div className="row__main">
                    <div className="row__name">
                      {image.name}
                      {done && <span className="chip chip--good">done</span>}
                    </div>
                    {done && (
                      <div className="row__meta truncate">
                        {done.width} × {done.height},
                        {' '}{(done.validFraction * 100).toFixed(0)}% covered
                      </div>
                    )}
                  </div>
                </label>
              )
            })}
          </div>
        )}
      </section>

      <section className="section">
        <div className="section__head"><span className="section__title">Output</span>
          <span className="section__rule" /></div>

        <div className="grid-2">
          <div className="field">
            <label className="field__label">Pixel X</label>
            <input
              type="number" step="0.01" className="numeric"
              value={projection.pixelSpacingX ?? 0.5}
              onChange={(event) => call(() => api.project.patch({
                projection: { ...projection, pixelSpacingX: Number(event.target.value) },
              }), { refresh: false })}
            />
          </div>
          <div className="field">
            <label className="field__label">Pixel Y</label>
            <input
              type="number" step="0.01" className="numeric"
              value={projection.pixelSpacingY ?? 0.5}
              onChange={(event) => call(() => api.project.patch({
                projection: { ...projection, pixelSpacingY: Number(event.target.value) },
              }), { refresh: false })}
            />
          </div>
        </div>

        <div className="field">
          <label className="field__label">
            <span>
              Resampling
              <InfoTip>
                <p>Nearest neighbour preserves original values; use it for
                classified or index data.</p>
                <p>Bilinear suits most photography. Cubic is sharper but slower,
                and may overshoot at hard edges.</p>
              </InfoTip>
            </span>
          </label>
          <select value={resampling} onChange={(event) => setResampling(event.target.value)}>
            <option value="nearest">Nearest neighbour</option>
            <option value="bilinear">Bilinear</option>
            <option value="cubic">Cubic</option>
          </select>
        </div>

        <details className="advanced">
          <summary>Options</summary>
          <div className="advanced__body">
            <div className="field">
              <label className="field__label" htmlFor="ortho-format">
                <span>
                  Format
                  <InfoTip>
                    <p>Cloud-Optimised GeoTIFF suits web GIS and sharing. PCIDSK
                    (.pix) and ERDAS Imagine suit workflows built on those formats;
                    JPEG 2000 is common for distributed imagery.</p>
                    <p>Every format keeps the coordinate system and overviews.</p>
                  </InfoTip>
                </span>
              </label>
              <select
                id="ortho-format"
                value={format}
                onChange={(event) => {
                  const next = FORMATS.find((f) => f.id === event.target.value)
                  setFormat(next.id)
                  if (!next.compressions.includes(compression)) {
                    setCompression(next.compressions[0] || '')
                  }
                }}
              >
                {FORMATS.map((entry) => (
                  <option key={entry.id} value={entry.id}>{entry.label}</option>
                ))}
              </select>
            </div>

            {formatInfo.compressions.length > 0 && (
              <div className="field">
                <label className="field__label" htmlFor="ortho-compression">
                  <span>
                    Compression
                    <InfoTip>
                      JPEG files are several times smaller but lossy. The
                      background is then stored as a transparency mask, because a
                      NoData value does not survive lossy compression.
                    </InfoTip>
                  </span>
                </label>
                <select
                  id="ortho-compression"
                  value={compression}
                  onChange={(event) => setCompression(event.target.value)}
                >
                  {formatInfo.compressions.map((id) => (
                    <option key={id} value={id}>{COMPRESSION_LABELS[id]}</option>
                  ))}
                </select>
              </div>
            )}

            {lossy && (
              <div className="field">
                <label className="field__label" htmlFor="ortho-quality">
                  Quality <span className="numeric dim">{quality}</span>
                </label>
                <input
                  id="ortho-quality"
                  type="range" min="50" max="100" step="5"
                  value={quality}
                  onChange={(event) => setQuality(Number(event.target.value))}
                />
              </div>
            )}

            <div className="field">
              <label className="field__label" htmlFor="ortho-background">
                <span>
                  Background value
                  <InfoTip>
                    Written to cells outside the photograph and recorded as the
                    file's NoData value. Values outside the image data type's
                    range are truncated to it.
                  </InfoTip>
                </span>
              </label>
              <input
                id="ortho-background"
                type="number"
                step="1"
                className="numeric"
                value={background}
                onChange={(event) => setBackground(event.target.value)}
              />
            </div>

            <label className="field field--row">
              <span className="field__label">
                <span>
                  Keep image values off the background
                  <InfoTip>
                    <p>Image pixels equal to the background value are moved one
                    step away from it, so the background marks only areas outside
                    the photograph. Without this, dark water and shadow can
                    display as holes in GIS software.</p>
                    <p>Turn off to write image values unchanged.</p>
                  </InfoTip>
                </span>
              </span>
              <input
                type="checkbox"
                checked={reserveBackground}
                onChange={(event) => setReserveBackground(event.target.checked)}
              />
            </label>
          </div>
        </details>

        {footprint && (
          <div className="measures">
            <div className="measure">
              <div className="measure__name">Output size</div>
              <div className="measure__value" style={{ fontSize: 13 }}>
                {footprint.width} × {footprint.height}
              </div>
            </div>
            <div className="measure">
              <div className="measure__name">Approx. each</div>
              <div className="measure__value" style={{ fontSize: 13 }}>
                {(footprint.estimatedBytes / 1e6).toFixed(0)} MB
              </div>
            </div>
          </div>
        )}

        {!dem.referencePath && (
          <div className="field__hint" style={{ color: 'var(--signal-warn)', marginTop: 6 }}>
            No DEM is set. Terrain displacement will not be corrected.
          </div>
        )}
        {dem.referencePath && dem.referencePath === project?.droneBlock?.dense?.path && (
          <div className="field__hint" style={{ marginTop: 6 }}>
            Terrain from the drone block's dense surface
            ({project.droneBlock.dense.resolution} m cells, {project.droneBlock.dense.pairs} pairs).
            {project.droneBlock.dense.stale && ' It predates the latest solve: rebuild it on the Stereo DEM step.'}
          </div>
        )}
        {dem.referencePath && dem.referencePath === project?.droneBlock?.surface?.path && (
          <div className="field__hint" style={{ marginTop: 6 }}>
            Terrain from the drone block's own tie points
            ({project.droneBlock.surface.points.toLocaleString()} points,
            {' '}{project.droneBlock.surface.cellM} m cells), rebuilt with every solve.
          </div>
        )}

        {dem.referencePath && (
          <label className="field field--row" style={{ marginTop: 'var(--step-2)' }}>
            <span className="field__label">
              <span>
                Leave out ground each photo can't see
                <InfoTip>
                  Ground hidden from a photo behind a tree or a building is left empty in
                  that photo's ortho, and the mosaic takes it from a photo that does see
                  it. Without this the hidden ground repeats the tree or wall in front of
                  it. Needs a surface with trees and buildings in it, such as a drone
                  block's own; on for those, off for a bare-earth DEM.
                </InfoTip>
              </span>
            </span>
            <input type="checkbox" checked={hideUnseen}
              onChange={(event) => setHideUnseen(event.target.checked)} />
          </label>
        )}

        {footprintError && (
          <div className="readiness readiness--blocked" style={{ marginTop: 'var(--step-2)' }}>
            <div className="readiness__item readiness__item--blocker">
              <span className="dot" /><span>{footprintError}</span>
            </div>
          </div>
        )}
      </section>

      <section className="section">
        <button
          className="btn btn--primary btn--block"
          disabled={!selected.length}
          onClick={() => {
            const nodata = background.trim() === '' ? 0 : Number(background)
            if (!Number.isFinite(nodata)) {
              toast('The background value must be a number', 'warn')
              return
            }
            call(() => api.ortho.generate({
              imageIds: selected,
              resampling,
              demPath: dem.referencePath,
              occlusion: hideUnseen,
              nodata,
              reserveNodata: reserveBackground,
              format,
              compression,
              jpegQuality: quality,
              author: useStore.getState().author,
            }))
          }}
        >
          Generate {selected.length} ortho{selected.length === 1 ? '' : 's'}
        </button>
      </section>

      {orthos.length > 0 && (
        <section className="section" ref={generatedRef}>
          <div className="section__head"><span className="section__title">Generated</span>
          <span className="section__rule" /></div>
          <OutputRows
            items={orthos.map((ortho) => ({
              key: ortho.id, path: ortho.path, name: ortho.name,
              meta: ortho.width ? `${ortho.width} × ${ortho.height}, ${ortho.generatedAt || ''}` : ortho.path,
            }))}
            step="ortho"
            current={outputView?.step === 'ortho' ? outputView.path : null}
            onView={showOutput}
            onPopOut={popOutOutput}
          />
        </section>
      )}
    </>
  )
}
