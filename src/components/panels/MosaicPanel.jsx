import { useCallback, useEffect, useRef, useState } from 'react'
import { useStore } from '../../state/store'
import { api } from '../../lib/api'
import InfoTip from '../InfoTip'
import OutputRows from '../OutputRows'
import { useReveal } from '../../lib/reveal'

/**
 * Mosaicking, with a live preview.
 *
 * Tuning a mosaic means changing one parameter at a time and comparing
 * results — which on a legacy workstation means regenerating a preview through a wizard
 * for every change. Here the preview is debounced and automatic, and the seam
 * map can be toggled on top of it, so the effect of a setting is visible in
 * about a second.
 */
export default function MosaicPanel() {
  const project = useStore((s) => s.project)
  const call = useStore((s) => s.call)

  // A drone block's orthos sit on a surface of sparse tie points, so trees
  // lean differently from photo to photo: seams that follow where the photos
  // agree, with a narrow blend, keep them from ghosting.
  const isDrone = !!project?.droneBlock
  const [settings, setSettings] = useState({
    colorBalance: 'linear',
    normalization: 'none',
    cutlineMethod: isDrone ? 'min_difference' : 'distance',
    blendWidth: isDrone ? 16 : 40,
    sortMethod: 'nearest_center',
    resampling: 'nearest',
    startingImage: '',
  })
  // The preview is drawn on the canvas, where it has room; the panel holds
  // the settings that change it.
  const preview = useStore((s) => s.mosaicPreview)
  const setPreview = useStore((s) => s.setMosaicPreview)
  const showSeams = useStore((s) => s.mosaicSeams)
  const setShowSeams = useStore((s) => s.setMosaicSeams)
  const outputView = useStore((s) => s.outputView)
  const showOutput = useStore((s) => s.showOutput)
  const closeOutput = useStore((s) => s.closeOutput)
  const popOutOutput = useStore((s) => s.popOutOutput)
  const generatedRef = useReveal('mosaic')
  const [busy, setBusy] = useState(false)
  const timer = useRef(null)

  const orthos = project?.orthos || []
  const mosaics = project?.mosaics || []
  const viewingOutput = outputView?.step === 'mosaic'

  const refresh = useCallback(async () => {
    if (orthos.length < 2) { setPreview(null); return }
    setBusy(true)
    try {
      setPreview(await api.mosaic.preview({
        inputs: orthos.map((o) => o.path),
        ...settings,
        startingImage: settings.startingImage || undefined,
      }))
    } catch {
      setPreview(null)
    } finally {
      setBusy(false)
    }
  }, [orthos.map((o) => o.path).join(','), settings])

  useEffect(() => {
    if (timer.current) clearTimeout(timer.current)
    timer.current = setTimeout(refresh, 350)
    return () => clearTimeout(timer.current)
  }, [refresh])

  function update(patch) {
    setSettings((current) => ({ ...current, ...patch }))
    // A changed setting is judged on the preview, so bring it back.
    if (viewingOutput) closeOutput()
  }

  if (orthos.length < 2) {
    return (
      <div className="empty-note">
        At least two orthoimages are required.
      </div>
    )
  }

  return (
    <>
      <section className="section">
        <div className="section__head">
          <span className="section__title">Preview</span>
          <span className="section__rule" />
          <button
            className={`btn btn--ghost btn--sm ${showSeams ? 'btn--primary' : ''}`}
            onClick={() => { setShowSeams(!showSeams); if (viewingOutput) closeOutput() }}
          >
            Seams
          </button>
        </div>

        {preview ? (
          <>
            <div className="field__hint">
              {viewingOutput
                ? 'A generated mosaic is on the canvas.'
                : 'Shown on the canvas, and updated as you change the settings below.'}
              {busy && <span className="chip chip--accent" style={{ marginLeft: 6 }}>updating</span>}
            </div>
            {viewingOutput && (
              <button className="btn btn--block btn--sm" style={{ marginTop: 6 }}
                      onClick={closeOutput}>
                Show the live preview
              </button>
            )}
            <div className="field__hint" style={{ marginTop: 6 }}>
              Full output <span className="numeric">
                {preview.fullWidth} × {preview.fullHeight}
              </span> at {preview.pixelSize.toFixed(2)} m.
              {' '}Order: {preview.order.join(', ')}.
            </div>
          </>
        ) : (
          <div className="empty-note">{busy ? 'Rendering preview…' : 'No preview'}</div>
        )}
      </section>

      <section className="section">
        <div className="section__head"><span className="section__title">Tone</span>
          <span className="section__rule" /></div>

        <div className="field">
          <label className="field__label">
            <span>
              Normalisation
              <InfoTip>
                <p>Evens out brightness within each image before images are balanced
                against one another (dodging).</p>
                <p>Across image fits a smooth trend surface, for the darkening from centre
                to edge. Hotspot follows broader patches too, such as the bright area
                opposite the sun.</p>
              </InfoTip>
            </span>
          </label>
          <select
            value={settings.normalization}
            onChange={(event) => update({ normalization: event.target.value })}
          >
            <option value="none">None</option>
            <option value="across_image">Across image (trend surface)</option>
            <option value="hotspot">Hotspot (broad dodging)</option>
          </select>
        </div>

        <div className="field">
          <label className="field__label">
            <span>
              Colour balance
              <InfoTip>
                <p>Makes overlapping images agree. Offset matches their levels; gain and
                offset also their contrast (mean and spread); histogram matching maps each
                image's whole distribution onto the images placed before it.</p>
              </InfoTip>
            </span>
          </label>
          <select
            value={settings.colorBalance}
            onChange={(event) => update({ colorBalance: event.target.value })}
          >
            <option value="none">None</option>
            <option value="linear">Offset</option>
            <option value="histogram">Gain and offset</option>
            <option value="histogram_match">Histogram matching</option>
          </select>
        </div>

        <div className="field">
          <label className="field__label">
            <span>
              Reference image
              <InfoTip>
                Colour balancing is anchored to this image. Avoid frames with
                saturated water or cloud.
              </InfoTip>
            </span>
          </label>
          <select
            value={settings.startingImage}
            onChange={(event) => update({ startingImage: event.target.value })}
          >
            <option value="">Automatic (most central)</option>
            {orthos.map((ortho) => (
              <option key={ortho.id} value={ortho.name}>{ortho.name}</option>
            ))}
          </select>
        </div>
      </section>

      <section className="section">
        <div className="section__head"><span className="section__title">Cutlines</span>
          <span className="section__rule" /></div>

        <div className="field">
          <label className="field__label">
            <span>
              Method
              <InfoTip>
                Least difference routes seams where overlapping images look alike, along
                roads and open ground, and around buildings and trees, whose lean differs
                between photos.
              </InfoTip>
            </span>
          </label>
          <select
            value={settings.cutlineMethod}
            onChange={(event) => update({ cutlineMethod: event.target.value })}
          >
            <option value="min_difference">Least difference between images</option>
            <option value="distance">Distance from image edge</option>
            <option value="nearest_center">Nearest centre</option>
            <option value="order">Image order</option>
          </select>
        </div>

        <details className="advanced">
          <summary>Options</summary>
          <div className="advanced__body">

        <div className="field">
          <label className="field__label">
            <span>
              Blend width
              <InfoTip>Wider blending softens seams but may blur detail across them.</InfoTip>
            </span>
            <span className="numeric dim">{settings.blendWidth} px</span>
          </label>
          <input
            type="range" min="0" max="200" step="5"
            value={settings.blendWidth}
            onChange={(event) => update({ blendWidth: Number(event.target.value) })}
          />
        </div>

        <div className="field">
          <label className="field__label">Resampling</label>
          <select
            value={settings.resampling}
            onChange={(event) => update({ resampling: event.target.value })}
          >
            <option value="nearest">Nearest</option>
            <option value="bilinear">Bilinear</option>
            <option value="cubic">Cubic</option>
          </select>
        </div>
          </div>
        </details>
      </section>

      <section className="section">
        <button
          className="btn btn--primary btn--block"
          onClick={() => call(() => api.mosaic.generate({
            inputs: orthos.map((o) => o.path),
            ...settings,
            startingImage: settings.startingImage || undefined,
          }))}
        >
          Generate mosaic
        </button>
      </section>

      {mosaics.length > 0 && (
        <section className="section" ref={generatedRef}>
          <div className="section__head"><span className="section__title">Generated</span>
          <span className="section__rule" /></div>
          <OutputRows
            items={mosaics.map((mosaic) => ({
              key: mosaic.id, path: mosaic.path, name: mosaic.name,
              meta: `${mosaic.width} × ${mosaic.height}, ${mosaic.generatedAt}`,
            }))}
            step="mosaic"
            current={viewingOutput ? outputView.path : null}
            onView={showOutput}
            onPopOut={popOutOutput}
          />
        </section>
      )}
    </>
  )
}
