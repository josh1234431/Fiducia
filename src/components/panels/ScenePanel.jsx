import { useState } from 'react'
import { useStore } from '../../state/store'
import { api } from '../../lib/api'
import InfoTip from '../InfoTip'

/**
 * Photos without positions, to a 3D model.
 *
 * Aligning places every photo from the photos alone, and learns each
 * camera's lens. The model then has no size: two points a known distance
 * apart give it one (or, roughly, the phones' own GPS). The dense cloud is a
 * point for nearly every pixel, coloured from the photos, and opens in the
 * point cloud tools like any LiDAR cloud.
 */
export default function ScenePanel() {
  const project = useStore((s) => s.project)
  const call = useStore((s) => s.call)
  const toast = useStore((s) => s.toast)
  const jobs = useStore((s) => s.jobs)
  const picks = useStore((s) => s.scenePicks)
  const clearPicks = useStore((s) => s.clearScenePicks)
  const picking = useStore((s) => s.scenePicking)
  const setPicking = useStore((s) => s.setScenePicking)
  const setLayer = useStore((s) => s.setSceneLayer)
  const setLidarCloud = useStore((s) => s.setLidarCloud)
  const setStep = useStore((s) => s.setStep)

  const scene = project?.scene
  const online = (project?.images || []).filter((i) => i.online)
  const running = (kind) => jobs.find((j) => j.kind === kind && (j.status === 'running' || j.status === 'queued'))
  const aligning = running('scene')
  const densifying = running('scene_dense')
  const [detail, setDetail] = useState('medium')
  const [distance, setDistance] = useState('')

  const transform = scene?.transform
  const units = transform?.scaleSource === 'none' ? 'units' : 'm'
  const picked = picks.length === 2
    ? Math.hypot(picks[0].x - picks[1].x, picks[0].y - picks[1].y, picks[0].z - picks[1].z)
    : null

  async function setScale() {
    const metres = Number(distance)
    if (!(metres > 0) || picks.length !== 2) return
    const result = await call(() => api.scene.scale(
      [picks[0].x, picks[0].y, picks[0].z], [picks[1].x, picks[1].y, picks[1].z], metres))
    if (result) {
      toast(`Scaled: the picked points are now ${metres} m apart`, 'good')
      clearPicks()
      setDistance('')
    }
  }

  async function exportCloud(which) {
    if (!window.fiducia?.isDesktop) {
      toast('Clouds can only be saved in the desktop app', 'warn')
      return
    }
    const name = `${project?.name || 'Model'} ${which}`
    const target = await window.fiducia.dialog.saveFile({
      title: `Save the ${which} cloud`,
      defaultPath: `${name}.ply`,
      filters: [
        { name: 'PLY (meshing and 3D software)', extensions: ['ply'] },
        { name: 'LAZ (point cloud software)', extensions: ['laz'] },
        { name: 'LAS', extensions: ['las'] },
      ],
    })
    if (!target) return
    const result = await call(() => api.scene.export(which, target), { refresh: false })
    if (result) toast(`Saved ${result.points.toLocaleString()} points to ${result.path}`, 'good', 9000)
  }

  function openInTools() {
    if (!scene?.dense?.lazPath) return
    api.scene.view().then((view) => {
      if (!view.lazPath) return
      setLidarCloud(view.lazPath)
      setStep('lidar')
    }).catch((e) => toast(e.message, 'bad'))
  }

  return (
    <>
      <section className="section">
        <div className="section__head">
          <span className="section__title">Alignment</span>
          <span className="section__rule" />
          <InfoTip>
            Finds the same features in overlapping photos and works out, from the photos
            alone, where each was taken and the lens of each camera. Photos from different
            phones, or from one phone's wide and zoom lenses, are told apart by what each
            photo records about its camera. About a minute for 30 photos.
          </InfoTip>
        </div>

        {!scene && (
          <div className="empty-note">
            {online.length < 3
              ? 'Add at least three photos on the Photos step first.'
              : `${online.length} photos ready. Each part of the subject should be in at least three of them.`}
          </div>
        )}

        {scene && (
          <>
            <div className="measures">
              <div className="measure">
                <div className="measure__name">Photos placed</div>
                <div className={`measure__value ${scene.unplaced?.length ? 'measure__value--warn' : ''}`}>
                  {scene.placed} of {scene.photos}
                </div>
              </div>
              <div className="measure">
                <div className="measure__name">Tie points</div>
                <div className="measure__value">{scene.points?.toLocaleString()}</div>
              </div>
              <div className="measure">
                <div className="measure__name">Reprojection</div>
                <div className="measure__value">{scene.reprojectionRmsPx?.toFixed(2)} px</div>
              </div>
              <div className="measure">
                <div className="measure__name">Seen in</div>
                <div className="measure__value">{scene.meanTrackLength?.toFixed(1)} photos</div>
              </div>
            </div>
            {scene.unplaced?.length > 0 && (
              <div className="field__hint" style={{ marginTop: 'var(--step-2)' }}>
                Not placed: {scene.unplaced.slice(0, 8).join(', ')}
                {scene.unplaced.length > 8 ? ` and ${scene.unplaced.length - 8} more` : ''}. They share too
                little with the rest; photos taken between them and their neighbours would join them in.
              </div>
            )}
            <div className="rows" style={{ marginTop: 'var(--step-3)' }}>
              {scene.cameras.map((camera) => (
                <div key={camera.name} className="row row--static">
                  <div className="row__main">
                    <div className="row__name truncate">{camera.name}</div>
                    <div className="row__meta truncate">
                      {camera.placed} of {camera.photos} photos · focal length{' '}
                      {Math.round(camera.focalPx).toLocaleString()} px
                      {camera.focalKnown
                        ? ` (photos say ${Math.round(camera.focalPrior).toLocaleString()})`
                        : ' (found from the photos)'}
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </>
        )}

        <button
          className={`btn btn--block ${scene ? '' : 'btn--primary'}`}
          style={{ marginTop: 'var(--step-3)' }}
          disabled={!!aligning || online.length < 3}
          onClick={() => {
            if (scene?.dense && !window.confirm('Aligning again replaces the model, its scale and its '
              + 'dense cloud. Align again?')) return
            call(() => api.scene.align(), { refresh: false })
          }}
        >
          {aligning ? (aligning.message || 'Aligning…') : scene ? 'Align again' : 'Align photos'}
        </button>
      </section>

      {scene && (
        <section className="section">
          <div className="section__head">
            <span className="section__title">Scale</span>
            <span className="section__rule" />
            <InfoTip>
              Photos alone show shape, not size. Pick two points on the model whose
              distance you know (the ends of a tape or a scale bar, the width of a door)
              and enter it. The further apart the points, the better the scale. Pick
              points on the dense cloud for the most precise ends.
            </InfoTip>
          </div>

          <div className="readiness" style={{
            borderLeftColor: transform.scaleSource === 'distance' ? 'var(--signal-good)' : 'var(--signal-warn)',
          }}>
            <div className={`readiness__item ${transform.scaleSource === 'distance' ? '' : 'readiness__item--warning'}`}>
              <span className="dot" />
              <span>
                {transform.scaleSource === 'distance' && `In metres, from a measured ${transform.scaleDistanceM} m.`}
                {transform.scaleSource === 'gps' && `In metres, roughly, from the photos' GPS `
                  + `(${transform.gpsPhotos} photos, ±${transform.gpsResidualM} m fit) in ${transform.crs}. `
                  + 'A measured distance is far better.'}
                {transform.scaleSource === 'none' && 'No scale yet: distances are in the model\'s own units.'}
              </span>
            </div>
          </div>

          <div className="field" style={{ marginTop: 'var(--step-3)' }}>
            <button className={`btn btn--block ${picking ? 'btn--on' : ''}`} aria-pressed={picking}
                    onClick={() => setPicking(!picking)}>
              {picking ? 'Picking: click two points on the model' : 'Pick two points'}
            </button>
          </div>
          {picks.length > 0 && (
            <div className="field__hint">
              {picks.length === 1 ? 'One point picked: click the second.' : (
                <>Picked points are <span className="numeric">{picked.toFixed(units === 'm' ? 3 : 4)} {units}</span> apart.</>
              )}{' '}
              <button className="btn btn--ghost btn--sm" onClick={clearPicks}>Clear</button>
            </div>
          )}
          <div className="grid-2" style={{ marginTop: 'var(--step-2)' }}>
            <div className="field">
              <label className="field__label">Real distance (m)</label>
              <input type="number" min="0" step="0.001" className="numeric" value={distance}
                     placeholder="e.g. 1.000" onChange={(event) => setDistance(event.target.value)}
                     onKeyDown={(event) => { if (event.key === 'Enter') setScale() }} />
            </div>
            <div className="field">
              <label className="field__label">&nbsp;</label>
              <button className="btn btn--block" disabled={picks.length !== 2 || !(Number(distance) > 0)}
                      onClick={setScale}>
                Set scale
              </button>
            </div>
          </div>
        </section>
      )}

      {scene && (
        <section className="section">
          <div className="section__head">
            <span className="section__title">Dense cloud</span>
            <span className="section__rule" />
            <InfoTip>
              A point for nearly every pixel, coloured from the photos. Each photo is matched
              pixel by pixel with neighbours that see the same place from a few steps away,
              and a point is kept only where another photo puts the surface at the same
              distance: moving leaves, reflections and sky mostly drop out.
            </InfoTip>
          </div>

          {scene.dense && (
            <>
              <div className="measures">
                <div className="measure">
                  <div className="measure__name">Points</div>
                  <div className="measure__value">{scene.dense.points.toLocaleString()}</div>
                </div>
                <div className="measure">
                  <div className="measure__name">Photos used</div>
                  <div className="measure__value">{scene.dense.references} of {scene.dense.photos}</div>
                </div>
              </div>
              <div className="field__hint" style={{ marginTop: 'var(--step-2)' }}>
                {scene.dense.detail} detail, built {scene.dense.builtAt}.
                {scene.dense.pairsFailed?.length > 0 && ` ${scene.dense.pairsFailed.length} pair`
                  + `${scene.dense.pairsFailed.length === 1 ? '' : 's'} could not be matched.`}
              </div>
            </>
          )}

          <div className="field" style={{ marginTop: 'var(--step-3)' }}>
            <label className="field__label">
              <span>
                Detail
                <InfoTip>
                  The size the photos are matched at, as their long side. More detail gives
                  more points and finer edges, and takes longer: high is about twice as slow
                  as medium.
                </InfoTip>
              </span>
            </label>
            <select value={detail} onChange={(event) => setDetail(event.target.value)}>
              <option value="low">Low (1,000 px)</option>
              <option value="medium">Medium (1,600 px)</option>
              <option value="high">High (2,400 px)</option>
            </select>
          </div>
          <button
            className={`btn btn--block ${scene.dense ? '' : 'btn--primary'}`}
            disabled={!!densifying || !!aligning}
            onClick={async () => {
              const result = await call(() => api.scene.dense(detail), { refresh: false })
              if (result) setLayer('dense')
            }}
          >
            {densifying ? (densifying.message || 'Building…') : scene.dense ? 'Build again' : 'Build dense cloud'}
          </button>
          {scene.dense?.lazPath && (
            <button className="btn btn--block" style={{ marginTop: 'var(--step-2)' }} onClick={openInTools}
                    title="Noise removal, ground, heights and classes, as for any LiDAR cloud">
              Open in the point cloud tools
            </button>
          )}
        </section>
      )}

      {scene && (
        <section className="section">
          <div className="section__head">
            <span className="section__title">Export</span>
            <span className="section__rule" />
            <InfoTip>
              PLY opens in meshing and 3D software (MeshLab, Blender, CloudCompare); LAS and
              LAZ in point cloud and GIS software. Both carry each point's colour, in the
              model's scale.
            </InfoTip>
          </div>
          <div className="grid-2">
            <button className="btn btn--block" disabled={!scene.dense} onClick={() => exportCloud('dense')}>
              Save dense cloud…
            </button>
            <button className="btn btn--block" onClick={() => exportCloud('sparse')}>
              Save tie points…
            </button>
          </div>
        </section>
      )}
    </>
  )
}
