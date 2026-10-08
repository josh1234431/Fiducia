import { useState } from 'react'
import { useStore } from '../state/store'
import { api } from '../lib/api'
import ImageViewer, { viewCentres } from './ImageViewer'
import Fallback from './Fallback'

/**
 * A stereo pair, side by side.
 *
 * Each photo gets its own full viewer, so either can be panned, zoomed,
 * turned or stretched on its own, and each carries the usual Pop out button
 * for a second monitor. A side that is popped out gives its space to the
 * other; closing its window brings it back.
 *
 * Line up moves one view to the ground at the centre of the other, through
 * the solved orientation and the terrain height, at the same ground scale.
 * The view last clicked or scrolled in leads; a popped-out photo follows.
 */
export default function StereoView({ pair }) {
  const project = useStore((s) => s.project)
  const poppedOut = useStore((s) => s.poppedOut)
  const setStereoPair = useStore((s) => s.setStereoPair)
  const goTo = useStore((s) => s.goTo)
  const toast = useStore((s) => s.toast)
  const [lead, setLead] = useState('left')
  const [busy, setBusy] = useState(false)

  const nameOf = (id) => project?.images?.find((entry) => entry.id === id)?.name || 'Photo'
  const sides = [
    { key: 'left', id: pair.leftId, label: 'Left' },
    { key: 'right', id: pair.rightId, label: 'Right' },
  ]
  const shown = sides.filter((side) => !poppedOut.includes(side.id))
  const away = sides.filter((side) => poppedOut.includes(side.id))

  // Which view leads: the one last used, unless it is in its own window, in
  // which case the one still here leads and the window follows.
  const source = shown.length === 2
    ? sides.find((side) => side.key === lead)
    : shown[0]
  const target = source && sides.find((side) => side.key !== source.key)

  async function lineUp() {
    const centre = source && viewCentres.get(source.id)
    if (!centre) return
    setBusy(true)
    try {
      const result = await api.images.transfer(source.id, centre.col, centre.row, target.id)
      if (!result.ahead) {
        toast(`The ground at the centre of ${nameOf(source.id)} is not seen by ${nameOf(target.id)}.`, 'warn')
        return
      }
      goTo(target.id, result.col, result.row, centre.scale / (result.scaleRatio || 1))
      if (!result.inside) {
        toast(`That ground lies outside ${nameOf(target.id)}; its view shows where it would be.`, 'warn')
      }
    } catch (error) {
      toast(error.message || 'The views could not be lined up', 'bad')
    } finally {
      setBusy(false)
    }
  }

  return (
    <div className={`stereo ${shown.length === 1 ? 'stereo--single' : ''}`}>
      {shown.map((side) => (
        <div
          className={`stereo__pane ${shown.length === 2 && lead === side.key ? 'stereo__pane--lead' : ''}`}
          key={side.key}
          onPointerDownCapture={() => setLead(side.key)}
          onWheelCapture={() => setLead(side.key)}
        >
          <span className="stereo__side">{side.label}</span>
          <Fallback area={`The ${side.key} photo`}>
            <ImageViewer
              imageId={side.id}
              onImageChange={(id) => setStereoPair({
                ...pair, [side.key === 'left' ? 'leftId' : 'rightId']: id,
              })}
            />
          </Fallback>
        </div>
      ))}

      {shown.length === 0 && (
        <div className="stereo__away">
          <p>Both photos of this pair are in their own windows.</p>
          <p className="dim">Close a window to show that photo here again.</p>
        </div>
      )}

      {source && (
        <div className="stereo__lineup">
          <button
            className="btn btn--sm"
            onClick={lineUp}
            disabled={busy}
            title="Move one view to the ground at the centre of the other, at the same ground scale. The view you last used leads."
          >
            {busy ? 'Lining up…'
              : away.length ? `Line up the ${target.label.toLowerCase()} window to this view`
              : `Line up ${target.label.toLowerCase()} to ${source.label.toLowerCase()}`}
          </button>
        </div>
      )}

      {away.length > 0 && shown.length > 0 && (
        <div className="stereo__note">
          {away.map((side) => `${side.label}: ${nameOf(side.id)}`).join(', ')}
          {' '}is in its own window. Close it to show it here again.
        </div>
      )}
    </div>
  )
}
