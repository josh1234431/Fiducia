import { useEffect, useState } from 'react'
import { useStore, STEPS } from './state/store'
import ImageViewer from './components/ImageViewer'
import OutputViewer from './components/OutputViewer'
import Toasts from './components/Toasts'
import Fallback from './components/Fallback'

/**
 * A popped-out image, on a monitor of its own.
 *
 * Nothing but the canvas: panels, rail and inspector stay in the main window,
 * so a second screen is all photograph. The strip along the bottom says what
 * a click here will do, because on a screen away from the panels that is the
 * one thing the operator cannot otherwise see.
 */
export default function PopoutApp({ imageId: initial }) {
  const project = useStore((s) => s.project)
  const pickMode = useStore((s) => s.pickMode)
  const activeStep = useStore((s) => s.activeStep)
  const connected = useStore((s) => s.connected)
  const [imageId, setImageId] = useState(initial)

  const image = project?.images?.find((i) => i.id === imageId)
  const step = STEPS.find((s) => s.id === activeStep)

  useEffect(() => {
    if (image) document.title = `${image.name} — Fiducia`
  }, [image?.name])

  function switchTo(nextId) {
    const next = project?.images?.find((i) => i.id === nextId)
    setImageId(nextId)
    window.fiducia?.windows?.retarget(nextId, next?.name)
  }

  // A generated raster rather than a photo: "output:<path>".
  if (imageId?.startsWith('output:')) {
    const path = imageId.slice('output:'.length)
    return (
      <div className="popout">
        <div className="popout__canvas">
          {connected ? (
            <Fallback area="This window">
              <OutputViewer output={{ path, label: path.split(/[\\/]/).pop() }} popout />
            </Fallback>
          ) : <div className="popout__note">Connecting to the engine…</div>}
        </div>
        <footer className="popout__strip">
          <span className="popout__dot" />
          <span>{path}</span>
          <span className="popout__spacer" />
          <button className="btn btn--ghost btn--sm"
                  onClick={() => window.fiducia?.windows?.focusMain()}>
            Main window
          </button>
        </footer>
      </div>
    )
  }

  let body
  if (!connected) {
    body = <div className="popout__note">Connecting to the engine…</div>
  } else if (!project) {
    body = <div className="popout__note">No project is open in the main window.</div>
  } else if (!image) {
    body = (
      <div className="popout__note">
        This image is no longer in the project.
        <button className="btn btn--sm" onClick={() => window.fiducia?.windows?.focusMain()}>
          Back to the main window
        </button>
      </div>
    )
  } else {
    body = (
      <Fallback area="This window">
        <ImageViewer imageId={imageId} onImageChange={switchTo} popout />
      </Fallback>
    )
  }

  return (
    <div className="popout">
      <div className="popout__canvas">{body}</div>
      <footer className={`popout__strip ${pickMode ? 'popout__strip--armed' : ''}`}>
        <span className="popout__dot" />
        {pickMode ? (
          <span><strong>{pickMode.label}</strong> — click here to measure it</span>
        ) : (
          <span>
            Linked to the main window{step ? `, ${step.name} step` : ''}.
            Measurements you start there can be finished here.
          </span>
        )}
        <span className="popout__spacer" />
        <button className="btn btn--ghost btn--sm"
                onClick={() => window.fiducia?.windows?.focusMain()}>
          Main window
        </button>
      </footer>
      <Toasts />
    </div>
  )
}
