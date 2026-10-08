import { useEffect } from 'react'
import { useStore, stepsFor } from './state/store'
import TitleBar from './components/TitleBar'
import StepRail from './components/StepRail'
import ImageViewer from './components/ImageViewer'
import StereoView from './components/StereoView'
import OutputViewer, { MosaicPreviewStage } from './components/OutputViewer'
import Inspector from './components/Inspector'
import JobBar from './components/JobBar'
import CommandPalette from './components/CommandPalette'
import Welcome from './components/Welcome'
import EngineBanner from './components/EngineBanner'
import StorageBanner from './components/StorageBanner'
import Fallback from './components/Fallback'
import Intro from './components/Intro'
import PanelToggle from './components/PanelToggle'
import Settings from './components/Settings'

export default function App() {
  const project = useStore((s) => s.project)
  const inspectorOpen = useStore((s) => s.inspectorOpen)
  const setPaletteOpen = useStore((s) => s.setPaletteOpen)
  const toggleInspector = useStore((s) => s.toggleInspector)
  const railOpen = useStore((s) => s.railOpen)
  const toggleRail = useStore((s) => s.toggleRail)
  const setStep = useStore((s) => s.setStep)
  const introOpen = useStore((s) => s.introOpen)
  const activeStep = useStore((s) => s.activeStep)
  const stereoPair = useStore((s) => s.stereoPair)
  const outputView = useStore((s) => s.outputView)
  const mosaicPreview = useStore((s) => s.mosaicPreview)
  const showPair = activeStep === 'dem' && stereoPair

  // What the canvas shows: an output opened from its step, the mosaic preview
  // while mosaicking, the stereo pair in Stereo DEM, otherwise the photos.
  let canvas = <ImageViewer />
  if (outputView && outputView.step === activeStep) {
    canvas = <OutputViewer output={outputView} />
  } else if (activeStep === 'mosaic' && mosaicPreview) {
    canvas = <MosaicPreviewStage />
  } else if (showPair) {
    canvas = <StereoView pair={stereoPair} />
  }

  // Global keyboard. Every one of these exists because the equivalent on a
  // legacy workstation costs a trip through a menu.
  useEffect(() => {
    function onKeyDown(event) {
      const target = event.target
      const typing =
        target instanceof HTMLElement &&
        (target.tagName === 'INPUT' ||
          target.tagName === 'TEXTAREA' ||
          target.isContentEditable)

      const mod = event.ctrlKey || event.metaKey

      if (mod && event.key.toLowerCase() === 'k') {
        event.preventDefault()
        setPaletteOpen(true)
        return
      }

      if (typing) return

      if (mod && event.key === '\\') {
        event.preventDefault()
        toggleInspector()
        return
      }

      if (mod && event.key.toLowerCase() === 'b') {
        event.preventDefault()
        toggleRail()
        return
      }

      // Alt+1..9 jumps to a step, in the order the rail lists them.
      if (event.altKey && /^[1-9]$/.test(event.key)) {
        event.preventDefault()
        const step = stepsFor(useStore.getState().project).all[Number(event.key) - 1]
        if (step) setStep(step.id)
      }
    }

    window.addEventListener('keydown', onKeyDown)
    return () => window.removeEventListener('keydown', onKeyDown)
  }, [setPaletteOpen, toggleInspector, toggleRail, setStep])

  return (
    <div className="app">
      <TitleBar />
      <EngineBanner />
      <StorageBanner />

      {project ? (
        <div className={`app__body ${inspectorOpen ? '' : 'app__body--wide'} ${railOpen ? '' : 'app__body--norail'}`}>
          <Fallback area="The step rail"><StepRail /></Fallback>
          <Fallback area="The image canvas">
            {canvas}
          </Fallback>
          {inspectorOpen && <Fallback area="The inspector"><Inspector /></Fallback>}

          {/* A hidden panel leaves its toggle at the same edge of the canvas. */}
          {!railOpen && (
            <PanelToggle side="left" open={false} onToggle={toggleRail}
                         label="steps" shortcut="Ctrl B" floating />
          )}
          {!inspectorOpen && (
            <PanelToggle side="right" open={false} onToggle={toggleInspector}
                         label="inspector" shortcut="Ctrl \" floating />
          )}
        </div>
      ) : (
        <Fallback area="The start screen"><Welcome /></Fallback>
      )}

      <JobBar />
      <CommandPalette />
      <Fallback area="Settings"><Settings /></Fallback>
      {introOpen && <Fallback area="The introduction"><Intro /></Fallback>}
    </div>
  )
}
