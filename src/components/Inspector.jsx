import { useStore, stepsFor, stepStatus } from '../state/store'
import ProjectPanel from './panels/ProjectPanel'
import CameraPanel from './panels/CameraPanel'
import ImagesPanel from './panels/ImagesPanel'
import PointsPanel from './panels/PointsPanel'
import ModelPanel from './panels/ModelPanel'
import OrthoPanel from './panels/OrthoPanel'
import MosaicPanel from './panels/MosaicPanel'
import DemPanel from './panels/DemPanel'
import TerrainPanel from './panels/TerrainPanel'
import LidarPanel from './panels/LidarPanel'
import ReportsPanel from './panels/ReportsPanel'
import ScenePanel from './panels/ScenePanel'
import Fallback from './Fallback'
import PanelToggle from './PanelToggle'

const PANELS = {
  project: ProjectPanel,
  camera: CameraPanel,
  images: ImagesPanel,
  points: PointsPanel,
  model: ModelPanel,
  ortho: OrthoPanel,
  mosaic: MosaicPanel,
  dem: DemPanel,
  terrain: TerrainPanel,
  lidar: LidarPanel,
  reports: ReportsPanel,
  scene: ScenePanel,
}

// Why a step that was done is now out of date, in the words of each step.
const STALE_REASON = {
  model: 'The control, the measurements or the camera have changed since this solution '
    + 'was computed. Compute it again; everything made from it follows.',
  ortho: 'Made from an earlier solution. Generate the orthophotos again.',
  mosaic: 'Made from orthophotos that have changed since. Generate the mosaic again.',
  dem: 'Made from an earlier solution. Extract the elevation again.',
}

/**
 * The contextual panel. Exactly one step's controls at a time, docked rather
 * than floating — so nothing ever covers the imagery you are measuring on.
 */
export default function Inspector() {
  const toggleInspector = useStore((s) => s.toggleInspector)
  const activeStep = useStore((s) => s.activeStep)
  const project = useStore((s) => s.project)
  // Named as the project's workflow names it ("Photos", not "Images", for a drone block).
  const steps = stepsFor(project).all
  const step = steps.find((s) => s.id === activeStep) || steps[0]
  const Panel = PANELS[activeStep] || ProjectPanel
  const stale = useStore((s) => stepStatus(s, activeStep) === 'stale')

  return (
    <aside className="inspector">
      <header className="inspector__header">
        <div className="inspector__title">
          <h2>{step.name}</h2>
          <span className="inspector__detail">{step.detail}</span>
          <PanelToggle side="right" open onToggle={toggleInspector} label="inspector" shortcut="Ctrl \" />
        </div>
        <div className="inspector__blurb">{step.blurb}</div>
        {stale && (
          <div className="readiness" style={{ marginTop: 'var(--step-3)', borderLeftColor: 'var(--signal-warn)' }}>
            <div className="readiness__item readiness__item--warning">
              <span className="dot" />
              <span>{STALE_REASON[activeStep] || 'Out of date: made before the latest changes. Run it again.'}</span>
            </div>
          </div>
        )}
      </header>

      <div className="inspector__body">
        <Fallback area={`The ${step.name} panel`} resetKey={activeStep}>
          <Panel />
        </Fallback>
      </div>
    </aside>
  )
}
