import { useEffect, useRef, useState } from 'react'
import { useStore, WORKFLOWS } from '../state/store'

const RECENT_KEY = 'fiducia.recent'

function loadRecent() {
  try {
    return JSON.parse(localStorage.getItem(RECENT_KEY) || '[]')
  } catch {
    return []
  }
}

/** Windows rejects \ / : * ? " < > | in a folder name; so do we, quietly. */
function safeFolderName(name) {
  return name.trim().replace(/[\\/:*?"<>|]/g, '_').replace(/\s+/g, ' ').slice(0, 80)
}

/** "Today, 14:03", "Yesterday, 09:10", or a date for anything older. */
function lastOpened(at) {
  const when = new Date(at)
  const time = when.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
  const startOfToday = new Date()
  startOfToday.setHours(0, 0, 0, 0)
  const days = Math.floor((startOfToday - when) / 86400000) + 1
  if (when >= startOfToday) return `Today, ${time}`
  if (days === 1) return `Yesterday, ${time}`
  return when.toLocaleDateString([], {
    day: 'numeric', month: 'short',
    ...(when.getFullYear() !== startOfToday.getFullYear() ? { year: 'numeric' } : {}),
  })
}

export function rememberProject(directory, name) {
  const recent = loadRecent().filter((entry) => entry.directory !== directory)
  recent.unshift({ directory, name, at: Date.now() })
  localStorage.setItem(RECENT_KEY, JSON.stringify(recent.slice(0, 8)))
}

/** A plain cog, drawn to match the panel toggles. */
function GearIcon() {
  return (
    <svg width="14" height="14" viewBox="0 0 16 16" aria-hidden="true" fill="none"
         stroke="currentColor" strokeWidth="1.3" strokeLinecap="round">
      <circle cx="8" cy="8" r="2.2" />
      <path d="M8 1.8v1.6M8 12.6v1.6M1.8 8h1.6M12.6 8h1.6M3.6 3.6l1.1 1.1M11.3 11.3l1.1 1.1M3.6 12.4l1.1-1.1M11.3 4.7l1.1-1.1" />
      <circle cx="8" cy="8" r="4.6" />
    </svg>
  )
}

export default function Welcome() {
  const createProject = useStore((s) => s.createProject)
  const openProject = useStore((s) => s.openProject)
  const loading = useStore((s) => s.loading)
  const toast = useStore((s) => s.toast)
  const author = useStore((s) => s.author)
  const setSettingsOpen = useStore((s) => s.setSettingsOpen)

  const [recent, setRecent] = useState([])
  const [draft, setDraft] = useState(null)
  const nameRef = useRef(null)

  useEffect(() => {
    const reload = () => setRecent(loadRecent())
    reload()
    window.addEventListener('fiducia:recent', reload)
    return () => window.removeEventListener('fiducia:recent', reload)
  }, [])
  useEffect(() => {
    if (draft) setTimeout(() => nameRef.current?.select(), 20)
  }, [draft?.open])

  const desktop = typeof window !== 'undefined' && window.fiducia?.isDesktop

  // The bundle folder Fiducia will actually create, shown before you commit
  // to it. `window.prompt` is not implemented in Electron — it returns null
  // and the flow dies silently — so naming happens here, in the interface.
  const bundlePath = draft?.folder && draft?.name
    ? `${draft.folder}\\${safeFolderName(draft.name)}.fidu`
    : ''

  function beginCreate() {
    if (!desktop) {
      toast('Projects can only be created in the desktop app', 'warn')
      return
    }
    setDraft({ open: true, name: 'Untitled block', folder: '', workflow: 'drone' })
  }

  async function chooseFolder() {
    const folder = await window.fiducia.dialog.openDirectory({
      title: 'Choose a project location',
    })
    if (folder) setDraft((d) => ({ ...d, folder }))
  }

  async function confirmCreate() {
    if (!draft?.name.trim()) {
      toast('Enter a project name', 'warn')
      nameRef.current?.focus()
      return
    }
    if (!draft.folder) {
      toast('Choose a project location', 'warn')
      return
    }
    if (await createProject(bundlePath, draft.name.trim(), undefined, draft.workflow)) {
      rememberProject(bundlePath, draft.name.trim())
      setRecent(loadRecent())
      setDraft(null)
    }
  }

  async function onOpen() {
    if (!desktop) {
      toast('Projects can only be opened in the desktop app', 'warn')
      return
    }
    const folder = await window.fiducia.dialog.openDirectory({
      title: 'Open a .fidu project bundle',
    })
    if (!folder) return
    if (await openProject(folder)) {
      rememberProject(folder, folder.split(/[\\/]/).pop())
      setRecent(loadRecent())
    }
  }

  return (
    <div className="welcome">
      <div className="welcome__left">
        <div className="welcome__inner">
          <h1 className="welcome__title">
            {author ? <>Welcome back,<br />{author}.</> : 'Welcome back.'}
          </h1>

          <p className="welcome__lede">
            {recent.length > 0
              ? 'Continue with a recent project, or start a new one.'
              : 'Create a new project, or open an existing .fidu bundle.'}
          </p>

          {draft ? (
            <div className="newproject">
              <div className="field">
                <label className="field__label" htmlFor="np-name">Project name</label>
                <input
                  id="np-name"
                  ref={nameRef}
                  value={draft.name}
                  onChange={(event) => setDraft({ ...draft, name: event.target.value })}
                  onKeyDown={(event) => {
                    if (event.key === 'Enter') confirmCreate()
                    if (event.key === 'Escape') setDraft(null)
                  }}
                />
              </div>

              <div className="field">
                <label className="field__label">What are you processing?</label>
                {WORKFLOWS.map((workflow) => (
                  <button
                    key={workflow.id}
                    type="button"
                    className={`option ${draft.workflow === workflow.id ? 'option--on' : ''}`}
                    onClick={() => setDraft({ ...draft, workflow: workflow.id })}
                  >
                    <span className="option__name">{workflow.name}</span>
                    <span className="option__hint">{workflow.hint}</span>
                  </button>
                ))}
              </div>

              <div className="field">
                <label className="field__label">Location</label>
                <button className="btn btn--block" onClick={chooseFolder}>
                  {draft.folder ? 'Change folder…' : 'Choose a folder…'}
                </button>
              </div>

              {bundlePath && (
                <p className="newproject__path data">{bundlePath}</p>
              )}

              <div className="welcome__actions" style={{ marginBottom: 0 }}>
                <button
                  className="btn btn--primary"
                  onClick={confirmCreate}
                  disabled={loading || !draft.folder || !draft.name.trim()}
                >
                  Create project
                </button>
                <button className="btn" onClick={() => setDraft(null)} disabled={loading}>
                  Cancel
                </button>
              </div>
            </div>
          ) : (
            <div className="welcome__actions">
              <button className="btn btn--primary" onClick={beginCreate} disabled={loading}>
                New project
              </button>
              <button className="btn" onClick={onOpen} disabled={loading}>
                Open project
              </button>
            </div>
          )}

          {recent.length > 0 && (
            <div className="welcome__recent">
              <div className="section__head">
                <span className="section__title">Recent</span>
                <span className="section__rule" />
              </div>
              <div className="rows">
                {recent.map((entry) => (
                  <button
                    key={entry.directory}
                    className="row"
                    onClick={async () => {
                      if (await openProject(entry.directory)) {
                        rememberProject(entry.directory, entry.name)
                        setRecent(loadRecent())
                      }
                    }}
                  >
                    <div className="row__main">
                      <div className="row__name">{entry.name}</div>
                      <div className="row__meta truncate">{entry.directory}</div>
                    </div>
                    {entry.at && (
                      <span className="welcome__opened" title={new Date(entry.at).toLocaleString()}>
                        {lastOpened(entry.at)}
                      </span>
                    )}
                  </button>
                ))}
              </div>
            </div>
          )}

          <div className="welcome__credit">
            <span>
              Created by Joshua Metcalf ·{' '}
              <a href="https://www.linkedin.com/in/joshua-metcalf-1b2184263" target="_blank" rel="noreferrer">
                LinkedIn
              </a>
            </span>
            <button className="btn btn--ghost btn--sm welcome__settings" onClick={() => setSettingsOpen(true)}>
              <GearIcon /> Settings
            </button>
          </div>
        </div>
      </div>

      <div className="welcome__plate">
        <RectificationPlate />
      </div>
    </div>
  )
}

/**
 * The hero: the central act of this discipline, drawn once.
 *
 * A tilted exposure — carried on its fiducial marks — resolving onto the
 * orthogonal ground grid beneath it, with the perspective rays that connect
 * them. This is what orthorectification *is*, and it is the one place in the
 * interface allowed to draw at length. It animates a single time on load
 * rather than scattering entrance effects across the page.
 */
function RectificationPlate() {
  return (
    <svg className="plate" viewBox="0 0 520 640" preserveAspectRatio="xMidYMid meet"
         role="img" aria-label="A tilted aerial exposure resolving onto a ground grid">
      <defs>
        <linearGradient id="rayFade" x1="0" y1="0" x2="0" y2="1">
          <stop offset="0%" stopColor="var(--accent)" stopOpacity="0.5" />
          <stop offset="100%" stopColor="var(--accent)" stopOpacity="0.22" />
        </linearGradient>
        <clipPath id="frameClip">
          <polygon points="150,108 400,150 372,272 122,224" />
        </clipPath>
      </defs>

      {/* The ground: an orthogonal grid, north up, the destination. */}
      <g className="plate__ground">
        {Array.from({ length: 11 }, (_, i) => (
          <line key={`gx${i}`} x1={70 + i * 38} y1={392} x2={70 + i * 38} y2={620} />
        ))}
        {Array.from({ length: 7 }, (_, i) => (
          <line key={`gy${i}`} x1={70} y1={392 + i * 38} x2={450} y2={392 + i * 38} />
        ))}
      </g>

      {/* Ground control: the surveyed points both worlds agree on. */}
      <g className="plate__control">
        {[[146, 468], [298, 430], [412, 544], [184, 582], [336, 520]].map(([x, y], i) => (
          <g key={i}>
            <line x1={x - 7} y1={y} x2={x + 7} y2={y} />
            <line x1={x} y1={y - 7} x2={x} y2={y + 7} />
          </g>
        ))}
      </g>

      {/* Perspective rays from the exposure down to the control points. */}
      <g className="plate__rays">
        {[[150, 108, 146, 468], [400, 150, 412, 544], [372, 272, 336, 520],
          [122, 224, 184, 582], [261, 188, 298, 430]].map(([x1, y1, x2, y2], i) => (
          <line key={i} x1={x1} y1={y1} x2={x2} y2={y2} />
        ))}
      </g>

      {/* The exposure: tilted, carrying its own terrain inside the frame. */}
      <g className="plate__frame">
        <polygon className="plate__frame-fill" points="150,108 400,150 372,272 122,224" />
        <g clipPath="url(#frameClip)" className="plate__terrain">
          {Array.from({ length: 9 }, (_, i) => (
            <path
              key={i}
              d={`M110,${118 + i * 20} C 190,${100 + i * 20} 250,${150 + i * 20} 330,${128 + i * 20} S 430,${168 + i * 20} 415,${176 + i * 20}`}
            />
          ))}
        </g>
        <polygon className="plate__frame-edge" points="150,108 400,150 372,272 122,224" />

        {/* Fiducial marks at the frame corners and edge midpoints. */}
        <g className="plate__fiducials">
          {[[150, 108], [400, 150], [372, 272], [122, 224],
            [275, 129], [386, 211], [247, 248], [136, 166]].map(([x, y], i) => (
            <g key={i}>
              <line x1={x - 6} y1={y} x2={x + 6} y2={y} />
              <line x1={x} y1={y - 6} x2={x} y2={y + 6} />
            </g>
          ))}
        </g>
      </g>

      {/* The perspective centre. */}
      <circle className="plate__centre" cx="261" cy="188" r="3" />

      <g className="plate__annotation">
        <text x="450" y="76" className="plate__label" textAnchor="end">exposure</text>
        <text x="450" y="90" className="plate__sublabel" textAnchor="end">
          tilted, unmeasured
        </text>
        <text x="70" y="360" className="plate__label">ground</text>
        <text x="70" y="374" className="plate__sublabel">Lo19, half-metre</text>
      </g>
    </svg>
  )
}
