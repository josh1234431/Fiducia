import { useEffect, useMemo, useRef, useState } from 'react'
import { useStore, stepsFor } from '../state/store'
import { api } from '../lib/api'

/**
 * Command palette.
 *
 * Ctrl+K reaches anything in the application by name. This is the direct answer
 * to "processes are convoluted and stretched out amongst various menus": the
 * operator who knows they want to compute the model does not have to remember
 * which step it lives under.
 */
export default function CommandPalette() {
  const open = useStore((s) => s.paletteOpen)
  const setOpen = useStore((s) => s.setPaletteOpen)
  const setStep = useStore((s) => s.setStep)
  const setActiveImage = useStore((s) => s.setActiveImage)
  const setTheme = useStore((s) => s.setTheme)
  const theme = useStore((s) => s.theme)
  const project = useStore((s) => s.project)
  const call = useStore((s) => s.call)
  const toast = useStore((s) => s.toast)
  const closeProject = useStore((s) => s.closeProject)
  const setIntroOpen = useStore((s) => s.setIntroOpen)

  const [query, setQuery] = useState('')
  const [index, setIndex] = useState(0)
  const inputRef = useRef(null)

  useEffect(() => {
    if (open) {
      setQuery('')
      setIndex(0)
      setTimeout(() => inputRef.current?.focus(), 10)
    }
  }, [open])

  const commands = useMemo(() => {
    const list = []

    stepsFor(project).all.forEach((step) => {
      list.push({
        id: `step-${step.id}`,
        group: 'Go to',
        name: step.name,
        hint: step.blurb,
        run: () => setStep(step.id),
      })
    })

    ;(project?.images || []).forEach((image) => {
      list.push({
        id: `img-${image.id}`,
        group: 'Images',
        name: image.name,
        hint: image.online
          ? `${image.width} × ${image.height}`
          : 'Offline',
        run: () => { setActiveImage(image.id); setStep('images') },
      })
    })

    if (project) {
      list.push(
        {
          id: 'compute',
          group: 'Actions',
          name: 'Compute sensor model',
          hint: 'Space resection or bundle adjustment',
          run: () => { setStep('model'); call(() => api.model.compute()) },
        },
        {
          id: 'snapshot',
          group: 'Actions',
          name: 'Take a snapshot',
          hint: 'Named restore point',
          run: () => call(() => api.project.snapshot('manual'),
            { successMessage: 'Snapshot taken' }),
        },
        {
          id: 'relink',
          group: 'Actions',
          name: 'Relink offline images',
          hint: 'Search known folders for moved files',
          run: () => call(() => api.project.relink(),
            { successMessage: 'Relink complete' }),
        },
        {
          id: 'report-project',
          group: 'Actions',
          name: 'Project report',
          hint: 'Open the Reports step',
          run: () => setStep('reports'),
        },
        {
          id: 'close-project',
          group: 'Actions',
          name: 'Close project',
          hint: 'Return to the start screen',
          run: () => closeProject(),
        },
      )
    }

    list.push(
      {
        id: 'theme',
        group: 'View',
        name: `Switch to ${theme === 'dark' ? 'light' : 'dark'} theme`,
        run: () => setTheme(theme === 'dark' ? 'light' : 'dark'),
      },
      {
        id: 'intro',
        group: 'Help',
        name: 'Show introduction',
        hint: 'The three-page overview shown on first launch',
        run: () => setIntroOpen(true),
      },
      {
        id: 'settings',
        group: 'Settings',
        name: 'Settings',
        hint: 'Name, certificate reading, appearance',
        run: () => useStore.getState().setSettingsOpen(true),
      },
      {
        id: 'author',
        group: 'Settings',
        name: 'Change name',
        hint: 'Recorded in new projects and orthophotos',
        run: () => useStore.getState().setSettingsOpen(true, 'profile'),
      },
    )

    return list
  }, [project, theme, setStep, setActiveImage, setTheme, call, closeProject, setIntroOpen])

  const filtered = useMemo(() => {
    const needle = query.trim().toLowerCase()
    if (!needle) return commands
    return commands.filter((command) =>
      `${command.name} ${command.hint} ${command.group}`.toLowerCase().includes(needle),
    )
  }, [commands, query])

  useEffect(() => { setIndex(0) }, [query])

  if (!open) return null

  function runAt(position) {
    const command = filtered[position]
    if (!command) return
    setOpen(false)
    try {
      command.run()
    } catch (error) {
      toast(String(error), 'bad')
    }
  }

  function onKeyDown(event) {
    if (event.key === 'Escape') { setOpen(false); return }
    if (event.key === 'ArrowDown') {
      event.preventDefault()
      setIndex((i) => Math.min(i + 1, filtered.length - 1))
    }
    if (event.key === 'ArrowUp') {
      event.preventDefault()
      setIndex((i) => Math.max(i - 1, 0))
    }
    if (event.key === 'Enter') { event.preventDefault(); runAt(index) }
  }

  let lastGroup = null

  return (
    <div className="palette" onClick={() => setOpen(false)}>
      <div className="palette__panel" onClick={(event) => event.stopPropagation()}>
        <input
          ref={inputRef}
          className="palette__input"
          placeholder="Search steps, images and actions…"
          value={query}
          onChange={(event) => setQuery(event.target.value)}
          onKeyDown={onKeyDown}
        />

        <div className="palette__list">
          {filtered.length === 0 && (
            <div className="palette__empty">Nothing matches “{query}”</div>
          )}

          {filtered.map((command, position) => {
            const header = command.group !== lastGroup ? command.group : null
            lastGroup = command.group
            return (
              <div key={command.id}>
                {header && <div className="palette__group">{header}</div>}
                <button
                  className={`palette__item ${position === index ? 'palette__item--active' : ''}`}
                  onMouseEnter={() => setIndex(position)}
                  onClick={() => runAt(position)}
                >
                  <span className="palette__item-main">
                    <span className="palette__item-name truncate">{command.name}</span>
                    <span className="palette__item-hint truncate">{command.hint}</span>
                  </span>
                  {position === index && <kbd>enter</kbd>}
                </button>
              </div>
            )
          })}
        </div>
      </div>
    </div>
  )
}
