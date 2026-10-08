import { useEffect, useState } from 'react'
import { useStore } from '../state/store'
import { api } from '../lib/api'
import { ReadingService } from './CertificateReader'
import { BrandMark } from './Brand'
import pkg from '../../package.json'

const RECENT_KEY = 'fiducia.recent'

function recentCount() {
  try {
    return JSON.parse(localStorage.getItem(RECENT_KEY) || '[]').length
  } catch {
    return 0
  }
}

/** A computer screen, for the statement that data stays on it. */
function LocalIcon() {
  return (
    <svg width="16" height="16" viewBox="0 0 16 16" aria-hidden="true" fill="none"
         stroke="currentColor" strokeWidth="1.3" strokeLinecap="round">
      <rect x="1.5" y="2.5" width="13" height="8.5" rx="1.5" />
      <path d="M5.5 13.5h5M8 11v2.5" />
    </svg>
  )
}

function Section({ id, title, children }) {
  return (
    <section className="settings__section" id={`settings-${id}`}>
      <h2 className="settings__heading">{title}</h2>
      {children}
    </section>
  )
}

/**
 * Settings that belong to the person and this computer rather than to a
 * project: their name, the certificate reading service, the theme.
 * Project settings stay in the Project step.
 */
export default function Settings() {
  const open = useStore((s) => s.settingsOpen)
  const section = useStore((s) => s.settingsSection)
  const setOpen = useStore((s) => s.setSettingsOpen)
  const author = useStore((s) => s.author)
  const setAuthor = useStore((s) => s.setAuthor)
  const theme = useStore((s) => s.theme)
  const setTheme = useStore((s) => s.setTheme)
  const setIntroOpen = useStore((s) => s.setIntroOpen)
  const toast = useStore((s) => s.toast)
  const loupe = useStore((s) => s.loupe)
  const setLoupe = useStore((s) => s.setLoupe)

  const [name, setName] = useState(author)
  const [status, setStatus] = useState(null)
  const [statusError, setStatusError] = useState(false)
  const [recent, setRecent] = useState(0)

  useEffect(() => {
    if (!open) return
    setName(author)
    setRecent(recentCount())
    setStatusError(false)
    api.certificate.status()
      .then(setStatus)
      .catch(() => setStatusError(true))
    if (section) {
      setTimeout(() => {
        document.getElementById(`settings-${section}`)
          ?.scrollIntoView({ block: 'start' })
      }, 30)
    }
  }, [open])

  useEffect(() => {
    if (!open) return undefined
    function onKeyDown(event) {
      if (event.key === 'Escape') close()
    }
    window.addEventListener('keydown', onKeyDown)
    return () => window.removeEventListener('keydown', onKeyDown)
  })

  if (!open) return null

  function saveName() {
    if (name.trim() === author) return
    setAuthor(name)
    toast(name.trim() ? 'Name saved' : 'Name removed', 'good')
  }

  function close() {
    saveName()
    setOpen(false)
  }

  function clearRecent() {
    try { localStorage.removeItem(RECENT_KEY) } catch { /* optional */ }
    window.dispatchEvent(new Event('fiducia:recent'))
    setRecent(0)
    toast('Recent projects cleared', 'good')
  }

  return (
    <div className="palette settings" onMouseDown={(event) => {
      if (event.target === event.currentTarget) close()
    }}>
      <div className="palette__panel settings__panel" role="dialog" aria-label="Settings">
        <header className="settings__head">
          <span className="settings__title">Settings</span>
          <button className="btn btn--ghost btn--sm" onClick={close} aria-label="Close settings">
            Done
          </button>
        </header>

        <div className="settings__body">
          <p className="settings__local">
            <LocalIcon />
            <span>
              Everything Fiducia keeps stays on this computer: projects, photographs,
              your name, keys and settings. There is no account and no cloud storage.
            </span>
          </p>

          <Section id="profile" title="Profile">
            <div className="field">
              <label className="field__label" htmlFor="settings-name">Name</label>
              <input
                id="settings-name"
                value={name}
                placeholder="Your name"
                autoComplete="name"
                onChange={(event) => setName(event.target.value)}
                onBlur={saveName}
                onKeyDown={(event) => { if (event.key === 'Enter') event.currentTarget.blur() }}
              />
              <div className="field__hint">
                Shown on the start screen and recorded as the author of new projects
                and orthophotos. Existing projects keep their recorded author.
              </div>
            </div>
          </Section>

          <Section id="reading" title="Certificate reading">
            <div className="field__hint settings__lede">
              Offline reading happens entirely on this computer. Claude and ChatGPT
              receive only the certificate you choose, using your own API key; that is
              the only time Fiducia uses the network. Keys are encrypted by Windows and
              stored on this computer.
            </div>
            {status?.services ? (
              <ReadingService status={status} onStatus={setStatus} />
            ) : (
              <div className="field__hint">
                {statusError ? 'The engine is not running, so these settings are unavailable.' : 'Loading…'}
              </div>
            )}
          </Section>

          <Section id="measuring" title="Measuring">
            <label className="field field--row">
              <span className="field__label">Magnifier when placing control points</span>
              <input
                id="settings-loupe"
                type="checkbox"
                checked={loupe}
                onChange={(event) => setLoupe(event.target.checked)}
              />
            </label>
            <div className="field__hint">
              While a control point is being placed and the view is zoomed out
              beyond 1:1, a round window follows the cursor showing the photo at
              1:1, with a crosshair on the pixel the click will record.
            </div>
          </Section>

          <Section id="appearance" title="Appearance">
            <div className="settings__choices">
              {['dark', 'light'].map((option) => (
                <button
                  key={option}
                  className={`option ${theme === option ? 'option--on' : ''}`}
                  onClick={() => setTheme(option)}
                >
                  <span className="option__name">{option === 'dark' ? 'Dark' : 'Light'}</span>
                </button>
              ))}
            </div>
          </Section>

          <Section id="general" title="General">
            <div className="settings__row">
              <div>
                <div className="settings__row-name">Recent projects</div>
                <div className="field__hint">
                  {recent ? `${recent} listed on the start screen. Clearing the list does not delete any files.` : 'The list is empty.'}
                </div>
              </div>
              <button className="btn btn--sm" onClick={clearRecent} disabled={!recent}>Clear</button>
            </div>
            <div className="settings__row">
              <div>
                <div className="settings__row-name">Introduction</div>
                <div className="field__hint">The three-page overview shown on first launch.</div>
              </div>
              <button className="btn btn--sm" onClick={() => { setOpen(false); setIntroOpen(true) }}>
                Show
              </button>
            </div>
          </Section>

          <Section id="about" title="About">
            <div className="settings__about">
              <BrandMark size={22} />
              <div>
                <div className="settings__row-name">Fiducia <span className="data dim">{pkg.version}</span></div>
                <div className="field__hint">
                  Created by Joshua Metcalf ·{' '}
                  <a href="https://www.linkedin.com/in/joshua-metcalf-1b2184263" target="_blank" rel="noreferrer">
                    LinkedIn
                  </a>
                </div>
              </div>
            </div>
          </Section>
        </div>
      </div>
    </div>
  )
}
