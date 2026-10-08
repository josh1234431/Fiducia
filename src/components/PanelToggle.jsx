/**
 * Shows or hides a side panel. One icon for both panels: a window with a
 * sidebar, mirrored for the right-hand one. The sidebar is filled while the
 * panel is open.
 *
 * Open, it sits on the panel it hides. Hidden, the app places it at the same
 * edge of the canvas, so the control is always where the panel was.
 */
export default function PanelToggle({ side = 'left', open, onToggle, label, shortcut, floating = false }) {
  const action = open ? `Hide ${label}` : `Show ${label}`
  return (
    <button
      type="button"
      className={`panel-toggle panel-toggle--${side} ${floating ? 'panel-toggle--floating' : ''}`}
      onClick={onToggle}
      title={`${action} (${shortcut})`}
      aria-label={action}
      aria-pressed={open}
    >
      <svg width="16" height="16" viewBox="0 0 16 16" aria-hidden="true"
           style={side === 'right' ? { transform: 'scaleX(-1)' } : undefined}>
        <rect x="1.5" y="2.5" width="13" height="11" rx="2" fill="none"
              stroke="currentColor" strokeWidth="1.3" />
        <line x1="6" y1="2.5" x2="6" y2="13.5" stroke="currentColor" strokeWidth="1.3" />
        {open && <rect x="2.5" y="3.5" width="3" height="9" rx="1" fill="currentColor" opacity="0.45" />}
      </svg>
    </button>
  )
}
