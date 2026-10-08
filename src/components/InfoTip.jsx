import { useLayoutEffect, useRef, useState } from 'react'
import { createPortal } from 'react-dom'

/**
 * A small ⓘ beside a label that reveals detail on hover or focus.
 *
 * The interface shows only what an experienced operator needs to act; the
 * reasoning behind a setting lives here, one step away, for when it is
 * wanted. The popover is portalled and clamped to the window so it is never
 * cut off by the narrow inspector column.
 */
export default function InfoTip({
  children,
  label = 'More information',
  className = 'infotip',
  trigger = 'i',
  onClick,
}) {
  const anchor = useRef(null)
  const bubble = useRef(null)
  const [open, setOpen] = useState(false)
  const [position, setPosition] = useState(null)

  useLayoutEffect(() => {
    if (!open || !anchor.current || !bubble.current) return
    const a = anchor.current.getBoundingClientRect()
    const b = bubble.current.getBoundingClientRect()
    const margin = 10
    let left = a.left + a.width / 2 - b.width / 2
    left = Math.max(margin, Math.min(left, window.innerWidth - b.width - margin))
    let top = a.top - b.height - 8
    if (top < margin) top = a.bottom + 8
    setPosition({ left, top })
  }, [open])

  const show = () => setOpen(true)
  const hide = () => { setOpen(false); setPosition(null) }

  return (
    <>
      <button
        ref={anchor}
        type="button"
        className={className}
        aria-label={label}
        aria-expanded={open}
        onMouseEnter={show}
        onMouseLeave={hide}
        onFocus={show}
        onBlur={hide}
        onClick={(event) => {
          event.preventDefault()
          event.stopPropagation()
          if (onClick) onClick(event)
          else if (open) hide()
          else show()
        }}
        onKeyDown={(event) => { if (event.key === 'Escape') hide() }}
      >
        {trigger}
      </button>
      {open && createPortal(
        <div
          ref={bubble}
          role="tooltip"
          className="infotip__bubble"
          style={position ?? { left: -9999, top: -9999 }}
        >
          {children}
        </div>,
        document.body,
      )}
    </>
  )
}
