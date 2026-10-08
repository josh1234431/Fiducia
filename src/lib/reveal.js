import { useEffect, useRef } from 'react'
import { useStore } from '../state/store'

/**
 * A ref for a step's list of outputs. When the activity bar asks to see that
 * step's outputs, the list scrolls into view and flashes once, so the eye
 * lands on what was just made.
 */
export function useReveal(step) {
  const ref = useRef(null)
  const request = useStore((s) => s.revealRequest)

  useEffect(() => {
    if (!request || request.step !== step) return undefined
    // The panel may only just have mounted; wait a frame for its layout.
    const frame = requestAnimationFrame(() => {
      const node = ref.current
      if (!node) return
      node.scrollIntoView({ behavior: 'smooth', block: 'start' })
      node.classList.remove('section--revealed')
      void node.offsetWidth   // restart the animation
      node.classList.add('section--revealed')
    })
    return () => cancelAnimationFrame(frame)
  }, [request?.token, step])

  return ref
}
