/**
 * The Fiducia mark and wordmark. Source artwork is in resources/brand.
 *
 * The mark is a film frame reduced to two opposite fiducials: the top-left one
 * is drawn as an F, and the teal dot is the principal point, the exact centre
 * where lines between opposite fiducials cross. The wordmark is plain type.
 */
export function BrandMark({ size = 18 }) {
  return (
    <svg
      className="brandmark"
      width={size}
      height={size}
      viewBox="10 10 44 44"
      aria-hidden="true"
    >
      <path
        d="M14 44V14h22M14 25.5h10M50 41v9h-9"
        stroke="currentColor"
        strokeWidth="4.5"
        strokeLinecap="round"
        strokeLinejoin="round"
        fill="none"
      />
      <circle className="brandmark__point" cx="32" cy="32" r="4.75" />
    </svg>
  )
}

/** "Fiducia" set in the display face. */
export function Wordmark({ className = '' }) {
  return <span className={`wordmark ${className}`}>Fiducia</span>
}
