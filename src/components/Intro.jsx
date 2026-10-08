import { useEffect, useRef, useState } from 'react'
import { useStore } from '../state/store'
import { BrandMark } from './Brand'

/**
 * First-launch introduction: three pages, each an animated plate in the same
 * drawing language as the start screen (ground grid, fiducial crosses, dashed
 * rays, the teal principal point).
 *
 * Every scene is keyed by its page, so its animation plays from the start
 * each time the page is shown. Reduced-motion settings show the final frame.
 */

const PAGES = [
  {
    kicker: 'Efficiency',
    title: <>Efficiency<br />at heart</>,
    body: 'Fiducia processes on every available core and keeps each step in a single workspace, '
      + 'taking aerial and satellite imagery directly to measured, map-ready products.',
    Scene: FrameScene,
  },
  {
    kicker: 'Transparency',
    title: <>Transparency<br />as you need it</>,
    body: 'Residuals, precision and suspect points are reported at every stage of the adjustment, '
      + 'from the first control point to the final solution. Any outliers are clearly labelled.',
    Scene: ControlScene,
  },
  {
    kicker: 'Robustness',
    // U+2011, a non-breaking hyphen, so "crash-proof" never splits across lines.
    title: <>Robust, crash‑proof<br />processing</>,
    body: (
      <>
        Orthorectify, mosaic and extract terrain from the same workspace.{' '}
        <strong className="intro__highlight">Every edit is saved to disk the moment it is made</strong>,
        so a crash or power cut loses nothing. Snapshots add named restore points, so the project
        can be returned to any earlier state.
      </>
    ),
    Scene: ProductScene,
  },
]

const reducedMotion = () =>
  typeof window !== 'undefined' && window.matchMedia?.('(prefers-reduced-motion: reduce)').matches

export default function Intro() {
  const setIntroOpen = useStore((s) => s.setIntroOpen)
  const startPage = useStore((s) => s.introPage)
  const author = useStore((s) => s.author)
  const setAuthor = useStore((s) => s.setAuthor)
  const [page, setPage] = useState(Math.min(startPage || 0, PAGES.length - 1))
  const [name, setName] = useState(author)
  const nextRef = useRef(null)
  const nameRef = useRef(null)
  const last = page === PAGES.length - 1
  const { kicker, title, body, Scene } = PAGES[page]

  // The introduction always ends by asking for the operator's name, which is
  // recorded as the author of their projects and orthophotos. Until one is
  // given, skipping leads to that question rather than past it.
  const finish = () => {
    if (!name.trim()) {
      setPage(PAGES.length - 1)
      nameRef.current?.focus()
      return
    }
    setAuthor(name)
    setIntroOpen(false)
  }
  const skip = () => (author ? setIntroOpen(false) : setPage(PAGES.length - 1))
  const next = () => (last ? finish() : setPage((p) => p + 1))
  const back = () => setPage((p) => Math.max(0, p - 1))

  useEffect(() => {
    if (last) nameRef.current?.focus()
    else nextRef.current?.focus()
  }, [page, last])

  useEffect(() => {
    const onKey = (event) => {
      const typing = event.target instanceof HTMLInputElement
      if (event.key === 'Escape') skip()
      else if (typing) return
      else if (event.key === 'ArrowRight') next()
      else if (event.key === 'ArrowLeft') back()
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  })

  return (
    <div className="intro" role="dialog" aria-modal="true" aria-label="Introduction to Fiducia">
      <div className="intro__grid" aria-hidden="true" />

      <header className="intro__top">
        <span className="intro__brand"><BrandMark size={20} /> Fiducia</span>
        {!(last && !author) && (
          <button className="btn btn--ghost btn--sm" onClick={skip}>Skip</button>
        )}
      </header>

      <main className="intro__body">
        <section className="intro__copy" key={`copy-${page}`}>
          <span className="intro__kicker">
            <span className="intro__count">{String(page + 1).padStart(2, '0')} / 03</span>
            {kicker}
          </span>
          <h1 className="intro__title">{title}</h1>
          <p className="intro__text">{body}</p>
          {last && (
            <form
              className="intro__name"
              onSubmit={(event) => { event.preventDefault(); finish() }}
            >
              <label className="intro__name-label" htmlFor="intro-author">
                Your name
                <span>Recorded as the author of your projects and orthophotos.</span>
              </label>
              <input
                id="intro-author"
                ref={nameRef}
                value={name}
                autoComplete="name"
                spellCheck={false}
                placeholder="First and last name"
                onChange={(event) => setName(event.target.value)}
              />
            </form>
          )}
          {last && (
            <p className="intro__credit">
              Created by Joshua Metcalf ·{' '}
              <a href="https://www.linkedin.com/in/joshua-metcalf-1b2184263" target="_blank" rel="noreferrer">
                LinkedIn
              </a>
            </p>
          )}
        </section>

        <div className="intro__stage" key={`scene-${page}`}>
          <Scene still={reducedMotion()} />
        </div>
      </main>

      <footer className="intro__foot">
        <div className="intro__dots" role="tablist" aria-label="Pages">
          {PAGES.map((entry, index) => (
            <button
              key={index}
              role="tab"
              aria-selected={index === page}
              aria-label={`Page ${index + 1}: ${entry.kicker}`}
              className={`intro__dot ${index === page ? 'intro__dot--on' : ''}`}
              onClick={() => setPage(index)}
            />
          ))}
        </div>
        <div className="intro__actions">
          {page > 0 && <button className="btn" onClick={back}>Back</button>}
          <button
            ref={nextRef}
            className="btn btn--primary"
            onClick={next}
            disabled={last && !name.trim()}
          >
            {last ? 'Get started' : 'Next'}
          </button>
        </div>
      </footer>
    </div>
  )
}

// -- shared drawing helpers ------------------------------------------------

/** A fiducial cross that pops in. */
function Cross({ x, y, size = 7, delay = 0, className = 'intro-fid' }) {
  return (
    <g className={`${className} pop`} style={{ animationDelay: `${delay}ms` }}>
      <line x1={x - size} y1={y} x2={x + size} y2={y} />
      <line x1={x} y1={y - size} x2={x} y2={y + size} />
    </g>
  )
}

/** A line or path that draws itself. pathLength=1 normalises the dash. */
function Draw({ as: Tag = 'line', delay = 0, duration, className = '', ...props }) {
  return (
    <Tag
      pathLength="1"
      className={`draw ${className}`}
      style={{ animationDelay: `${delay}ms`, ...(duration ? { animationDuration: `${duration}ms` } : {}) }}
      {...props}
    />
  )
}

function Label({ x, y, delay = 0, anchor = 'start', children, sub }) {
  return (
    <g className="fade" style={{ animationDelay: `${delay}ms` }}>
      <text x={x} y={y} className="intro-label" textAnchor={anchor}>{children}</text>
      {sub && <text x={x} y={y + 14} className="intro-sublabel" textAnchor={anchor}>{sub}</text>}
    </g>
  )
}

function Ground({ x0 = 50, x1 = 470, y0 = 230, y1 = 410, step = 30, delay = 0 }) {
  // The outer edge is always drawn, even when the extent is not a whole
  // number of steps, so the grid never ends open.
  const at = (from, to) => {
    const out = []
    for (let v = from; v <= to; v += step) out.push(v)
    if (out[out.length - 1] !== to) out.push(to)
    return out
  }
  const lines = []
  for (const x of at(x0, x1)) lines.push([x, y0, x, y1])
  for (const y of at(y0, y1)) lines.push([x0, y, x1, y])
  return (
    <g className="intro-ground">
      {lines.map(([ax, ay, bx, by], i) => (
        <Draw key={i} x1={ax} y1={ay} x2={bx} y2={by} delay={delay + i * 18} duration={700} />
      ))}
    </g>
  )
}

// -- page 1: the frame and its fiducials -----------------------------------

function FrameScene() {
  const frame = [[130, 90], [400, 130], [372, 300], [102, 262]]
  const mids = [[265, 110], [386, 215], [237, 281], [116, 176]]
  const centre = [251.8, 195.7]   // where the corner diagonals cross
  const points = frame.map((p) => p.join(',')).join(' ')

  return (
    <svg className="intro-scene" viewBox="0 0 520 420" role="img"
         aria-label="A tilted aerial frame: its fiducial marks appear, lines join opposite marks, and the principal point appears where they cross">
      <defs>
        <clipPath id="intro-frame-clip"><polygon points={points} /></clipPath>
      </defs>

      <g className="fade" style={{ animationDelay: '100ms' }}>
        <polygon points={points} className="intro-frame-fill" />
        <g clipPath="url(#intro-frame-clip)" className="intro-terrain">
          {Array.from({ length: 9 }, (_, i) => (
            <Draw key={i} as="path" delay={300 + i * 60} duration={1100}
                  d={`M80,${104 + i * 24} C 170,${86 + i * 24} 230,${140 + i * 24} 310,${118 + i * 24} S 420,${160 + i * 24} 430,${170 + i * 24}`} />
          ))}
        </g>
      </g>
      <Draw as="polygon" points={points} className="intro-frame-edge" delay={150} duration={1100} />

      {[...frame, ...mids].map(([x, y], i) => (
        <Cross key={i} x={x} y={y} delay={1000 + i * 110} />
      ))}

      <g className="intro-collimation">
        <Draw x1={130} y1={90} x2={372} y2={300} delay={2000} duration={800} />
        <Draw x1={400} y1={130} x2={102} y2={262} delay={2200} duration={800} />
      </g>

      <circle className="intro-pulse" cx={centre[0]} cy={centre[1]} r="14" style={{ animationDelay: '3000ms' }} />
      <circle className="intro-point pop" cx={centre[0]} cy={centre[1]} r="6" style={{ animationDelay: '2900ms' }} />

      <Label x={410} y={78} anchor="end" delay={1700} sub="measured on every frame">fiducial marks</Label>
      <Label x={272} y={176} delay={3200} sub="the true image centre">principal point</Label>
    </svg>
  )
}

// -- page 2: rays to ground control ----------------------------------------

function ControlScene() {
  const centre = [260, 58]
  const gcps = [[120, 300], [220, 362], [350, 278], [420, 370], [300, 330]]
  const residuals = [[16, -10], [-14, -12], [12, 14], [-16, 8], [10, -15]]

  return (
    <svg className="intro-scene" viewBox="0 0 520 420" role="img"
         aria-label="Rays from the camera reach down to ground control points, and their residual errors shrink as the block is adjusted">
      <Ground delay={0} />

      <g className="fade" style={{ animationDelay: '500ms' }}>
        <polygon points="214,96 306,104 300,126 208,118" className="intro-frame-fill intro-frame-edge--solid" />
      </g>
      <circle className="intro-point pop" cx={centre[0]} cy={centre[1]} r="5" style={{ animationDelay: '650ms' }} />
      <Label x={276} y={52} delay={800} sub="solved for every image">perspective centre</Label>

      {gcps.map(([x, y], i) => (
        <g key={i}>
          <Draw x1={centre[0]} y1={centre[1]} x2={x} y2={y} className="intro-ray"
                delay={1000 + i * 180} duration={700} />
          <line x1={centre[0]} y1={centre[1]} x2={x} y2={y} className="intro-ray-flow fade"
                style={{ animationDelay: `${1900 + i * 180}ms, ${1900 + i * 180}ms` }} />
          <Cross x={x} y={y} size={8} delay={1500 + i * 180} className="intro-gcp" />
          <line
            x1={x} y1={y} x2={x + residuals[i][0] * 2.4} y2={y + residuals[i][1] * 2.4}
            className="intro-residual shrink"
            style={{ transformOrigin: `${x}px ${y}px`, animationDelay: `${2600 + i * 60}ms` }}
          />
        </g>
      ))}

      <g className="intro-readout">
        <text x={50} y={216} className="intro-label fade swap-out" style={{ animationDelay: '2300ms, 3300ms' }}>
          RMS 2.41 m
        </text>
        <text x={50} y={216} className="intro-label intro-label--good fade" style={{ animationDelay: '3500ms' }}>
          RMS 0.18 m · converged
        </text>
      </g>
    </svg>
  )
}

// -- page 3: rectify, mosaic, contour --------------------------------------

function ProductScene({ still }) {
  const tilted = '160,70 390,112 362,252 128,214'
  const square = '120,130 270,130 270,300 120,300'

  // The tilted frame settles onto the ground grid. Interpolated in script
  // rather than with SVG <animate>, whose clock starts with the page, not
  // with this scene, and whose progress Chromium does not report reliably.
  const [points, setPoints] = useState(still ? square : tilted)
  useEffect(() => {
    if (still) return undefined
    const from = tilted.split(/[ ,]/).map(Number)
    const to = square.split(/[ ,]/).map(Number)
    const ease = (t) => 1 - (1 - t) ** 3
    let frame = 0
    let startAt = null
    const tick = (now) => {
      startAt ??= now + 900
      const t = Math.min(1, Math.max(0, (now - startAt) / 1300))
      const k = ease(t)
      const current = from.map((value, i) => value + (to[i] - value) * k)
      setPoints(current.reduce((out, value, i) =>
        out + (i % 2 ? `,${value.toFixed(1)}` : `${i ? ' ' : ''}${value.toFixed(1)}`), ''))
      if (t < 1) frame = requestAnimationFrame(tick)
    }
    frame = requestAnimationFrame(tick)
    return () => cancelAnimationFrame(frame)
  }, [still])

  return (
    <svg className="intro-scene" viewBox="0 0 520 420" role="img"
         aria-label="The tilted frame straightens onto the ground grid, a second image joins it as a mosaic along a cutline, and contour lines are drawn across both">
      <Ground x0={50} x1={470} y0={100} y1={380} step={30} delay={0} />

      <polygon className="intro-ortho fade" points={points} style={{ animationDelay: '200ms' }} />
      <Label x={120} y={104} delay={2200} sub="terrain displacement removed">orthophoto</Label>

      <rect x="270" y="130" width="150" height="170" className="intro-ortho intro-ortho--second slide-in"
            style={{ animationDelay: '2600ms' }} />
      <Draw as="path" d="M270,130 C 262,170 280,200 268,236 S 276,280 270,300" className="intro-cutline"
            delay={3300} duration={900} />
      <Label x={420} y={104} anchor="end" delay={3500} sub="seamless across the cutline">mosaic</Label>

      <g className="intro-contours">
        {[0, 1, 2, 3].map((i) => (
          <Draw key={i} as="path" delay={4000 + i * 160} duration={1200}
                d={`M122,${160 + i * 36} C 180,${140 + i * 36} 230,${196 + i * 36} 300,${170 + i * 36} S 390,${150 + i * 36} 418,${182 + i * 36}`} />
        ))}
      </g>
      <Label x={120} y={330} delay={4700} sub="and terrain models">contours</Label>
    </svg>
  )
}
