import { Component } from 'react'
import { useStore } from '../state/store'

/**
 * A fault in one part of the interface stays in that part.
 *
 * Without this, any rendering error unmounts the whole tree and the window
 * goes blank -- indistinguishable from a crash, with nothing to report. The
 * project itself is never at risk (the engine owns it and every edit is
 * already on disk), so the honest thing is to say which part failed, show the
 * error, and let the operator carry on with everything else.
 */
export default class Fallback extends Component {
  constructor(props) {
    super(props)
    this.state = { error: null }
  }

  static getDerivedStateFromError(error) {
    return { error }
  }

  componentDidCatch(error, info) {
    console.error(`[${this.props.area}] render failed`, error, info?.componentStack)
  }

  componentDidUpdate(previous) {
    // Moving to a different step is a fresh start for that step's panel.
    if (this.state.error && previous.resetKey !== this.props.resetKey) {
      this.setState({ error: null })
    }
  }

  render() {
    const { error } = this.state
    if (!error) return this.props.children

    const detail = `${error.name}: ${error.message}`
    return (
      <div className="fallback" role="alert">
        <div className="fallback__title">{this.props.area} could not be displayed</div>
        <div className="fallback__body">
          All changes have been saved. The rest of the application is
          unaffected.
        </div>
        <code className="fallback__detail">{detail}</code>
        <div className="fallback__actions">
          <button className="btn btn--primary btn--sm"
                  onClick={() => this.setState({ error: null })}>
            Try again
          </button>
          <button
            className="btn btn--ghost btn--sm"
            onClick={() => {
              navigator.clipboard?.writeText(`${detail}\n${error.stack || ''}`)
              useStore.getState().toast('Error details copied', 'good')
            }}
          >
            Copy details
          </button>
        </div>
      </div>
    )
  }
}
