import './lib/carryOverPreferences'
import React from 'react'
import { createRoot } from 'react-dom/client'
import App from './App'
import PopoutApp from './PopoutApp'
import './styles/theme.css'
import './styles/app.css'
import { useStore } from './state/store'

// Apply the stored theme before first paint so there is no flash of the
// wrong surface — which matters more here than usual, because the operator's
// eyes are about to be judging image tone.
document.documentElement.setAttribute(
  'data-theme',
  localStorage.getItem('fiducia.theme') || 'dark',
)

// A popped-out image window loads the same bundle with #popout=<imageId>.
const popout = new URLSearchParams(window.location.hash.slice(1)).get('popout')
if (popout) useStore.setState({ popoutImageId: popout })

// Development only: lets the store be inspected from the console.
if (import.meta.env.DEV) window.__fiduciaStore = useStore

useStore.getState().init()

createRoot(document.getElementById('root')).render(
  <React.StrictMode>
    {popout ? <PopoutApp imageId={popout} /> : <App />}
  </React.StrictMode>,
)
