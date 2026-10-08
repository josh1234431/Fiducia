import { useEffect, useState } from 'react'
import { useStore } from '../state/store'
import { api } from '../lib/api'
import InfoTip from './InfoTip'

/**
 * Reading a calibration certificate.
 *
 * Every value arrives as a proposal carrying the text it was read from, so the
 * operator confirms against the page rather than trusting a number. Nothing
 * reaches the project until they press Apply.
 *
 * The distortion table gets a second, mechanical check: it is fitted to the
 * polynomial, and the residual is reported. A misread digit moves that
 * residual by orders of magnitude, which no amount of model confidence can
 * talk its way past.
 */

function confidenceTone(confidence) {
  return confidence === 'high' ? '' : confidence === 'medium' ? 'chip--warn' : 'chip--bad'
}

function Reading({ label, entry, unit = 'mm' }) {
  if (!entry || entry.value == null) return null
  return (
    <div className="reading">
      <div className="reading__head">
        <span className="reading__label">{label}</span>
        <span className="reading__value data">
          {Number(entry.value).toLocaleString(undefined, { maximumFractionDigits: 6 })}
          {unit ? ` ${unit}` : ''}
        </span>
        {entry.confidence !== 'high' && (
          <span className={`chip ${confidenceTone(entry.confidence)}`}>
            {entry.confidence}
          </span>
        )}
      </div>
      {entry.sourceText && (
        <div className="reading__source">Source: “{entry.sourceText}”</div>
      )}
    </div>
  )
}

/**
 * Which service reads the certificate, and the operator's key for it.
 *
 * Keys go one way. The desktop shell encrypts them with the operating
 * system's credential protection and hands them to the engine; nothing here
 * can read one back, so the field only ever shows the last four characters.
 */
export function ReadingService({ status, onStatus }) {
  const toast = useStore((s) => s.toast)
  const [draft, setDraft] = useState('')
  const [saving, setSaving] = useState(false)
  const [persisted, setPersisted] = useState(true)

  const provider = status?.provider || 'anthropic'
  const services = status?.services || {}
  const current = services[provider] || {}
  const [model, setModel] = useState('')

  useEffect(() => {
    setDraft('')
    setModel(services[provider]?.model === services[provider]?.defaultModel
      ? '' : services[provider]?.model || '')
  }, [provider, services[provider]?.model])

  async function update(change) {
    setSaving(true)
    try {
      if (window.fiducia?.reader) {
        const result = await window.fiducia.reader.update(change)
        setPersisted(result.persisted)
        if (result.status) onStatus(result.status)
        else toast('The engine did not accept the settings. Check that it is running.', 'bad')
      } else {
        onStatus(await api.certificate.configure(change))
        setPersisted(false)
      }
      return true
    } catch (error) {
      toast(error.message || 'The reading service settings could not be saved', 'bad')
      return false
    } finally {
      setSaving(false)
    }
  }

  async function saveKey() {
    if (!draft.trim()) return
    if (await update({ keys: { [provider]: draft } })) {
      setDraft('')
      toast(`${current.label} key saved`, 'good')
    }
  }

  function keyState(service) {
    if (service.needsKey === false) {
      if (!service.installed) return 'Unavailable on this computer'
      return service.ocr ? 'On this computer, no key needed' : 'PDFs only on this computer'
    }
    if (!service.installed) return 'Package not installed'
    if (service.keyOrigin === 'saved') {
      return service.keyHint ? `Key saved, ending ${service.keyHint}` : 'Key saved'
    }
    if (service.keyOrigin === 'environment') return 'Key from environment'
    return 'No key'
  }

  return (
    <div className="reader-service">
      {Object.entries(services).map(([id, service]) => (
        <button
          key={id}
          className={`option ${provider === id ? 'option--on' : ''}`}
          onClick={() => provider !== id && update({ provider: id })}
          disabled={saving}
        >
          <span className="option__name">{service.label}</span>
          <span className="option__hint">{keyState(service)}</span>
        </button>
      ))}

      {current.needsKey === false ? (
        <div className="field__hint" style={{ marginTop: 'var(--step-2)' }}>
          Reads the text of a PDF directly, and scans with the text recognition built
          into Windows. Nothing leaves this computer. Suited to clearly printed
          certificates; faded or skewed scans read better with Claude or ChatGPT.
          {current.note && <><br />{current.note}</>}
        </div>
      ) : (
      <>
      <div className="field" style={{ marginTop: 'var(--step-2)' }}>
        <label className="field__label" htmlFor="reader-key">
          <span>
            {current.label} API key
            <InfoTip>
              {persisted
                ? 'Encrypted by the operating system for the current user and stored outside all projects. Sharing a project does not share the key.'
                : 'Encryption is unavailable, so the key is held only until Fiducia closes.'}
            </InfoTip>
          </span>
        </label>
        <div className="reader-service__key">
          <input
            id="reader-key"
            type="password"
            autoComplete="off"
            spellCheck={false}
            value={draft}
            placeholder={current.hasKey && current.keyOrigin === 'saved'
              ? `Saved, ending ${current.keyHint || '…'}. Paste to replace.`
              : provider === 'openai' ? 'sk-…' : 'sk-ant-…'}
            onChange={(event) => setDraft(event.target.value)}
            onKeyDown={(event) => { if (event.key === 'Enter') saveKey() }}
          />
          <button
            className="btn btn--primary btn--sm"
            onClick={saveKey}
            disabled={saving || !draft.trim()}
          >
            Save
          </button>
        </div>
        {current.keyOrigin === 'saved' && (
          <button
            className="btn btn--ghost btn--sm"
            style={{ marginTop: 6 }}
            disabled={saving}
            onClick={async () => {
              if (await update({ keys: { [provider]: null } })) {
                toast(`${current.label} key removed`, 'good')
              }
            }}
          >
            Remove saved key
          </button>
        )}
      </div>

      <div className="field">
        <label className="field__label" htmlFor="reader-model">Model</label>
        <input
          id="reader-model"
          spellCheck={false}
          value={model}
          placeholder={current.defaultModel}
          onChange={(event) => setModel(event.target.value)}
          onBlur={() => {
            const effective = model.trim() || current.defaultModel
            if (effective !== current.model) update({ models: { [provider]: model.trim() } })
          }}
        />
      </div>

      <div className="field__hint">
        Keys are issued by{' '}
        <a href={current.keyUrl} target="_blank" rel="noreferrer">
          {provider === 'openai' ? 'the OpenAI platform' : 'the Anthropic console'}
        </a>
        . Each certificate costs a few cents.
      </div>
      </>
      )}
    </div>
  )
}

export default function CertificateReader({ onApply }) {
  const call = useStore((s) => s.call)
  const toast = useStore((s) => s.toast)
  const jobs = useStore((s) => s.jobs)

  const [status, setStatus] = useState(null)
  const [jobId, setJobId] = useState(null)
  const [result, setResult] = useState(null)
  const [open, setOpen] = useState(false)
  const settingsOpen = useStore((s) => s.settingsOpen)
  const setSettingsOpen = useStore((s) => s.setSettingsOpen)
  const openService = () => setSettingsOpen(true, 'reading')

  const job = jobs.find((j) => j.id === jobId)

  // Read again whenever Settings closes, since that is where the service changes.
  useEffect(() => {
    if (settingsOpen) return
    api.certificate.status()
      .then(setStatus)
      .catch(() => setStatus(null))
  }, [settingsOpen])

  useEffect(() => {
    if (job?.status === 'done' && job.result) {
      setResult(job.result)
      setJobId(null)
    } else if (job?.status === 'failed') {
      setJobId(null)
      // A rejected key or an unknown model is fixed in the service settings,
      // so put them in front of the operator rather than just the toast.
      if (/API key|model/i.test(job.error || '')) openService()
    }
  }, [job?.status])

  async function choose() {
    if (!window.fiducia?.isDesktop) {
      toast('Certificates can only be read in the desktop app', 'warn')
      return
    }
    if (status && !status.available) {
      openService()
      toast(status.reason, 'warn')
      return
    }
    const paths = await window.fiducia.dialog.openFiles({
      title: 'Open the calibration certificate',
      multiple: false,
      filters: [
        { name: 'Certificate', extensions: ['pdf', 'png', 'jpg', 'jpeg'] },
        { name: 'All files', extensions: ['*'] },
      ],
    })
    if (!paths?.length) return
    setResult(null)
    const response = await call(() => api.certificate.read(paths[0]), { refresh: false })
    if (response?.job) setJobId(response.job.id)
  }

  const busy = job?.status === 'running' || job?.status === 'queued'
  const extraction = result?.extraction
  const check = result?.distortionCheck

  return (
    <section className="section">
      <div className="section__head">
        <span className="section__title">
          Calibration certificate
          <InfoTip>
            <p>Extracts focal length, principal point, fiducial positions and the
            distortion table from a PDF or scanned certificate. Each value is shown
            with its source text for verification before it is applied.</p>
            <p>This is the only feature that uses the network.</p>
          </InfoTip>
        </span>
        <span className="section__rule" />
        <button
          className="btn btn--ghost btn--sm"
          onClick={openService}
          title="Reading service and API key, in Settings"
        >
          Reading service
        </button>
        <button className="btn btn--primary btn--sm" onClick={choose} disabled={busy}>
          {busy ? 'Reading…' : 'Read…'}
        </button>
      </div>

      {status && !status.available && (
        <div className="field__hint" style={{ color: 'var(--signal-warn)' }}>
          {status.reason}
        </div>
      )}

      {!result && !busy && status?.available && (
        <div className="field__hint">
          Reads with {status.services?.[status.provider]?.label}.
        </div>
      )}

      {busy && (
        <div className="field__hint">
          {job.message || 'Reading the certificate…'}
        </div>
      )}

      {result && (
        <>
          <div className="reading__summary">
            <button
              className="btn btn--ghost btn--sm"
              onClick={() => setOpen((v) => !v)}
            >
              {open ? 'Hide values' : 'Show all values'}
            </button>
            <span className="dim data">
              {result.provider === 'offline'
                ? `Read offline, ${result.usage.method}`
                : `${result.usage.inputTokens.toLocaleString()} in, ${result.usage.outputTokens.toLocaleString()} out`}
            </span>
          </div>

          {check?.fitted && (
            <div
              className="readiness"
              style={{
                borderLeftColor:
                  check.verdict === 'good' ? 'var(--signal-good)'
                    : check.verdict === 'check' ? 'var(--signal-warn)'
                    : 'var(--signal-bad)',
              }}
            >
              <div className="readiness__item">
                <span className="dot" />
                <span>{check.note}</span>
              </div>
            </div>
          )}

          {(result.review || []).map((note, index) => (
            <div className="readiness" key={index}
                 style={{ borderLeftColor: 'var(--signal-warn)' }}>
              <div className="readiness__item readiness__item--warning">
                <span className="dot" />
                <span>{note}</span>
              </div>
            </div>
          ))}

          {result.camera.kind === 'digital' ? (
            // A digital frame has no fiducials or distortion table to show;
            // its sensor is what the operator needs to check.
            <div className="measures" style={{ marginBottom: 'var(--step-3)' }}>
              <div className="measure">
                <div className="measure__value" style={{ fontSize: 13 }}>
                  {result.camera.focalMm ? `${result.camera.focalMm} mm` : '—'}
                </div>
                <div className="measure__name">Focal length</div>
              </div>
              <div className="measure">
                <div className="measure__value" style={{ fontSize: 13 }}>
                  {result.camera.pixelPitchMm
                    ? `${+(result.camera.pixelPitchMm * 1000).toFixed(3)} µm` : '—'}
                </div>
                <div className="measure__name">Pixel size</div>
              </div>
              <div className="measure">
                <div className="measure__value" style={{ fontSize: 13 }}>
                  {result.camera.columns && result.camera.rows
                    ? `${result.camera.columns} × ${result.camera.rows}` : '—'}
                </div>
                <div className="measure__name">Sensor, px</div>
              </div>
              <div className="measure">
                <div className="measure__value" style={{ fontSize: 13 }}>
                  {`${(result.camera.ppoXMm ?? 0).toFixed(3)}, ${(result.camera.ppoYMm ?? 0).toFixed(3)}`}
                </div>
                <div className="measure__name">Principal point, mm</div>
              </div>
            </div>
          ) : (
          <div className="measures" style={{ marginBottom: 'var(--step-3)' }}>
            <div className="measure">
              <div className="measure__value" style={{ fontSize: 13 }}>
                {result.camera.focalMm ? `${result.camera.focalMm} mm` : '—'}
              </div>
              <div className="measure__name">Focal length</div>
            </div>
            <div className="measure">
              <div className="measure__value" style={{ fontSize: 13 }}>
                {Object.keys(result.camera.fiducialsMm || {}).length || '—'}
              </div>
              <div className="measure__name">Fiducial marks</div>
            </div>
            <div className="measure">
              <div className="measure__value" style={{ fontSize: 13 }}>
                {(extraction?.distortionTable || []).length || '—'}
              </div>
              <div className="measure__name">Distortion rows</div>
            </div>
            <div className="measure">
              <div className={`measure__value ${
                check?.verdict === 'good' ? 'measure__value--good'
                  : check?.verdict === 'check' ? 'measure__value--warn'
                  : check?.fitted ? 'measure__value--bad' : ''}`}
                   style={{ fontSize: 13 }}>
                {check?.fitted ? `${check.rmsUm.toFixed(2)} µm` : '—'}
              </div>
              <div className="measure__name">Table fit</div>
            </div>
          </div>
          )}

          {open && extraction && (
            <div className="readings">
              <Reading label="Focal length" entry={extraction.focalLength} />
              <Reading label="PPO x" entry={extraction.ppoX} />
              <Reading label="PPO y" entry={extraction.ppoY} />
              <Reading label="PPA x" entry={extraction.ppaX} />
              <Reading label="PPA y" entry={extraction.ppaY} />
              <Reading label="PPS x" entry={extraction.ppsX} />
              <Reading label="PPS y" entry={extraction.ppsY} />
              <Reading label="Pixel pitch" entry={extraction.pixelPitchMm} />
              <Reading label="Columns" entry={extraction.columns} unit="" />
              <Reading label="Rows" entry={extraction.rows} unit="" />

              {(extraction.fiducials || []).length > 0 && (
                <>
                  <div className="reading__group">Fiducial marks (mm)</div>
                  <table className="table">
                    <thead>
                      <tr><th>Position</th><th className="num">x</th>
                          <th className="num">y</th><th>As printed</th></tr>
                    </thead>
                    <tbody>
                      {extraction.fiducials.map((mark) => (
                        <tr key={mark.slot}>
                          <td>{mark.slot.replace(/_/g, ' ')}</td>
                          <td className="num">{mark.x}</td>
                          <td className="num">{mark.y}</td>
                          <td className="dim">{mark.label}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </>
              )}

              {(extraction.distortionTable || []).length > 0 && (
                <>
                  <div className="reading__group">
                    Radial distortion
                    {extraction.distortionUnitsAsPrinted
                      ? ` (printed in ${extraction.distortionUnitsAsPrinted})`
                      : ''}
                  </div>
                  <table className="table">
                    <thead>
                      <tr><th className="num">r (mm)</th><th className="num">Δr (µm)</th>
                          <th className="num">Fitted</th><th className="num">Residual</th></tr>
                    </thead>
                    <tbody>
                      {extraction.distortionTable.map((row, index) => {
                        const sample = check?.samples?.[index]
                        const off = sample && Math.abs(sample.residualUm) > 1.0
                        return (
                          <tr key={index}>
                            <td className="num">{row.radiusMm}</td>
                            <td className="num">{row.distortionUm}</td>
                            <td className="num dim">
                              {sample ? sample.fittedUm.toFixed(2) : '—'}
                            </td>
                            <td className={`num ${off ? 'over' : ''}`}>
                              {sample ? sample.residualUm.toFixed(2) : '—'}
                            </td>
                          </tr>
                        )
                      })}
                    </tbody>
                  </table>
                </>
              )}
            </div>
          )}

          <div style={{ display: 'flex', gap: 6, marginTop: 'var(--step-3)' }}>
            <button
              className="btn btn--primary btn--sm"
              onClick={() => {
                onApply(result.camera)
                setResult(null)
                toast('Certificate values applied. Verify them before solving.', 'good')
              }}
            >
              Apply to camera
            </button>
            <button className="btn btn--ghost btn--sm" onClick={() => setResult(null)}>
              Discard
            </button>
          </div>
        </>
      )}
    </section>
  )
}
