/**
 * Choosing a coordinate system in two short steps instead of one long list.
 *
 * First the family (South African Lo zones, UTM, or other), then, for the Lo
 * zones, the zone and the form (orientation and datum), and for UTM the zone
 * and hemisphere. The value is still a single preset key, so nothing else in
 * Fiducia changes.
 */

const LAST_KEY = 'fiducia.lastProjection'

// On Hartebeesthoek94 unless named; the datum is also shown under the picker.
const LO_FORMS = [
  { suffix: '', label: 'South-oriented Y/X' },
  { suffix: '_EN', label: 'North-oriented E/N' },
  { suffix: '_WGS84', label: 'South-oriented Y/X, WGS84' },
  { suffix: '_CAPE', label: 'Cape datum (legacy)' },
]

function parse(key) {
  const lo = /^ZALO(\d+)(_EN|_WGS84|_CAPE)?$/.exec(key || '')
  if (lo) return { family: 'lo', zone: Number(lo[1]), form: lo[2] || '' }
  const utm = /^UTM(\d+)([NS])$/.exec(key || '')
  if (utm) return { family: 'utm', zone: Number(utm[1]), hemisphere: utm[2] }
  return { family: key ? 'other' : '' }
}

/** The last output projection chosen on this computer, for new projects. */
export function lastProjection() {
  try { return localStorage.getItem(LAST_KEY) || '' } catch { return '' }
}

export function rememberProjection(key) {
  try { if (key) localStorage.setItem(LAST_KEY, key) } catch { /* optional */ }
}

export default function ProjectionPicker({ id, value, presets, onChange }) {
  const current = parse(value)
  const keys = new Set(presets.map((p) => p.key))
  const loZones = [...new Set(presets.map((p) => parse(p.key)).filter((p) => p.family === 'lo')
    .map((p) => p.zone))].sort((a, b) => a - b)
  const utmZones = [...new Set(presets.map((p) => parse(p.key)).filter((p) => p.family === 'utm')
    .map((p) => p.zone))].sort((a, b) => a - b)
  const others = presets.filter((p) => parse(p.key).family === 'other')
  const known = !value || keys.has(value)

  function chooseFamily(family) {
    if (!family) { onChange(''); return }
    // Keep what can be kept: the last choice in that family, else its usual default.
    const last = parse(lastProjection())
    if (family === 'lo') {
      const zone = last.family === 'lo' ? last.zone : (loZones.includes(19) ? 19 : loZones[0])
      const form = last.family === 'lo' ? last.form : ''
      onChange(`ZALO${zone}${form}`)
    } else if (family === 'utm') {
      const zone = last.family === 'utm' ? last.zone : (utmZones.includes(34) ? 34 : utmZones[0])
      onChange(`UTM${zone}${last.family === 'utm' ? last.hemisphere : 'S'}`)
    } else {
      onChange(others[0]?.key || '')
    }
  }

  return (
    <div className="crs-picker">
      <select
        id={id}
        value={current.family}
        onChange={(event) => chooseFamily(event.target.value)}
        aria-label="Coordinate system family"
      >
        <option value="">Not set</option>
        {loZones.length > 0 && <option value="lo">South Africa, Lo zones (Gauss Conform)</option>}
        {utmZones.length > 0 && <option value="utm">UTM, WGS84</option>}
        <option value="other">Other</option>
      </select>

      {current.family === 'lo' && (
        <div className="crs-picker__row">
          <select
            value={current.zone}
            onChange={(event) => onChange(`ZALO${event.target.value}${current.form}`)}
            aria-label="Lo zone"
          >
            {loZones.map((zone) => <option key={zone} value={zone}>Lo{zone}</option>)}
          </select>
          <select
            value={current.form}
            onChange={(event) => onChange(`ZALO${current.zone}${event.target.value}`)}
            aria-label="Orientation and datum"
          >
            {LO_FORMS.filter((form) => keys.has(`ZALO${current.zone}${form.suffix}`))
              .map((form) => <option key={form.suffix} value={form.suffix}>{form.label}</option>)}
          </select>
        </div>
      )}

      {current.family === 'utm' && (
        <div className="crs-picker__row">
          <select
            value={current.zone}
            onChange={(event) => onChange(`UTM${event.target.value}${current.hemisphere}`)}
            aria-label="UTM zone"
          >
            {utmZones.map((zone) => <option key={zone} value={zone}>Zone {zone}</option>)}
          </select>
          <select
            value={current.hemisphere}
            onChange={(event) => onChange(`UTM${current.zone}${event.target.value}`)}
            aria-label="Hemisphere"
          >
            <option value="S">South</option>
            <option value="N">North</option>
          </select>
        </div>
      )}

      {current.family === 'other' && (
        <div className="crs-picker__row">
          <select value={value} onChange={(event) => onChange(event.target.value)}
                  aria-label="Coordinate system">
            {!known && <option value={value}>{value}</option>}
            {others.map((preset) => (
              <option key={preset.key} value={preset.key}>{preset.label}</option>
            ))}
          </select>
        </div>
      )}
    </div>
  )
}
