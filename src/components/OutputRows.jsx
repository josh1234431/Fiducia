/**
 * A step's list of generated files: open one on the canvas, in a window of
 * its own, or in the file manager.
 */
export default function OutputRows({ items, step, current, onView, onPopOut }) {
  const canPopOut = !!window.fiducia?.windows
  return (
    <div className="rows">
      {items.map((item) => {
        const viewable = /\.(tif|tiff|img|vrt|jp2)$/i.test(item.path || '')
        const open = current === item.path
        return (
          <div key={item.key} className={`row ${open ? 'row--active' : ''}`}>
            <div className="row__main">
              <div className="row__name truncate" title={item.path}>{item.name}</div>
              {item.meta && <div className="row__meta">{item.meta}</div>}
            </div>
            <div className="row__actions">
              {viewable && (
                <button
                  className={`btn btn--sm ${open ? 'btn--primary' : 'btn--ghost'}`}
                  onClick={() => onView({ path: item.path, label: item.name, step })}
                  title="Show on the canvas"
                >
                  View
                </button>
              )}
              {viewable && canPopOut && (
                <button
                  className="btn btn--ghost btn--sm"
                  onClick={() => onPopOut({ path: item.path, label: item.name, step })}
                  title="Open in a separate window"
                >
                  ⧉
                </button>
              )}
              {window.fiducia?.shell && (
                <button
                  className="btn btn--ghost btn--sm"
                  onClick={() => window.fiducia.shell.showItem(item.path)}
                  title="Show in folder"
                >
                  ↗
                </button>
              )}
            </div>
          </div>
        )
      })}
    </div>
  )
}
