// Dev notes: read notes.md and turn it into posts.
//
// Each post starts with "## Title" and a "date: YYYY-MM-DD" line; everything
// until the next "## " is the body, in a small, safe subset of Markdown. All
// text is escaped first, so a note can never inject markup into the page.
(function () {
  function escapeHtml(s) {
    return s.replace(/[&<>"']/g, function (c) {
      return { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c];
    });
  }
  function safeUrl(url) {
    url = url.trim();
    return /^(https?:\/\/|mailto:|#|[\w./-]+(#[\w-]*)?$)/i.test(url) && !/^javascript:/i.test(url) ? url : '#';
  }
  // Inline: images, links, code, bold, italic. Input is already escaped.
  function inline(text) {
    var codes = [];
    text = text.replace(/`([^`]+)`/g, function (_, c) { codes.push(c); return '\u0000' + (codes.length - 1) + '\u0000'; });
    text = text
      .replace(/!\[([^\]]*)\]\(([^)\s]+)\)/g, function (_, alt, src) {
        return '<img src="' + safeUrl(src) + '" alt="' + alt + '" loading="lazy">';
      })
      .replace(/\[([^\]]+)\]\(([^)\s]+)\)/g, function (_, label, href) {
        var url = safeUrl(href), external = /^https?:/i.test(url);
        return '<a href="' + url + '"' + (external ? ' target="_blank" rel="noreferrer"' : '') + '>' + label + '</a>';
      })
      .replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>')
      .replace(/(^|[^*])\*([^*\s][^*]*)\*/g, '$1<em>$2</em>');
    return text.replace(/\u0000(\d+)\u0000/g, function (_, i) { return '<code>' + codes[+i] + '</code>'; });
  }
  function render(markdown) {
    var lines = escapeHtml(markdown).split(/\r?\n/), out = [], para = [], list = null, quote = [], code = null;
    function flush() {
      if (para.length) { out.push('<p>' + inline(para.join(' ')) + '</p>'); para = []; }
      if (list) { out.push('<' + list.tag + '>' + list.items.map(function (i) { return '<li>' + inline(i) + '</li>'; }).join('') + '</' + list.tag + '>'); list = null; }
      if (quote.length) { out.push('<blockquote><p>' + inline(quote.join(' ')) + '</p></blockquote>'); quote = []; }
    }
    lines.forEach(function (line) {
      if (code !== null) {
        if (/^```/.test(line)) { out.push('<pre><code>' + code.join('\n') + '</code></pre>'); code = null; }
        else code.push(line);
        return;
      }
      var m;
      if (/^```/.test(line)) { flush(); code = []; }
      else if (!line.trim()) flush();
      else if ((m = line.match(/^(#{3,4})\s+(.*)$/))) { flush(); out.push('<h' + m[1].length + '>' + inline(m[2]) + '</h' + m[1].length + '>'); }
      else if ((m = line.match(/^\s*[-*]\s+(.*)$/))) { if (para.length || quote.length || (list && list.tag !== 'ul')) flush(); (list = list || { tag: 'ul', items: [] }).items.push(m[1]); }
      else if ((m = line.match(/^\s*\d+[.)]\s+(.*)$/))) { if (para.length || quote.length || (list && list.tag !== 'ol')) flush(); (list = list || { tag: 'ol', items: [] }).items.push(m[1]); }
      else if ((m = line.match(/^&gt;\s?(.*)$/))) { if (para.length || list) flush(); quote.push(m[1]); }
      else if (list && /^\s{2,}\S/.test(line)) list.items[list.items.length - 1] += ' ' + line.trim();
      else { if (list || quote.length) flush(); para.push(line.trim()); }
    });
    if (code !== null) out.push('<pre><code>' + code.join('\n') + '</code></pre>');
    flush();
    return out.join('\n');
  }
  function plain(markdown) {
    return markdown
      .replace(/!\[[^\]]*\]\([^)]*\)/g, '')
      .replace(/\[([^\]]+)\]\([^)]*\)/g, '$1')
      .replace(/[*`>#]/g, '')
      .replace(/\s+/g, ' ').trim();
  }
  function slugify(s) {
    return s.toLowerCase().replace(/[^a-z0-9]+/g, '-').replace(/^-+|-+$/g, '').slice(0, 60) || 'note';
  }
  var MONTHS = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August',
                'September', 'October', 'November', 'December'];
  function dateText(iso) {
    var m = /^(\d{4})-(\d{2})-(\d{2})$/.exec(iso || '');
    return m ? (+m[3]) + ' ' + MONTHS[+m[2] - 1] + ' ' + m[1] : (iso || '');
  }

  function parse(text) {
    text = text.replace(/<!--[\s\S]*?-->/g, '');
    var posts = [], seen = {};
    text.split(/^##\s+/m).slice(1).forEach(function (chunk) {
      var lines = chunk.split(/\r?\n/);
      var title = lines.shift().trim();
      var meta = {};
      while (lines.length && /^\s*(date|tags)\s*:/i.test(lines[0])) {
        var kv = lines.shift().split(':');
        meta[kv[0].trim().toLowerCase()] = kv.slice(1).join(':').trim();
      }
      var body = lines.join('\n').trim();
      var firstPara = (body.split(/\n\s*\n/)[0] || '');
      var slug = slugify(title), n = 2;
      while (seen[slug]) slug = slugify(title) + '-' + n++;
      seen[slug] = true;
      posts.push({
        title: title, slug: slug, date: meta.date || '', dateText: dateText(meta.date),
        tags: meta.tags ? meta.tags.split(',').map(function (t) { return t.trim(); }).filter(Boolean) : [],
        excerpt: plain(firstPara), html: render(body), titleHtml: escapeHtml(title)
      });
    });
    // Newest first, whatever order they were written in.
    return posts.sort(function (a, b) { return (b.date || '').localeCompare(a.date || ''); });
  }

  var cached = null;
  window.FiduciaNotes = {
    load: function () {
      if (!cached) {
        cached = fetch('notes.md', { cache: 'no-cache' })
          .then(function (r) { if (!r.ok) throw new Error(r.status); return r.text(); })
          .then(parse);
      }
      return cached;
    },
    escape: escapeHtml
  };
})();
