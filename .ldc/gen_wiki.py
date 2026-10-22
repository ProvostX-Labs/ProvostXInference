import re, html, pathlib

DOCS = pathlib.Path("docs")

sections = [
    ("contract", "Consumer Contract"),
    ("CAPABILITY", "Capability Matrix"),
    ("CONFIGURATION", "Configuration"),
    ("api-glossary", "API Glossary"),
    ("faq", "FAQ"),
]


CS_KEYWORDS = set("""abstract as async await base bool break byte case catch char checked class const
continue decimal default delegate do double dynamic else enum event explicit extern false finally fixed
float for foreach get goto if implicit in init int interface internal is lock long namespace new null
object operator out override params partial private protected public readonly record ref return sbyte
sealed set short sizeof stackalloc static string struct switch this throw true try typeof uint ulong
unchecked unsafe ushort using var virtual void volatile when where while with yield""".split())

_CS_TOKEN = re.compile(
    r'(?P<c>//[^\n]*|/\*[\s\S]*?\*/)'
    r'|(?P<s>@"(?:[^"]|"")*"|@?"(?:[^"\\\n]|\\.|\n)*")'
    r"|(?P<sq>'(?:[^'\\\n]|\\.)*')"
    r'|(?P<n>\b\d[\d_]*(?:\.[\d_]+)?(?:[eE][+-]?\d+)?[fFmMdDlLuU]*\b)'
    r'|(?P<i>[A-Za-z_][A-Za-z0-9_]*)'
)

def highlight_cs(text):
    out, pos = [], 0
    for m in _CS_TOKEN.finditer(text):
        if m.start() > pos:
            out.append(html.escape(text[pos:m.start()]))
        t = m.lastgroup
        raw = m.group(0)
        esc = html.escape(raw)
        if t == "c":
            out.append('<span class="tok-c">%s</span>' % esc)
        elif t in ("s", "sq"):
            out.append('<span class="tok-s">%s</span>' % esc)
        elif t == "n":
            out.append('<span class="tok-n">%s</span>' % esc)
        else:
            out.append('<span class="tok-k">%s</span>' % esc if raw in CS_KEYWORDS else esc)
        pos = m.end()
    out.append(html.escape(text[pos:]))
    return "".join(out)

def md_to_html(md):
    lines = md.splitlines()
    out, i = [], 0
    while i < len(lines):
        ln = lines[i]
        if ln.startswith("```"):
            lang = ln[3:].strip()
            i += 1
            buf = []
            while i < len(lines) and not lines[i].startswith("```"):
                buf.append(lines[i]); i += 1
            i += 1
            out.append('<pre class="code" data-lang="%s">%s</pre>' % (html.escape(lang), highlight_cs("\n".join(buf))))
            continue
        m = re.match(r"^(#{1,4}) (.*)$", ln)
        if m:
            lvl = len(m.group(1)); txt = m.group(2).strip()
            slug = re.sub(r"[^a-z0-9]+", "-", txt.lower()).strip("-")
            out.append('<h%d id="%s">%s</h%d>' % (lvl, slug, inline(txt), lvl))
            i += 1; continue
        if ln.strip() == "":
            i += 1; continue
        if re.match(r"^\s*\|", ln):
            rows = []
            while i < len(lines) and re.match(r"^\s*\|", lines[i]):
                rows.append(lines[i]); i += 1
            out.append(table(rows)); continue
        if re.match(r"^\s*[-*] ", ln):
            items = []
            while i < len(lines) and re.match(r"^\s*[-*] ", lines[i]):
                items.append(re.sub(r"^\s*[-*] ", "", lines[i])); i += 1
            out.append("<ul>" + "".join("<li>%s</li>" % inline(x) for x in items) + "</ul>")
            continue
        if re.match(r"^\s*\d+\. ", ln):
            items = []
            while i < len(lines) and re.match(r"^\s*\d+\. ", lines[i]):
                items.append(re.sub(r"^\s*\d+\. ", "", lines[i])); i += 1
            out.append("<ol>" + "".join("<li>%s</li>" % inline(x) for x in items) + "</ol>")
            continue
        buf = []
        while i < len(lines) and lines[i].strip() != "" and not lines[i].lstrip().startswith(("```", "#", "|", "- ", "* ")) and not re.match(r"^\s*\d+\. ", lines[i]):
            buf.append(lines[i]); i += 1
        out.append("<p>%s</p>" % inline(" ".join(buf)))
    return "\n".join(out)

def inline(t):
    t = html.escape(t)
    t = re.sub(r"`([^`]+)`", r"<code>\1</code>", t)
    t = re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", t)
    def link(m):
        text, href = m.group(1), m.group(2)
        h = re.sub(r"^(\./)?(docs/)?", "", href.strip())
        if ".md#" in h:
            doc, anchor = h.split(".md#", 1)
            return '<a href="#pane-%s" data-doc="%s" data-anchor="%s">%s</a>' % (doc, doc, anchor, text)
        if h.endswith(".md"):
            return '<a href="#pane-%s">%s</a>' % (h[:-3], text)
        return '<a href="%s">%s</a>' % (href, text)
    t = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", link, t)
    return t

def cells(r):
    return [c.strip() for c in r.strip().strip("|").split("|")]

def table(rows):
    has_head = len(rows) > 1 and set(rows[1].replace("|", "").replace("-", "").strip()) == set()
    h = ""
    data = rows
    if has_head:
        h = "<thead><tr>" + "".join("<th>%s</th>" % inline(c) for c in cells(rows[0])) + "</tr></thead>"
        data = rows[2:]
    body = "".join("<tr>" + "".join("<td>%s</td>" % inline(c) for c in cells(r)) + "</tr>" for r in data)
    return "<table>%s<tbody>%s</tbody></table>" % (h, body)

nav_items, panes = [], []
for slug, title in sections:
    md = (DOCS / (slug + ".md")).read_text()
    body = md_to_html(md)
    panes.append('<section class="pane" id="pane-%s" data-title="%s">%s</section>' % (slug, html.escape(title, quote=True), body))
    nav_items.append('<a class="nav-link" href="#pane-%s" data-target="%s">%s</a>' % (slug, slug, html.escape(title)))

CSS = """
:root {
  --bg: #ffffff; --fg: #334155; --muted: #5A6B80; --border: #E2E8F0;
  --accent: #6366F1; --accent-soft: #EEF2FF; --sidebar: #F8FAFC; --code-bg: #EEF1F6;
  --tok-k: #00539B; --tok-s: #A34E00; --tok-n: #067D62; --tok-c: #456A45;
}
@media (prefers-color-scheme: dark) {
  :root {
    --bg: #0E1E3E; --fg: #E6EBF2; --muted: #A4B8D8; --border: #28406E;
    --accent: #5B9CFF; --accent-soft: #152B54; --sidebar: #152B54; --code-bg: #152B54;
    --tok-k: #569CD6; --tok-s: #CE9179; --tok-n: #B5CEA8; --tok-c: #7FA88C;
  }
}
* { box-sizing: border-box; }
html, body { margin: 0; padding: 0; }
body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif;
  background: var(--bg); color: var(--fg); line-height: 1.6; font-size: 16px; }
a { color: var(--accent); text-decoration: none; }
a:hover { text-decoration: underline; }
.layout { display: flex; min-height: 100vh; }
aside {
  width: 280px; flex-shrink: 0; background: var(--sidebar); border-right: 1px solid var(--border);
  position: sticky; top: 0; height: 100vh; overflow-y: auto; padding: 24px 0;
}
.brand { font-weight: 700; font-size: 17px; padding: 0 24px 4px; }
.brand-sub { font-size: 12px; color: var(--muted); padding: 0 24px 16px; }
.search-wrap { padding: 0 16px 12px; }
#search { width: 100%; padding: 8px 10px; border: 1px solid var(--border); border-radius: 6px;
  background: var(--bg); color: var(--fg); font-size: 14px; }
nav { display: flex; flex-direction: column; }
.nav-link { padding: 7px 24px; color: var(--fg); font-size: 14px; border-left: 3px solid transparent; }
.nav-link:hover { background: var(--accent-soft); text-decoration: none; }
.nav-link.active { border-left-color: var(--accent); background: var(--accent-soft); font-weight: 600; }
main { flex: 1; min-width: 0; padding: 40px 56px 80px; max-width: 980px; }
.pane { display: none; }
.pane.visible { display: block; }
h1 { font-size: 30px; border-bottom: 1px solid var(--border); padding-bottom: 10px; }
h2 { font-size: 22px; margin-top: 36px; border-bottom: 1px solid var(--border); padding-bottom: 6px; }
h3 { font-size: 17px; margin-top: 28px; }
code { background: var(--code-bg); padding: .15em .35em; border-radius: 5px; font-size: 87%;
  font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; }
pre.code { background: var(--code-bg); color: var(--fg); padding: 14px 16px; border-radius: 8px;
  overflow-x: auto; font-size: 13.5px; line-height: 1.5; border: 1px solid var(--border);
  font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; white-space: pre-wrap; }
pre.code code { background: none; padding: 0; font-size: inherit; }
pre.code .tok-k { color: var(--tok-k); }
pre.code .tok-s { color: var(--tok-s); }
pre.code .tok-n { color: var(--tok-n); }
pre.code .tok-c { color: var(--tok-c); font-style: italic; }
table { border-collapse: collapse; width: 100%; margin: 14px 0; font-size: 14.5px; }
th, td { border: 1px solid var(--border); padding: 7px 12px; text-align: left; vertical-align: top; }
th { background: var(--sidebar); font-weight: 600; }
tbody tr:nth-child(2n) td { background: var(--sidebar); }
li { margin: 4px 0; }
.match-hidden { display: none !important; }
.copy-btn { position: absolute; top: 6px; right: 6px; padding: 2px 10px; font-size: 12px;
  line-height: 1.4; color: var(--muted); background: var(--bg); border: 1px solid var(--border);
  border-radius: 5px; opacity: 0; transition: opacity .15s ease; cursor: pointer; z-index: 5; }
pre.code { position: relative; }
pre.code:hover > .copy-btn, .copy-btn.copied { opacity: 1; }
.copy-btn:hover, .copy-btn.copied { color: var(--accent); border-color: var(--accent); font-weight: 600; }
.copy-btn:focus-visible { opacity: 1; }
.footnote { color: var(--muted); font-size: 13px; margin-top: 60px; border-top: 1px solid var(--border); padding-top: 16px; }
@media (max-width: 900px) {
  .layout { flex-direction: column; }
  aside { position: static; width: 100%; height: auto; border-right: none; border-bottom: 1px solid var(--border); }
  main { padding: 24px 20px 60px; }
}
"""

JS = """
(function () {
  var links = Array.prototype.slice.call(document.querySelectorAll('.nav-link'));
  var panes = Array.prototype.slice.call(document.querySelectorAll('.pane'));

  function show(slug, push) {
    panes.forEach(function (p) { p.classList.toggle('visible', p.id === 'pane-' + slug); });
    links.forEach(function (l) { l.classList.toggle('active', l.getAttribute('data-target') === slug); });
    if (push !== false && history.replaceState) { history.replaceState(null, '', '#' + slug); }
    window.scrollTo(0, 0);
  }

  links.forEach(function (l) {
    l.addEventListener('click', function (e) {
      e.preventDefault();
      show(l.getAttribute('data-target'));
    });
  });

  // In-page anchors scroll within the active pane.
  document.getElementById('main').addEventListener('click', function (e) {
    var a = e.target.closest ? e.target.closest('a') : null;
    if (!a || a.classList.contains('nav-link')) return;
    var href = a.getAttribute('href') || '';
    var doc = a.getAttribute('data-doc'), anchor = a.getAttribute('data-anchor');
    if (doc && anchor) {
      e.preventDefault();
      show(doc);
      var t = document.getElementById(anchor);
      if (t) t.scrollIntoView();
      return;
    }
    if (href.indexOf('#') === 0) {
      var id = href.slice(1);
      if (id.indexOf('pane-') === 0) { e.preventDefault(); show(id.slice(5)); return; }
      var el = document.getElementById(id);
      if (el) { e.preventDefault(); el.scrollIntoView(); }
    }
  });

  var input = document.getElementById('search');
  input.addEventListener('input', function () {
    var q = input.value.trim().toLowerCase();
    panes.forEach(function (p) {
      var title = (p.getAttribute('data-title') || '').toLowerCase();
      var heads = Array.prototype.slice.call(p.querySelectorAll('h1, h2, h3, h4'));
      if (!q) {
        p.classList.remove('match-hidden');
        heads.forEach(function (h) { h.classList.remove('match-hidden'); });
        Array.prototype.slice.call(p.querySelectorAll('[data-search-hidden]')).forEach(function (el) {
          el.removeAttribute('data-search-hidden'); el.classList.remove('match-hidden');
        });
        return;
      }
      var titleMatch = title.indexOf(q) !== -1;
      var anyHeadMatch = false;
      heads.forEach(function (h) {
        var m = h.textContent.toLowerCase().indexOf(q) !== -1;
        h.classList.toggle('match-hidden', !m);
        if (m) anyHeadMatch = true;
      });
      p.classList.toggle('match-hidden', !(titleMatch || anyHeadMatch));
      if (titleMatch) {
        heads.forEach(function (h) { h.classList.remove('match-hidden'); });
      }
      var blocks = Array.prototype.slice.call(p.children);
      var currentMatch = titleMatch;
      blocks.forEach(function (b) {
        if (/^H[1-4]$/.test(b.tagName)) {
          currentMatch = !b.classList.contains('match-hidden');
          if (b.hasAttribute('data-search-hidden')) {
            b.removeAttribute('data-search-hidden'); b.classList.remove('match-hidden');
          }
          return;
        }
        if (currentMatch) {
          if (b.hasAttribute('data-search-hidden')) {
            b.removeAttribute('data-search-hidden'); b.classList.remove('match-hidden');
          }
        } else {
          b.setAttribute('data-search-hidden', '1'); b.classList.add('match-hidden');
        }
      });
    });
    var first = null;
    for (var i = 0; i < panes.length; i++) {
      if (!panes[i].classList.contains('match-hidden')) { first = panes[i]; break; }
    }
    if (first) {
      var slug = first.id.replace(/^pane-/, '');
      links.forEach(function (l) { l.classList.toggle('active', l.getAttribute('data-target') === slug); });
    }
  });

  // Code copy buttons (corporate design: standard copy on every code block)
  Array.prototype.slice.call(document.querySelectorAll('pre.code')).forEach(function (pre) {
    var btn = document.createElement('button');
    btn.type = 'button';
    btn.className = 'copy-btn';
    btn.setAttribute('aria-label', 'Copy code to clipboard');
    btn.textContent = 'Copy';
    btn.addEventListener('click', function () {
      function done(ok) {
        btn.textContent = ok ? 'Copied \u2713' : 'Copy failed';
        btn.classList.add('copied');
        setTimeout(function () { btn.textContent = 'Copy'; btn.classList.remove('copied'); }, 1600);
      }
      function fallback() {
        var ta = document.createElement('textarea');
        ta.value = pre.textContent;
        ta.style.position = 'fixed'; ta.style.opacity = '0';
        document.body.appendChild(ta); ta.select();
        var ok = false;
        try { ok = document.execCommand('copy'); } catch (e) { ok = false; }
        document.body.removeChild(ta);
        return ok;
      }
      if (navigator.clipboard && navigator.clipboard.writeText) {
        navigator.clipboard.writeText(pre.textContent).then(function () { done(true); }, function () { done(fallback()); });
      } else { done(fallback()); }
    });
    pre.appendChild(btn);
  });

  var initial = (location.hash || '').replace(/^#/, '');
  var valid = false;
  for (var j = 0; j < panes.length; j++) { if (panes[j].id === 'pane-' + initial) { valid = true; break; } }
  show(valid ? initial : panes[0].id.replace(/^pane-/, ''), false);
})();
"""

TEMPLATE = pathlib.Path(".ldc/wiki_template.html").read_text()
page = TEMPLATE.replace("@@CSS@@", CSS).replace("@@JS@@", JS).replace("@@NAV@@", "\n".join(nav_items)).replace("@@CONTENT@@", "\n".join(panes))
(DOCS / "wiki.html").write_text(page)
print("written", len(page))
