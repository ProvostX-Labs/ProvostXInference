// ProvostX corporate design — code copy buttons (2026-10-13)
// Overrides the default template's main.js (a signature-only no-op file).
// Adds a standard copy button to every code block, both themes.
(function () {
  'use strict';
  function addButtons() {
    var pres = Array.prototype.slice.call(document.querySelectorAll('pre'));
    pres.forEach(function (pre) {
      if (pre.querySelector('.pvx-copy-btn')) return;
      var btn = document.createElement('button');
      btn.type = 'button';
      btn.className = 'pvx-copy-btn';
      btn.setAttribute('aria-label', 'Copy code to clipboard');
      btn.textContent = 'Copy';
      btn.addEventListener('click', function () {
        var text = pre.textContent;
        function done(ok) {
          btn.textContent = ok ? 'Copied \u2713' : 'Copy failed';
          btn.classList.add('pvx-copied');
          setTimeout(function () {
            btn.textContent = 'Copy';
            btn.classList.remove('pvx-copied');
          }, 1600);
        }
        if (navigator.clipboard && navigator.clipboard.writeText) {
          navigator.clipboard.writeText(text).then(function () { done(true); }, function () {
            done(fallbackCopy(text));
          });
        } else {
          done(fallbackCopy(text));
        }
      });
      pre.style.position = 'relative';
      pre.appendChild(btn);
    });
  }
  function fallbackCopy(text) {
    var ta = document.createElement('textarea');
    ta.value = text;
    ta.style.position = 'fixed';
    ta.style.opacity = '0';
    document.body.appendChild(ta);
    ta.select();
    var ok = false;
    try { ok = document.execCommand('copy'); } catch (e) { ok = false; }
    document.body.removeChild(ta);
    return ok;
  }
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', addButtons);
  } else {
    addButtons();
  }
})();
