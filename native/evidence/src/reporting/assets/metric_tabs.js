(() => {
  const main = document.querySelector('main');
  const nav = main && main.querySelector('.metric-navigation');
  const toggle = main && main.querySelector('.metric-display-toggle');
  if (!nav || !toggle) return;
  const links = Array.from(nav.querySelectorAll('a[href^="#metric-group-"]'));
  const headings = links.map(link => document.getElementById(link.hash.slice(1)));
  const panels = headings.map(heading => heading && heading.closest('.metric-group'));
  if (links.length < 2 || panels.some(panel => !panel || !main.contains(panel))) return;
  let active = 0;
  let showingAll = false;
  const indexFor = target => panels.findIndex(panel => panel.contains(target));
  const hashTarget = () => {
    const id = window.location.hash.slice(1);
    return /^metric-(group|result)-[0-9]+$/.test(id) ? document.getElementById(id) : null;
  };
  const update = () => {
    if (showingAll) nav.removeAttribute('role');
    else nav.setAttribute('role', 'tablist');
    links.forEach((link, index) => {
      const panel = panels[index];
      panel.hidden = !showingAll && index !== active;
      if (showingAll) {
        ['role', 'aria-selected', 'aria-controls', 'tabindex'].forEach(attr => link.removeAttribute(attr));
        panel.removeAttribute('role');
        panel.setAttribute('aria-labelledby', headings[index].id);
      } else {
        link.id = 'metric-tab-' + (index + 1);
        link.setAttribute('role', 'tab');
        link.setAttribute('aria-selected', String(index === active));
        link.setAttribute('aria-controls', panel.id);
        link.tabIndex = index === active ? 0 : -1;
        panel.setAttribute('role', 'tabpanel');
        panel.setAttribute('aria-labelledby', link.id);
      }
    });
    toggle.textContent = showingAll ? 'Use metric tabs' : 'Show all metrics';
  };
  const select = (index, focusTab) => {
    active = index;
    update();
    if (focusTab) links[index].focus();
  };
  const remember = index => {
    // Fragment replacement remains local, including file:// reports.
    try { window.history.replaceState(null, '', links[index].hash); } catch (_) {}
  };
  links.forEach((link, index) => {
    link.addEventListener('click', event => {
      if (showingAll || event.ctrlKey || event.metaKey || event.shiftKey || event.altKey) return;
      event.preventDefault();
      select(index, true);
      remember(index);
    });
    link.addEventListener('keydown', event => {
      if (showingAll || event.ctrlKey || event.metaKey || event.altKey) return;
      let next;
      if (event.key === 'ArrowRight') next = (index + 1) % links.length;
      else if (event.key === 'ArrowLeft') next = (index + links.length - 1) % links.length;
      else if (event.key === 'Home') next = 0;
      else if (event.key === 'End') next = links.length - 1;
      else return;
      event.preventDefault();
      select(next, true);
      remember(next);
    });
  });
  const followHash = () => {
    const target = hashTarget();
    const index = target ? indexFor(target) : -1;
    if (index < 0) return;
    select(index, false);
    target.focus();
    target.scrollIntoView({block: 'start'});
  };
  main.querySelectorAll('.results-overview a[href^="#metric-result-"], .findings a[href^="#metric-result-"]').forEach(link => {
    link.addEventListener('click', event => {
      if (event.ctrlKey || event.metaKey || event.shiftKey || event.altKey) return;
      const target = document.getElementById(link.hash.slice(1));
      const index = target ? indexFor(target) : -1;
      if (index >= 0) {
        select(index, false);
        target.focus();
      }
    });
  });
  toggle.addEventListener('click', () => {
    showingAll = !showingAll;
    update();
  });
  window.addEventListener('hashchange', followHash);
  const initial = hashTarget();
  const initialIndex = initial ? indexFor(initial) : -1;
  const adverse = panels.findIndex(panel => panel.querySelector('.metric.fail, .metric.insufficient'));
  active = initialIndex >= 0 ? initialIndex : Math.max(0, adverse);
  update();
  toggle.hidden = false;
  if (initialIndex >= 0) followHash();
})();