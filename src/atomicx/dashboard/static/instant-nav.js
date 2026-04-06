/**
 * INSTANT NAVIGATION SYSTEM
 *
 * Provides instant page transitions for AtomicX dashboard:
 * 1. Prefetch on hover (65ms delay to avoid false positives)
 * 2. Prefetch on touchstart for mobile
 * 3. Cache fetched pages in memory
 * 4. Instant swap on navigation (no full page reload)
 * 5. Preserve scroll position
 * 6. Update URL with pushState
 *
 * Inspired by instant.page and Turbo, optimized for trading dashboards.
 */

class InstantNav {
  constructor() {
    this.cache = new Map();
    this.prefetchDelay = 65; // ms - sweet spot for intent detection
    this.prefetchTimer = null;
    this.currentUrl = window.location.href;
    this.scrollPositions = new Map();
    this.isNavigating = false;

    this.init();
  }

  init() {
    // Prefetch on hover
    document.addEventListener('mouseover', this.onHover.bind(this), { passive: true });

    // Prefetch on touchstart (mobile)
    document.addEventListener('touchstart', this.onTouchStart.bind(this), { passive: true });

    // Intercept clicks
    document.addEventListener('click', this.onClick.bind(this));

    // Handle back/forward
    window.addEventListener('popstate', this.onPopState.bind(this));

    // Save scroll position before navigation
    window.addEventListener('beforeunload', () => {
      this.saveScrollPosition(this.currentUrl);
    });

    console.log('[INSTANT-NAV] ⚡ Instant navigation enabled');
  }

  onHover(event) {
    if (this.isNavigating) return;

    const link = event.target.closest('a');
    if (!this.shouldPrefetch(link)) return;

    // Delay to avoid prefetching on accidental hovers
    clearTimeout(this.prefetchTimer);
    this.prefetchTimer = setTimeout(() => {
      this.prefetch(link.href);
    }, this.prefetchDelay);
  }

  onTouchStart(event) {
    if (this.isNavigating) return;

    const link = event.target.closest('a');
    if (!this.shouldPrefetch(link)) return;

    // On mobile, prefetch immediately on touch
    this.prefetch(link.href);
  }

  onClick(event) {
    // Don't intercept if modifier keys pressed (Cmd+Click, Ctrl+Click, etc.)
    if (event.metaKey || event.ctrlKey || event.shiftKey || event.altKey) return;
    if (event.button !== 0) return; // Only left click

    const link = event.target.closest('a');
    if (!this.shouldIntercept(link)) return;

    event.preventDefault();
    this.navigate(link.href);
  }

  shouldPrefetch(link) {
    if (!link || !link.href) return false;

    // Only prefetch same-origin links
    if (link.origin !== window.location.origin) return false;

    // Don't prefetch if already in cache
    if (this.cache.has(link.href)) return false;

    // Don't prefetch API endpoints
    if (link.href.includes('/api/')) return false;

    // Don't prefetch downloads
    if (link.hasAttribute('download')) return false;

    // Don't prefetch if explicitly disabled
    if (link.hasAttribute('data-no-instant')) return false;

    return true;
  }

  shouldIntercept(link) {
    if (!link || !link.href) return false;

    // Only same-origin
    if (link.origin !== window.location.origin) return false;

    // Don't intercept external links
    if (link.target === '_blank') return false;

    // Don't intercept API calls
    if (link.href.includes('/api/')) return false;

    // Don't intercept downloads
    if (link.hasAttribute('download')) return false;

    // Don't intercept if explicitly disabled
    if (link.hasAttribute('data-no-instant')) return false;

    return true;
  }

  async prefetch(url) {
    if (this.cache.has(url)) return;

    try {
      console.log(`[INSTANT-NAV] 📥 Prefetching: ${url}`);

      const response = await fetch(url, {
        credentials: 'same-origin',
        headers: {
          'X-Instant-Prefetch': 'true'
        }
      });

      if (!response.ok) return;

      const html = await response.text();
      const parser = new DOMParser();
      const doc = parser.parseFromString(html, 'text/html');

      // Cache the parsed document
      this.cache.set(url, {
        html: html,
        doc: doc,
        timestamp: Date.now()
      });

      console.log(`[INSTANT-NAV] ✅ Cached: ${url}`);
    } catch (error) {
      console.warn(`[INSTANT-NAV] ❌ Prefetch failed: ${url}`, error);
    }
  }

  async navigate(url) {
    if (this.isNavigating) return;
    if (url === this.currentUrl) return;

    this.isNavigating = true;
    console.log(`[INSTANT-NAV] 🚀 Navigating to: ${url}`);

    try {
      // Save current scroll position
      this.saveScrollPosition(this.currentUrl);

      // Try to get from cache
      let cached = this.cache.get(url);

      // If not cached, fetch now
      if (!cached) {
        const response = await fetch(url, {
          credentials: 'same-origin'
        });

        if (!response.ok) {
          // Fallback to traditional navigation
          window.location.href = url;
          return;
        }

        const html = await response.text();
        const parser = new DOMParser();
        const doc = parser.parseFromString(html, 'text/html');

        cached = { html, doc, timestamp: Date.now() };
        this.cache.set(url, cached);
      }

      // Perform instant swap
      this.swap(cached.doc, url);

    } catch (error) {
      console.error('[INSTANT-NAV] ❌ Navigation failed:', error);
      // Fallback to traditional navigation
      window.location.href = url;
    } finally {
      this.isNavigating = false;
    }
  }

  swap(newDoc, url) {
    // Extract title
    const newTitle = newDoc.querySelector('title')?.textContent || '';

    // Find main content container (customize this selector for your dashboard)
    const mainSelectors = [
      'main',
      '[data-instant-main]',
      '#content',
      '.dashboard-content',
      'body'
    ];

    let oldMain = null;
    let newMain = null;

    for (const selector of mainSelectors) {
      oldMain = document.querySelector(selector);
      newMain = newDoc.querySelector(selector);
      if (oldMain && newMain) break;
    }

    if (!oldMain || !newMain) {
      console.warn('[INSTANT-NAV] ⚠️ Main content not found, falling back');
      window.location.href = url;
      return;
    }

    // Fade out animation (optional)
    oldMain.style.opacity = '0';
    oldMain.style.transition = 'opacity 150ms ease-out';

    setTimeout(() => {
      // Swap content
      oldMain.innerHTML = newMain.innerHTML;

      // Update title
      document.title = newTitle;

      // Update URL
      history.pushState({ url: url }, newTitle, url);
      this.currentUrl = url;

      // Restore or reset scroll position
      const savedScroll = this.scrollPositions.get(url);
      if (savedScroll) {
        window.scrollTo(savedScroll.x, savedScroll.y);
      } else {
        window.scrollTo(0, 0);
      }

      // Fade in
      oldMain.style.opacity = '1';

      // Re-run scripts if needed (for components with initialization)
      this.executeScripts(newMain);

      // Dispatch custom event for other code to react
      window.dispatchEvent(new CustomEvent('instant-nav:complete', {
        detail: { url, timestamp: Date.now() }
      }));

      console.log('[INSTANT-NAV] ✅ Navigation complete');
    }, 150);
  }

  executeScripts(container) {
    // Find and execute inline scripts in the new content
    const scripts = container.querySelectorAll('script');
    scripts.forEach(oldScript => {
      const newScript = document.createElement('script');
      Array.from(oldScript.attributes).forEach(attr => {
        newScript.setAttribute(attr.name, attr.value);
      });
      newScript.textContent = oldScript.textContent;
      oldScript.parentNode.replaceChild(newScript, oldScript);
    });
  }

  onPopState(event) {
    if (event.state && event.state.url) {
      this.navigate(event.state.url);
    }
  }

  saveScrollPosition(url) {
    this.scrollPositions.set(url, {
      x: window.scrollX,
      y: window.scrollY
    });
  }

  clearCache() {
    this.cache.clear();
    console.log('[INSTANT-NAV] 🗑️ Cache cleared');
  }

  getCacheSize() {
    return this.cache.size;
  }

  getCacheStats() {
    return {
      size: this.cache.size,
      urls: Array.from(this.cache.keys()),
      totalBytes: Array.from(this.cache.values())
        .reduce((sum, item) => sum + item.html.length, 0)
    };
  }
}

// Auto-initialize
if (typeof window !== 'undefined') {
  window.instantNav = new InstantNav();

  // Expose for debugging
  window.addEventListener('keydown', (e) => {
    // Ctrl+Shift+I = show cache stats
    if (e.ctrlKey && e.shiftKey && e.key === 'I') {
      console.log('[INSTANT-NAV] Cache stats:', window.instantNav.getCacheStats());
    }
  });
}

export default InstantNav;
