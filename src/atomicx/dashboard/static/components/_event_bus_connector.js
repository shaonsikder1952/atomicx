/**
 * Event Bus Connector - Component Integration
 *
 * This script connects components to the global event bus from god_mode.html
 * Usage: Include this at the end of each component's <script> section
 */

(function() {
  // Wait for parent event bus to be available
  function connectToEventBus(renderCallback) {
    if (window.parent && window.parent.godModeEventBus) {
      // Subscribe to parent's event bus
      window.parent.godModeEventBus.subscribe(renderCallback);
      console.log('[EVENT-BUS] Component connected to global event bus');
    } else {
      // Retry in 100ms if not available yet
      setTimeout(() => connectToEventBus(renderCallback), 100);
    }
  }

  // Export connector
  window.connectToGodModeEventBus = connectToEventBus;
})();
