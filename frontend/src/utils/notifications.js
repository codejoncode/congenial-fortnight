/**
 * Browser push notification utilities.
 * Registers the service worker and provides helpers for showing local
 * notifications — no external push service needed for local/private use.
 */

let _swRegistration = null;

export async function registerServiceWorker() {
  if (!('serviceWorker' in navigator) || !('Notification' in window)) return null;
  try {
    _swRegistration = await navigator.serviceWorker.register('/sw.js');
    return _swRegistration;
  } catch (err) {
    console.warn('Service worker registration failed:', err);
    return null;
  }
}

export async function requestNotificationPermission() {
  if (!('Notification' in window)) return 'denied';
  if (Notification.permission === 'granted') return 'granted';
  const result = await Notification.requestPermission();
  return result;
}

/**
 * Show a local browser notification for a new signal.
 * Falls back to Notification API directly if SW not registered.
 */
export function showSignalNotification(signal) {
  const label = signal.signal === 'bullish' ? 'BUY' : signal.signal === 'bearish' ? 'SELL' : 'WAIT';
  const prob  = typeof signal.probability === 'number' ? (signal.probability * 100).toFixed(1) : '?';
  const title = `${label} ${signal.pair} — ${prob}% confidence`;
  const body  = `Entry: ${signal.entry_price || signal.entry || '?'}  SL: ${signal.stop_loss || '?'}  TP: ${signal.take_profit || '?'}`;

  if (Notification.permission !== 'granted') return;

  if (_swRegistration) {
    _swRegistration.showNotification(title, {
      body,
      icon: '/favicon.ico',
      tag: `signal-${signal.pair}`,
      requireInteraction: true,
    });
  } else {
    new Notification(title, { body, icon: '/favicon.ico' });
  }
}

/**
 * Play an alert sound using the Web Audio API — no sound file needed.
 * Produces a short, professional "ping" tone.
 */
export function playAlertSound() {
  try {
    const ctx   = new (window.AudioContext || window.webkitAudioContext)();
    const osc   = ctx.createOscillator();
    const gain  = ctx.createGain();

    osc.connect(gain);
    gain.connect(ctx.destination);

    osc.type      = 'sine';
    osc.frequency.setValueAtTime(880, ctx.currentTime);          // A5
    osc.frequency.exponentialRampToValueAtTime(440, ctx.currentTime + 0.3); // A4

    gain.gain.setValueAtTime(0.4, ctx.currentTime);
    gain.gain.exponentialRampToValueAtTime(0.001, ctx.currentTime + 0.5);

    osc.start(ctx.currentTime);
    osc.stop(ctx.currentTime + 0.5);

    // Second tone (confirmation pip)
    setTimeout(() => {
      const osc2  = ctx.createOscillator();
      const gain2 = ctx.createGain();
      osc2.connect(gain2);
      gain2.connect(ctx.destination);
      osc2.type = 'sine';
      osc2.frequency.setValueAtTime(660, ctx.currentTime);
      gain2.gain.setValueAtTime(0.3, ctx.currentTime);
      gain2.gain.exponentialRampToValueAtTime(0.001, ctx.currentTime + 0.3);
      osc2.start(ctx.currentTime);
      osc2.stop(ctx.currentTime + 0.3);
    }, 250);
  } catch (e) {
    // Audio not available — silent fail
  }
}
