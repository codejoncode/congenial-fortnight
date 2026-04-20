/* Service Worker for trading signal push notifications.
   Registered by frontend/src/utils/notifications.js on app load.
   Works on localhost — no HTTPS required for local use. */

self.addEventListener('install', () => self.skipWaiting());
self.addEventListener('activate', (e) => e.waitUntil(self.clients.claim()));

self.addEventListener('push', (event) => {
  const data = event.data ? event.data.json() : {};
  const title   = data.title   || 'Trading Signal';
  const options = {
    body:    data.body    || 'New signal generated',
    icon:    data.icon    || '/favicon.ico',
    badge:   '/favicon.ico',
    tag:     data.tag     || 'signal',
    requireInteraction: true,
    data:    { url: data.url || '/' },
    actions: [
      { action: 'open',    title: 'Open Dashboard' },
      { action: 'dismiss', title: 'Dismiss' },
    ],
  };
  event.waitUntil(self.registration.showNotification(title, options));
});

self.addEventListener('notificationclick', (event) => {
  event.notification.close();
  if (event.action !== 'dismiss') {
    event.waitUntil(
      self.clients.matchAll({ type: 'window' }).then((clients) => {
        for (const client of clients) {
          if ('focus' in client) return client.focus();
        }
        return self.clients.openWindow('/');
      })
    );
  }
});
