from django.db import models


class Signal(models.Model):
    """
    One trading signal per pair per day.
    entry_price, stop_loss, take_profit are ACTUAL PRICES (not pip distances).
    """
    SIGNAL_CHOICES = [
        ('bullish',   'Bullish'),
        ('bearish',   'Bearish'),
        ('no_signal', 'No Signal'),
    ]
    SOURCE_CHOICES = [
        ('engine',  'SignalEngine'),
        ('unified', 'UnifiedSignalService'),
        ('manual',  'Manual'),
    ]

    pair        = models.CharField(max_length=10, db_index=True)
    signal      = models.CharField(max_length=10, choices=SIGNAL_CHOICES)
    probability = models.FloatField(help_text='P(bullish), 0–1')
    confidence  = models.FloatField(null=True, blank=True, help_text='abs(prob-0.5)*2, 0–1')
    entry_price = models.FloatField(null=True, blank=True)
    stop_loss   = models.FloatField(null=True, blank=True, help_text='Actual price level')
    take_profit = models.FloatField(null=True, blank=True, help_text='Actual price level')
    risk_reward = models.FloatField(null=True, blank=True)
    atr         = models.FloatField(null=True, blank=True)
    source      = models.CharField(max_length=20, choices=SOURCE_CHOICES, default='engine')
    date        = models.DateField(db_index=True)
    created_at  = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering            = ['-date', '-id']
        unique_together     = [('pair', 'date')]   # one signal per pair per day
        verbose_name        = 'Signal'
        verbose_name_plural = 'Signals'

    def __str__(self):
        return f'{self.pair} {self.signal} @ {self.date} (p={self.probability:.3f})'
