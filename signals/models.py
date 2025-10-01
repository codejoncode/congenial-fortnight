from django.db import models

# Create your models here.

class Signal(models.Model):
    pair = models.CharField(max_length=10)
    signal = models.CharField(max_length=10)  # bullish, bearish, no_signal
    stop_loss = models.FloatField(null=True)
    probability = models.FloatField()
    date = models.DateField()
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.pair} {self.signal} on {self.date}"
