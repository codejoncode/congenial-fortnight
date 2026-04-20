from django.db import migrations, models


def deduplicate_signals(apps, schema_editor):
    """Keep only the latest Signal per (pair, date) before adding unique_together."""
    Signal = apps.get_model('signals', 'Signal')
    from django.db.models import Max

    # Find the latest id per (pair, date)
    latest_ids = (
        Signal.objects
        .values('pair', 'date')
        .annotate(max_id=Max('id'))
        .values_list('max_id', flat=True)
    )
    # Delete all rows that are NOT the latest per (pair, date)
    deleted, _ = Signal.objects.exclude(id__in=list(latest_ids)).delete()
    if deleted:
        print(f'  Deduplicated {deleted} duplicate Signal rows.')


class Migration(migrations.Migration):

    dependencies = [
        ('signals', '0001_initial'),
    ]

    operations = [
        # Deduplicate first so unique_together can be applied cleanly
        migrations.RunPython(deduplicate_signals, migrations.RunPython.noop),

        # Add new price-level fields
        migrations.AddField(
            model_name='signal',
            name='confidence',
            field=models.FloatField(blank=True, help_text='abs(prob-0.5)*2, 0\u20131', null=True),
        ),
        migrations.AddField(
            model_name='signal',
            name='entry_price',
            field=models.FloatField(blank=True, null=True),
        ),
        migrations.AddField(
            model_name='signal',
            name='take_profit',
            field=models.FloatField(blank=True, help_text='Actual price level', null=True),
        ),
        migrations.AddField(
            model_name='signal',
            name='risk_reward',
            field=models.FloatField(blank=True, null=True),
        ),
        migrations.AddField(
            model_name='signal',
            name='atr',
            field=models.FloatField(blank=True, null=True),
        ),
        migrations.AddField(
            model_name='signal',
            name='source',
            field=models.CharField(
                choices=[('engine', 'SignalEngine'), ('unified', 'UnifiedSignalService'), ('manual', 'Manual')],
                default='engine',
                max_length=20,
            ),
        ),
        # Change stop_loss help text (already FloatField(null=True), no schema change needed)
        # Update signal choices to include 'no_signal'
        migrations.AlterField(
            model_name='signal',
            name='signal',
            field=models.CharField(
                choices=[('bullish', 'Bullish'), ('bearish', 'Bearish'), ('no_signal', 'No Signal')],
                max_length=10,
            ),
        ),
        # Add unique_together constraint
        migrations.AlterUniqueTogether(
            name='signal',
            unique_together={('pair', 'date')},
        ),
        # Add db_index to pair and date
        migrations.AlterField(
            model_name='signal',
            name='pair',
            field=models.CharField(db_index=True, max_length=10),
        ),
        migrations.AlterField(
            model_name='signal',
            name='date',
            field=models.DateField(db_index=True),
        ),
    ]
