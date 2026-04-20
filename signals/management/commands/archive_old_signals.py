"""
management command: archive_old_signals

Deletes Signal records older than --days (default 30) to keep the database
lean.  Keeps at least --keep-per-pair recent records per pair regardless of age
so the dashboard always has something to show.

Usage:
    python manage.py archive_old_signals
    python manage.py archive_old_signals --days 60
    python manage.py archive_old_signals --dry-run
"""

from datetime import timedelta

from django.core.management.base import BaseCommand
from django.utils import timezone


class Command(BaseCommand):
    help = 'Delete Signal records older than N days (default 30)'

    def add_arguments(self, parser):
        parser.add_argument(
            '--days', type=int, default=30,
            help='Delete signals older than this many days (default: 30)',
        )
        parser.add_argument(
            '--keep-per-pair', type=int, default=5,
            help='Always keep at least this many recent records per pair (default: 5)',
        )
        parser.add_argument(
            '--dry-run', action='store_true',
            help='Show what would be deleted without deleting anything',
        )

    def handle(self, *args, **options):
        from signals.models import Signal

        days          = options['days']
        keep_per_pair = options['keep_per_pair']
        dry_run       = options['dry_run']
        cutoff        = timezone.now().date() - timedelta(days=days)

        pairs = Signal.objects.values_list('pair', flat=True).distinct()
        total_deleted = 0

        for pair in pairs:
            # IDs to keep (most recent keep_per_pair)
            keep_ids = list(
                Signal.objects.filter(pair=pair)
                .order_by('-date', '-id')
                .values_list('id', flat=True)[:keep_per_pair]
            )

            old_qs = Signal.objects.filter(
                pair=pair, date__lt=cutoff
            ).exclude(id__in=keep_ids)

            count = old_qs.count()
            if count == 0:
                continue

            if dry_run:
                self.stdout.write(
                    f'  [{pair}] Would delete {count} signal(s) before {cutoff} [dry-run]'
                )
            else:
                old_qs.delete()
                self.stdout.write(
                    self.style.SUCCESS(f'  [{pair}] Deleted {count} signal(s) before {cutoff}')
                )
            total_deleted += count

        if total_deleted == 0:
            self.stdout.write('  No signals to archive.')
        elif not dry_run:
            self.stdout.write(
                self.style.SUCCESS(f'\nArchived {total_deleted} signal(s) older than {days} days.')
            )
        else:
            self.stdout.write(f'\nWould archive {total_deleted} signal(s) [dry-run, nothing deleted].')
