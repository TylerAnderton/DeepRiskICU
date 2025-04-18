from django.core.management.base import BaseCommand
from django.contrib.auth.models import Group
from accounts.models import Caregiver

class Command(BaseCommand):
    help = 'Creates initial caregiver groups based on the Caregiver model'

    def handle(self, *args, **options):
        self.stdout.write('Creating caregiver groups...')
        
        # Create a group for each caregiver type defined in the model
        for code, name in Caregiver.CAREGIVER_GROUPS:
            group, created = Group.objects.get_or_create(name=name)
            
            if created:
                self.stdout.write(self.style.SUCCESS(f'Created group "{name}"'))
            else:
                self.stdout.write(f'Group "{name}" already exists')
        
        self.stdout.write(self.style.SUCCESS('Done!'))