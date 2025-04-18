from django.db.models.signals import post_migrate
from django.dispatch import receiver
from django.core.management import call_command
from django.contrib.auth.models import User # Group
from django.db.models.signals import post_save
from django.dispatch import receiver
from .models import Caregiver
import logging

logger = logging.getLogger(__name__)

# @receiver(post_save, sender=User)
# def create_caregiver_profile(sender, instance, created, **kwargs):
#     """
#     Create a Caregiver profile for new users automatically.
#     """
#     if created:
#         Caregiver.objects.create(user=instance)
#         logger.info(f"Created caregiver profile for new user: {instance.username}")


# @receiver(post_save, sender=Caregiver)
# def assign_user_to_group(sender, instance, created, **kwargs):
#     """
#     When a Caregiver is created or updated, assign the user to the corresponding Django Group.
#     If the group doesn't exist yet, create it.
#     """
#     # Get or create the group based on caregiver_group
#     group_name = dict(Caregiver.CAREGIVER_GROUPS).get(instance.caregiver_group)
#     logger.info(f"Processing caregiver: {instance.user.username}, group: {group_name}")
    
#     group, created = Group.objects.get_or_create(name=group_name)
#     if created:
#         logger.info(f"Created new group: {group_name}")
#     else:
#         logger.info(f"Using existing group: {group_name}")
    
#     # Remove user from all existing caregiver groups first
#     caregiver_group_names = [name for _, name in Caregiver.CAREGIVER_GROUPS]
#     user_groups = instance.user.groups.filter(name__in=caregiver_group_names)
#     for g in user_groups:
#         instance.user.groups.remove(g)
#         logger.info(f"Removed user {instance.user.username} from group {g.name}")
    
#     # Add user to the new group
#     instance.user.groups.add(group)
#     logger.info(f"Added user {instance.user.username} to group {group_name}")

@receiver(post_migrate)
def create_groups_after_migration(sender, **kwargs):
    """
    Signal handler to create caregiver groups after migrations are complete.
    This only runs the command when migrations for the 'accounts' app are processed.
    """
    # Only run when the accounts app is migrated
    if sender.name == 'accounts':
        call_command('create_caregiver_groups')