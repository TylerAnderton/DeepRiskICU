from django.contrib import admin
from django.contrib.auth.admin import UserAdmin as BaseUserAdmin
from django.contrib.auth.models import User, Group
from .models import Caregiver

# Register the Caregiver model
class CaregiverAdmin(admin.ModelAdmin):
    list_display = ('user', 'caregiver_id', 'get_caregiver_group')
    
    def get_caregiver_group(self, obj):
        return dict(Caregiver.CAREGIVER_GROUPS).get(obj.caregiver_group)
    get_caregiver_group.short_description = 'Caregiver Group'

admin.site.register(Caregiver, CaregiverAdmin)

# Define an inline admin descriptor for Caregiver model
class CaregiverInline(admin.StackedInline):
    model = Caregiver
    can_delete = False
    verbose_name_plural = 'Caregiver Information'

# Define a new User admin
class UserAdmin(BaseUserAdmin):
    inlines = (CaregiverInline,)

# Re-register UserAdmin
admin.site.unregister(User)
admin.site.register(User, UserAdmin)

# Make sure Group model is registered
if not admin.site.is_registered(Group):
    admin.site.register(Group)