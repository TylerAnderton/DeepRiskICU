from django.shortcuts import redirect
from django.contrib import messages
from functools import wraps
from django.conf import settings

def login_required_with_return(function):
    """
    Decorator for views that checks if the user is logged in, redirecting
    to the login page if necessary and showing a message.
    """
    @wraps(function)
    def wrapper(request, *args, **kwargs):
        if request.user.is_authenticated:
            return function(request, *args, **kwargs)
        else:
            messages.warning(request, "You need to log in to access this feature.")
            return redirect(request.META.get('HTTP_REFERER', '/'))
    return wrapper