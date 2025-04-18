from django.contrib.auth import login, logout, authenticate
from django.shortcuts import redirect
from django.contrib import messages

import logging
logger = logging.getLogger(__name__)

def login_request(request):
    if request.method == "POST":
        caregiver_id = request.POST['caregiver_id']
        password = request.POST['password']

        assert caregiver_id and password
        user = authenticate(username=caregiver_id, password=password)
        
        if user is not None:
            login(request, user)
            return redirect('home')
        else:
            messages.error(request, "Invalid CGID or password.")
            return redirect('home')

    else:
        return redirect('home')


def logout_request(request):
    logout(request)
    return redirect('home')