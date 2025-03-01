"""
WSGI config for backend project.

It exposes the WSGI callable as a module-level variable named ``application``.

For more information on this file, see
https://docs.djangoproject.com/en/5.1/howto/deployment/wsgi/
"""

import os
from django.core.wsgi import get_wsgi_application 
from whitenoise import WhiteNoise


settings_module = 'crud.deployment' if 'WEBSITE_HOSTNAME' in os.environ else 'crud.settings'

os.environ.setdefault('DJANGO_SETTINGS_MODULE', settings_module)
 
application = get_wsgi_application()
application = WhiteNoise(application)
