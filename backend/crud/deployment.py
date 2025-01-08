from .settings import *
import os

DEBUG = False
ALLOWED_HOSTS = [os.environ['WEBSITE_HOSTNAME']]
CSRF_TRUSTED_ORIGINS = ['https://'+os.environ['WEBSITE_HOSTNAME']]
SECRET_KEY = os.environ['MY_SECRET_KEY']

MIDDLEWARE = [
    'corsheaders.middleware.CorsMiddleware',
    'django.middleware.security.SecurityMiddleware',
    'whitenoise.middleware.WhiteNoiseMiddleware',
    'django.contrib.sessions.middleware.SessionMiddleware',
    'django.middleware.common.CommonMiddleware',
    'django.middleware.csrf.CsrfViewMiddleware',
    'django.contrib.auth.middleware.AuthenticationMiddleware',
    'django.contrib.messages.middleware.MessageMiddleware',
    'django.middleware.clickjacking.XFrameOptionsMiddleware',
]

# CORS_ALLOWED_ORIGINS = [
#     'https://gentle-sand-0e04dc01e.4.azurestaticapps.net' 
# ]


STORAGES = {
    "default": {
        "BACKEND": "django.core.files.storage.FileSystemStorage",
    },
    "staticfiles": {
        "BACKEND": "whitenoise.storage.CompressedManifestStaticFilesStorage",
    },
}


CONNECTION = os.environ['AZURE_POSTGRESQL_CONNECTIONSTRING']
CONNECTION_STR = {pair.split('=')[0]:pair.split('=')[1] for pair in CONNECTION.split(' ')}
 
 
DATABASES = {
    "default": {
        "ENGINE": "django.db.backends.postgresql",
        "NAME": CONNECTION_STR['dbname'],
        "HOST": CONNECTION_STR['host'],
        "USER": CONNECTION_STR['user'],
        "PASSWORD": CONNECTION_STR['password'],
    }
}
 
DATABASE_URL = f"postgresql://{CONNECTION_STR['user']}:{CONNECTION_STR['password']}@{CONNECTION_STR['host']}/{CONNECTION_STR['dbname']}"


LOGGING = {
    'version': 1,
    'disable_existing_loggers': False,
    'handlers': {
        'mail_admins': {
            'level': 'ERROR',
            'class': 'django.utils.log.AdminEmailHandler',
        },
    },
    'loggers': {
        'django': {
            'handlers': ['mail_admins'],
            'level': 'ERROR',
            'propagate': True,
        },
    },
}
 
STATIC_ROOT = BASE_DIR/'staticfiles'
STATIC_URL = '/static/'