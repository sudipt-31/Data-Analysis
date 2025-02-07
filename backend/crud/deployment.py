from .settings import *
import os

DEBUG = False

GOOGLE_API_KEY = os.environ['GOOGLE_API_KEY']
ALLOWED_HOSTS = [
    os.environ.get('WEBSITE_HOSTNAME', ''),
    '.azurewebsites.net',
    'localhost',
    '127.0.0.1',
    '*'
]


CSRF_TRUSTED_ORIGINS = [
    'https://' + os.environ.get('WEBSITE_HOSTNAME', ''),
    'https://*.azurewebsites.net'
]


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

CORS_ALLOWED_ORIGINS = [
    'https://proud-glacier-0f0c54c0f.4.azurestaticapps.net',
    # 'localhost:5173', 
]


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