cd /home/site/wwwroot
python manage.py collectstatic --noinput
python manage.py migrate
uvicorn crud.asgi:application --host 0.0.0.0 --port 8000
 