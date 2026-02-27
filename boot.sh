#!/bin/sh
exec gunicorn -b :5000 --workers 1 --access-logfile - --error-logfile - app --timeout 300
