#!/bin/sh

if [ -z "${APP_NAME}" ]; then
    echo "ERROR: APP_NAME is not set." >&2
    exit 1
fi
exec "${PWD}/${APP_NAME}"