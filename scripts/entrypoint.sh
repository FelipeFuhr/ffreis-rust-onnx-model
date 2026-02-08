#!/bin/sh

APP="${PWD}/release/${APP_NAME}"

if [ -z "${APP_NAME}" ]; then
    echo "ERROR: APP_NAME is not set." >&2
    exit 1
fi

if [ ! -x "${APP}" ]; then
    echo "ERROR: Binary '${APP_NAME}' does not exist or is not executable at ${APP}" >&2
    exit 1
fi

exec "${APP}"