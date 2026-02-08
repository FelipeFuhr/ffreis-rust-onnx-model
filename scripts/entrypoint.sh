#!/bin/sh

if [ -z "${APP_NAME}" ]; then
    echo "ERROR: APP_NAME is not set." >&2
    exit 1
fi

if [ ! -x "${PWD}/${APP_NAME}" ]; then
    echo "ERROR: Binary '${APP_NAME}' does not exist or is not executable at ${PWD}/${APP_NAME}" >&2
    exit 1
fi

exec "${PWD}/${APP_NAME}"