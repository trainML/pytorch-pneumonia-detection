#!/bin/bash
usage() {
  # Display the usage and exit.
  echo "Usage: ${0} ENDPOINT_URL IMAGE" >&2
  echo 'Send an image to an endpoint to be classified' >&2
  exit 1
}

if [[ "${#}" -lt 2 ]]
then
  usage
fi

ENDPOINT_URL=$1
PID=$(basename ${2} .dcm)

if command -v jq &> /dev/null
then
  (echo -n "{\"pId\": \"${PID}\", \"dicom\": \""; base64 ${2}; echo '"}') |
  curl -s -H "Content-Type: application/json" -d @-  ${ENDPOINT_URL}/predict | \
  jq -r .[\"${PID}\"].image | base64 --decode > annotated_image.png
  echo "Wrote annotated image to $(pwd)/annotated_image.png"
else
  (echo -n "{\"pId\": \"${PID}\", \"dicom\": \""; base64 ${2}; echo '"}') |
  curl -H "Content-Type: application/json" -d @-  ${ENDPOINT_URL}/predict 
fi