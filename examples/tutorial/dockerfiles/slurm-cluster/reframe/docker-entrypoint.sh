#!/bin/bash

trap exit 0 INT

sudo service munge start

echo "Container up and running: "
echo "==> Run 'docker exec -it <container-id> /bin/bash' to run interactively"
echo "==> Press Ctrl-C to exit"
sleep infinity &
wait
