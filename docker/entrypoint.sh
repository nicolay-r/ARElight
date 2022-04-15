#!/bin/bash

echo
SetEnv PYTHONPATH /arelight

printf "\nSetEnv IP_ADDRESS $IP_ADDRESS" >> /etc/apache2/sites-enabled/demo.conf

$@