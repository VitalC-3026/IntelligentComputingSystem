export CNRT_GET_HARDWARE_TIME=on
export CNRT_PRINT_INFO=on
rm *.jpg
./bin/style_transfer chicago udnie_int8
./bin/style_transfer chicago udnie_int8_power_diff
