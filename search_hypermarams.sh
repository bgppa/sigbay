# This script executes my python small programm with all the parameter
# combination that we need
#!/bin/sh

datafile="data/btc-days-600.data"
filename="main.py"

for i in {2..50}
do
	echo "Running: python3 ${filename} ${datafile} ${i}"
        cmd=`python3 ${filename} ${datafile} ${i}`
done
