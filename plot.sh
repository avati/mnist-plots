#!/bin/bash

export PATH=$PATH:/usr/local/octave/3.8.0/bin


(
    echo "hold on;"
    colors="k b g"
    start="11"
    end="end"
    set $colors
#   for hdim_nlayers in 32_1 32_2 64_1 64_2 128_2 128_3 256_3; do
#   for hdim_nlayers in 32_1 64_1 128_2; do
    for hdim_nlayers in 32_1 64_1 128_2; do
	hdim=`echo $hdim_nlayers | cut -f1 -d_`
	nlayers=`echo $hdim_nlayers | cut -f2 -d_`
    
	echo "h${hdim}n${nlayers} = load('hdim-$hdim-nlayers-$nlayers.csv');"
	echo "plot(h${hdim}n${nlayers}($start:$end, 1), h${hdim}n${nlayers}($start:$end, 2), color='$1');"
	shift;

    done

    echo "sm = load('softmax.csv');"
    echo "plot(sm($start:$end, 1), sm($start:$end, 2), color='r')"
    echo "xlabel ('fraction of data set used');"
    echo "ylabel ('performance');"
    echo "title ('MNIST performance vs data size vs model size');"
    echo "legend ('small (dim=32,nlayers=1)', 'medium (dim=64,nlayers=1)', 'large (dim=128,nlayers=2)', 'softmax', 'location', 'southeast');"
    echo "print('plot.png', '-dpng');"

) | octave -q
