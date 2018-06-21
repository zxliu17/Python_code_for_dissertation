#!/gnuplot
set ylabel 'Number of final belief'
set xlabel 'Threshold'
unset key
set xrange [0:1]
set yrange [0:20]
set mxtics 2
set mytics 2
set output 'bn_hammin_100a.tex'
plot 'c:\users\liuzx\desktop\data\bn_hammin_100a' using 1:4 with linespoints  lc -1 lw 2
#    EOF