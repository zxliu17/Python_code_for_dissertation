#!/gnuplot
set ylabel 'Number of final belief'
set xlabel 'Threshold'
unset key
set xrange [0:1]
set yrange [0:12]
set mxtics 2
set mytics 2
set output 'bn_hammin_50.tex'
plot 'c:\users\liuzx\desktop\data\beliefnumber50a50p' using 1:2 with linespoints  lc -1 lw 2
#    EOF
