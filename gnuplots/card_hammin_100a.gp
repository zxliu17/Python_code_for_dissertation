#!/gnuplot
set ylabel 'Cardinality'
set xlabel 'Threshold'
#unset key
set xrange [0:1]
#set yrange [0:1]
set mxtics 2
set mytics 2
set output 'Cardbn_hammin_100a.tex'
plot 'c:\users\liuzx\desktop\data\bn_hammin_100a' using 1:3 with linespoints title "Cardinality" lt 0 lw 2,'' using 1:4 with linespoints title "Number of beliefs" lc -1 lw 2
#    EOF