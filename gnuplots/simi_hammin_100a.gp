#!/gnuplot
set ylabel 'Similarity' offset 2 font 'times new roman,10'
set xlabel 'Threshold' offset 0,0.5 font 'times new roman,10'
unset key
set xrange [0:1]
set yrange [0:1]
set xtics font 'times new roman'
set ytics font 'times new roman'
set mxtics 2
set mytics 2
#set output 'Simi_hammin_100a.pdf'
plot 'c:\users\liuzx\desktop\data\bn_hammin_100a' using 1:2 with linespoints  lc -1 lw 1
#    EOF