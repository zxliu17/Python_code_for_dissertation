#!/gnuplot
set ylabel 'Hamming distance between final clusters'
set xlabel 'Threshold'
unset key
set xrange [0:1]
set yrange [0.4:1.1]
set mxtics 2
set mytics 2
set output 'c:\users\liuzx\desktop\Ave_hamming_diff_befs.tex'
#plot 'c:\users\liuzx\desktop\data\Ave_sim_diff_befs' using 1:2 with points lc 7, 
plot 'c:\users\liuzx\desktop\data\bn_hammin_100a' using 1:5:6 with yerrorbars lt 2 lc -1 lw 1.5
#    EOF
#    EOF