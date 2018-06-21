#!/gnuplot
set ylabel 'Hamming distance between final clusters'
set xlabel 'Threshold'
unset key
set xrange [0:1]
set yrange [0.45:0.55]
set mxtics 2
set mytics 2
set output 'Ave_dis_diff_befs_max.tex'
#plot 'c:\users\liuzx\desktop\data\Ave_sim_diff_befs' using 1:2 with points lc 7, 
plot 'c:\users\liuzx\desktop\data\thre_maxhamming_100a' using 1:10:11 with yerrorbars lt 2 lc -1 lw 1.5
#    EOF
#    EOF