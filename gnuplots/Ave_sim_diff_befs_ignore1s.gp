#!/gnuplot
set ylabel 'Similarity between final clusters'
set xlabel 'Threshold'
unset key
set xrange [0:1]
set yrange [-0.05:0.15]
set mxtics 2
set mytics 2
set output 'Ave_sim_diff_befs.tex'
#plot 'c:\users\liuzx\desktop\data\Ave_sim_diff_befs' using 1:2 with points lc 7, 
plot 'c:\users\liuzx\desktop\data\Ave_sim_diff_befs_ignore1s' using 1:2:3 with yerrorbars lt 2 lc -1 lw 1.5
#    EOF
#    EOF