set term svg size 800,500 dynamic font "sans-serif"
set term post enh
set out 'plottime.eps'
set autoscale xfix
set grid
set style data linespoints
set pointsize 0.1
set key right below box
set yrange [:]
set xrange [:]
set size ratio 0.5

set datafile commentschars "#%;"
set ylabel "Tempo (ms)"
set xlabel "Frame"
plot \
 "$TESTDIR\\hex_const\\benchmark.txt" using 1 title "  Hex Const" pt 5, \
 "$TESTDIR\\hex_gauss\\benchmark.txt" using 1 title "  Hex Gauss" pt 5, \
 "$TESTDIR\\hex_fixed5\\benchmark.txt" using 1 title "  Hex Fixed 5" pt 5, \
 "$TESTDIR\\hex_fixed10\\benchmark.txt" using 1 title "  Hex Fixed 10" pt 5, \
 "$TESTDIR\\hex_fixed20\\benchmark.txt" using 1 title "  Hex Fixed 20" pt 5, \
 "$TESTDIR\\tet_gauss\\benchmark.txt" using 1 title "  Tet Gauss" pt 5, \
 "$TESTDIR\\tet_const\\benchmark.txt" using 1 title "  Tet Const" pt 5, \
 "$TESTDIR\\tet_pre\\benchmark.txt" using 1 title "  Tet Pre" pt 5, \
 "$TESTDIR\\tet_fixed5\\benchmark.txt" using 1 title "  Tet Fixed 5" pt 5, \
 "$TESTDIR\\tet_fixed10\\benchmark.txt" using 1 title "  Tet Fixed 10" pt 5, \
 "$TESTDIR\\tet_fixed20\\benchmark.txt" using 1 title "  Tet Fixed 20" pt 5
 
 set terminal png
 set output "plottime.png"
 replot
 
 
