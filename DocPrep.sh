ls | grep '\.TXT' | xargs cat > CompleteText.txt
gcsplit CompleteText.txt /DOCUMENTS/ {*}
for item in `ls | grep xx`
do 
    mv $item "$item".txt
done

for i in 1890 2611 2558 2536 1616 2372  205 15 1183  797 1084 728  986 1910 2537 2272 1260 2703 2322 2785 1358  256 2640 2421 1977  326 2855  311 1080 1858 2145 1348 2845 1231   28 1684  878 2213  774 2240 2856  209 2826 2216 1037 922  895  899  705  183 1863 1565  149 2596 2754 2823  548 2634 2777  787 319 1235 1659  741 1081 1445  983  371 2592 2927 2765 2208 1196 2320  200 1093 2310  281 2931 2381 2802 1686 1859 1692 2902 2096 1629 1960 2625 2120 2577 2796 1454   55  699 2167 2604 1570 2250 2645
do
    cp xx"$i".txt SampleArticles/
done