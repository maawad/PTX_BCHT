load_factor=(0.60 0.65 0.70 0.75 0.80 0.82 0.84 0.86 0.88 0.90 0.91 0.92 0.93 0.94 0.95 0.96 0.97 0.98 0.99)

echo "   Millions    |      %      |        Million keys/s      "
echo "Number of keys | Load factor | Insertion rate | Find rate "

for lf in "${load_factor[@]}"
do
    ./build/ptx_cuckoo_hashtable  --num-keys=50'000'000 --load-factor=$lf --quiet=1 --exist-ratio=0.0
done


for lf in "${load_factor[@]}"
do
    ./build/ptx_cuckoo_hashtable  --num-keys=50'000'000 --load-factor=$lf --quiet=1 --exist-ratio=0.5
done


for lf in "${load_factor[@]}"
do
    ./build/ptx_cuckoo_hashtable  --num-keys=50'000'000 --load-factor=$lf --quiet=1 --exist-ratio=1.0
done