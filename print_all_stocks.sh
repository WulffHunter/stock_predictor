for f in ./data/daily_*.csv; do
    echo $f | cut -d'_' -f 2 | cut -d'.' -f 1
done