set -ex

./build/bin/profile
gprof ./build/bin/profile ./gmon.out > report.txt
gprof2dot -n 5 -e 5 report.txt | dot -Tsvg -o pg.svg
