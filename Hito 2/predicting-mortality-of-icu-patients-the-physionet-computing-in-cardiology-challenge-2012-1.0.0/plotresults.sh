#! /bin/sh

# file: plotresults.sh		G. Moody	23 March 2012
#
# Plot results generated using genresults.sh
#
# This script requires plt (http://physionet.org/physiotools/plt/) and
# ImageMagick (http://www.imagemagick.org/).  After running genresults.sh
# to generate its input files (Score-a.txt and Plot-a.txt), run it as
#      plotresults.sh a
# to produce EPS- and PNG-format plots (results.eps and results.png).

S=$1

E1=`grep "Event 1" <Score-$S.txt | cut -c27-`
E2=`grep "Event 2" <Score-$S.txt | cut -c27-`

plt Plot-$S.txt -p "0,1Scircle(Cblue) 0,2Striangle(Cred)" -X 0 - \
    -sf t "Fh,P14" -t "Set $S: Event 1 score: $E1   Event 2 score: $E2" \
    -x "Predicted Risk" -y "Number of Deaths" -sf a "Fh,P12" \
    -sf f "Fh,P10" -lp .15 .95 1.05 -le 0 0 Observed -le 1 1 Predicted -Tlw | \
   lwcat -eps -golden >results-$S.eps

convert -density 100x100 results-$S.eps results-$S.png

