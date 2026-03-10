#! /bin/sh
# file: genresults.sh	G. Moody	22 March 2012

# Shell script for processing a Challenge 2012 data set with a Challenge entry
# Version 1.0 (22 March 2012)

# A script similar to this one will be used as part of the evaluation of
# Challenge entries written in C.  A companion script will be used to perform
# the same function for Challenge entries written in MATLAB m-code.  We have
# provided these scripts so that Challenge participants can test their entries
# to verify that they run properly in the environment that will be used to test
# them.

# Each Challenge .txt file (record) contains data for one patient, in 3 columns
# (timestamp, parameter, and value).  The three Challenge data sets contain
# 4000 records each.

# This script supplies a complete set of 4000 records, one at a time, to an
# entry, and collects its output for each record (a binary prediction of
# survival, for event 1, and an estimate of mortality risk, for event 2)
# in a summary output file.

# The summary output file is then scored by comparing its contents with the
# patients' known outcomes.  This script invokes a small C program ('score')
# that can calculate unofficial scores for the training data (set A), for which
# the known outcomes are provided to participants.  Entries will be ranked using
# official scores obtained using the same methods, but based on testing with
# sets B and C, for which the outcomes are not provided to participants.

# If your entry is written in C, it must be compilable into an executable
# program named physionet2012, which reads the contents of a Challenge record
# from its standard input and writes the results of its analysis to its
# standard output, as a single line containing the RecordID (read from the
# input), a binary prediction (0: survival, 1: death), and a risk estimate
# (0: certain survival, 1: certain death).  Example:
#     123456,0,0.16
# See the sample entry at http://physionet.org/challenge/2012/physionet2012.c
# for details.

# To use this script to obtain an unofficial score for your entry on set A:
#  1. Download these files from http://physionet.org/challenge/2012/ and save
#     them in your working directory:
#	genresults.sh		   (this file)
#	score.c 		   (program needed to calculate scores)
#	set-a.zip or set-a.tar.gz  (zip archive or tarball of set A files)
#	Outcomes-a.txt		   (known outcomes for set A)
#  2. Compile score.c .  Using gcc, this can be done by:
#       gcc -o score score.c
#  3. Unzip set-a.zip (or unpack set-a.tar.gz), creating a subdirectory within
#	your working directory called 'set-a'.  When you have completed this
#	step, the set-a directory should contain 4000 individual .txt files.
#  4. Save an executable version of your entry (physionet2012) in your working
#     directory.
#  5. Run this script by:
#       ./genresults set-a

# You can also use this script to run your entry on test set B, but it will
# not be able to calculate scores in this case, since the outcomes are provided
# for set A only.  To do this, download and unzip/unpack set B into a 'set-b'
# subdirectory as you did for set A, and run:
#       ./genresults set-b

if [ $# != 1 ]
then
    echo Supply the name of the directory containing the data set.  Type either
    echo "   $0 set-a"
    echo or
    echo "   $0 set-b"
    exit
fi

case $1 in
    set-a) S=a; OUT=Outputs-a.txt ;;
    set-b) S=b; OUT=Outputs-b.txt ;;
    *) echo Choose either set-a or set-b.
       exit ;;
esac

for R in $1/*.txt
do
    ./physionet2012 <$R >>$OUT
    echo $R
done

if [ -s Outcomes-$S.txt ]
then
    ./score $OUT Outcomes-$S.txt >Score-$S.txt 2>Plot-$S.txt
    echo
    cat Score-$S.txt
fi
