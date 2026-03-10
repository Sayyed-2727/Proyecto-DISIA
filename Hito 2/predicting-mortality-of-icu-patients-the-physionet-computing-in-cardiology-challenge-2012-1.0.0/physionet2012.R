# file: physionet2012.R
# sample Challenge 2012 entry in R
# To test this script, copy it and the first Challenge data file from set A
# into the current directory, and run this command:
#    Rscript physionet2012.R <132539.txt
# This command should write one line to the standard output:
#    132539,0,0.5

rec = read.csv(file("stdin"), header=TRUE, stringsAsFactors=FALSE);
RecordID = rec[1, 'Value'];
prediction = 0;
risk = 0.5;
cat(sprintf("%i,%i,%g\n", RecordID, prediction, risk));
