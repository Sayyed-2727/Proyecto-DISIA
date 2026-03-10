% MATLAB script for processing a Challenge 2012 data set with a Challenge entry
% Version 2.0  (22 March 2012)
%
% A script similar to this one will be used as part of the evaluation of
% Challenge entries written in m-code.  A companion script will be used to
% perform the same function for Challenge entries written in C and other
% languages.  We have provided these scripts so that Challenge participants
% can test their entries to verify that they run properly in the environment
% that will be used to test them.
%
% Each Challenge .txt file (record) contains data for one patient, in 3 columns
% (timestamp, parameter, and value).  The three Challenge data sets contain
% 4000 records each.
%
% This script supplies a complete set of 4000 records, one at a time, to an
% entry, and collects its output for each record (a binary prediction of
% survival, for event 1, and an estimate of mortality risk, for event 2)
% in a summary output file.
%
% The summary output file is then scored by comparing its contents with the
% patients' known outcomes.  This script includes MATLAB code that can
% calculate unofficial scores for the training data (set A), for which the
% known outcomes are provided to participants.  Entries will be ranked using
% official scores obtained using the same methods, but based on testing with
% sets B and C, for which the outcomes are not provided to participants.
%
% If your entry is written in m-code, it must be in the form of a function
% named physionet2012, with this signature:
%     [risk,survival]=physionet2012(tm,category,val);
% See the sample entry at http://physionet.org/challenge/2012/physionet2012.m
% for descriptions of the input and output variables.
%
% To use this script to obtain an unofficial score for your entry on set A:
%  1. Download these files from http://physionet.org/challenge/2012/ and save
%     them in your MATLAB working directory:
%	genresults.m		   (this file)
%	lemeshow.m		   (function needed to calculate Event 2 score)
%	set-a.zip or set-a.tar.gz  (zip archive or tarball of set A files)
%	Outcomes-a.txt		   (known outcomes for set A)
%  2. Unzip set-a.zip (or unpack set-a.tar.gz), creating a subdirectory within
%	your working directory called 'set-a'.  When you have completed this
%	step, the set-a directory should contain 4000 individual .txt files.
%  3. Save a copy of your entry (physionet2012.m) in your working directory.
%  4. The next few lines are MATLAB code that clears any previously set
%     variables, and sets the name of the directory containing the input
%     data, the name for the summary output file, and the name of the file
%     containing the known outcomes.  Change them if necessary.

        clear all;close all;clc
        set_name='set-a';
	fname_out='Outputs-a.txt';
	results='Outcomes-a.txt';

%  5. Start MATLAB and type
%       Main_Challenge
%     The fname_out file will be generated in the
%     current directory.
%
% You can also use this script to run your entry on test set B, but it will
% not be able to calculate scores in this case, since the outcomes are provided
% for set A only.  To do this, download and unzip/unpack set B into a 'set-b'
% subdirectory as you did for set A, and uncomment the next three lines:
%
%       set_name='set-b';
%       fname_out='Outputs-b.txt'
%       results=[];
%
% Note that if you run a test on either set A or B more than once, you should
% delete (or rename) your summary output file (named in fname_out), since this
% code appends new outputs to any existing output file rather than starting
% over.


cur_dir=pwd;
fdel='/';
if(ispc)
    fdel='\';
end
data_dir=[cur_dir fdel set_name fdel];

cd(data_dir)
records=dir('*.txt');
cd(cur_dir)

I=length(records);
DATA=zeros(I,3) + NaN;
display(['Processing records ...'])

% Open fname_out and append to any previous contents
fid_out=fopen(fname_out,'a');

% Each Challenge .txt file (record) contains data for one patient, in 3 columns
% (timestamp, parameter, and value).  During each iteration of the loop below,
% the contents of a single record are loaded into arrays named tm,
% category, and val.  Each data set (A, B, and C) contains 4000 records.
header={'tm','category','val'};
for i=1:I
    record_id=records(i).name(1:end-4);
    fname=[data_dir record_id '.txt'];
    
    fid_in=fopen(fname,'r');
    C=textscan(fid_in,'%q %q %f','delimiter', ',','HeaderLines',1);
    fclose(fid_in);
    for n=1:length(header)
        eval([header{n} '=C{:,n};'])
    end

    % The contents of one .txt input file are now ready to be given to your
    % physionet2012 function for analysis in the next line:

    [risk,survival]=physionet2012(tm,category,val);

    % The outputs of the analysis are now available in the risk (event 2)
    % and survival (event 1) variables output by the function.
    
    DATA(i,1)=str2double(record_id);
    DATA(i,2)=risk;
    DATA(i,3)=survival;
    if(~mod(i,500))
        display(['Processed: ' num2str(i) ' records out of ' num2str(I)])
    end

    % Format the output in 3 columns (RecordID, survival, risk).
    del_ind=strfind(fname,fdel);
    if(isempty(del_ind))
        del_ind=0;
    end
    count=fprintf(fid_out,'%s,%u,%f\n',fname(del_ind(end)+1:end-4),survival,risk);
    if(~count)
        warning(['**No data written to output file:' fname_out ' from input file: ' fname])
    end
end
fclose(fid_out);
display(['*** All records processed'])

% If the known outcomes are available, the code below calculates unoffical
% event 1 and event 2 scores.  Scores based on set A are not used to rank
% entries, since all participants have been given the correct answers!

if(~isempty(results))

    % The file of known outcomes contains six columns.  The Challenge goal is
    % to predict the sixth column, IHD (in-hospital death).
    variables={'record_id_res','SAPS','SOFA','LOS','Survival','IHD'};
    fid_result=fopen(results,'r');
    C=textscan(fid_result,'%f %f %f %f %f %f','delimiter', ',','HeaderLines',1);
    fclose(fid_result);
    for n=1:length(variables)
        eval([variables{n} '=C{:,n};'])
    end

    % Calculate sensitivity (Se) and positive predictivity (PPV)
    TP=sum(DATA(IHD==1,3));
    FN=sum(~DATA(IHD==1,3));
    FP=sum(DATA(IHD==0,3));  
    Se=TP/(TP+FN);
    PPV=TP/(TP+FP);
    
    show=1; % if show is 1, the decile graph will be displayed by lemeshow()
    H=lemeshow([IHD DATA(:,2)],show);

    % Use the title of figure to display the results
    title(['H= ' num2str(H) ' Se= ' num2str(Se) ' PPV= ' num2str(PPV)])

    % The event 1 score is the smaller of Se and PPV.
    score1 = min(Se, PPV);
    display(['Unofficial Event 1 score: ' num2str(score1)]);

    % The event 2 score is the Hosmer-Lemeshow statistic (H).
    display(['Unofficial Event 2 score: ' num2str(H)]);
end
