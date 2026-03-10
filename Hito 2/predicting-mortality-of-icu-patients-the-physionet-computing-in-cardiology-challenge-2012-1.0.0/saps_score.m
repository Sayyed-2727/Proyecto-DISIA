function [SAPS_SCORE]=saps_score(varargin)
% [SAPS_SCORE]=saps_score(tm,category,val,truncated)
%
% Calculates SAPS scores. Variables are:
%
%tm      - (Nx1 Cell Array) Cell array containing time of measurement
%category- (Nx1 Cell Array) Cell array containing type (category)
%           measurement
%value   - (Nx1 Cell Array) Cell array containing value of measurement
%truncated - (Logical) Optional flag, if true, will attempt to calculate
%             "truncated" SAPS even if some of the input variables are
%             missing. Default is 0 (false) .
% SAP_SCORE - (Scalar) Value between 0 and 56 representing the severity of
%             the patient's status (higher scores are worse).  A NaN
%             value is returned along with a warning message if the SAPS
%             score cannot be calculated.
%
% Written by Ikaro Silva, 2012
%
% Version 1.0
%
% Reference:
% Gall et al, "A simplified acute physiology score for ICU patients",
% Critical Care Medicine (1984), 12(11).

truncated=0;
inputs={'tm','category','val','truncated'};
for n=1:nargin
    if(~isempty(varargin{n}))
        eval([inputs{n} '=varargin{n};'])
    end
end

%SAPS variable names
SAPS={{'Age'},{'HR'},{'SysABP','NISysABP'},{'Temp'},...
    {'RespRate','MechVent'},{'Urine'},{'BUN'},{'HCT'},...
    {'WBC'},{'Glucose'},{'K'},{'Na'},{'HCO3'},{'GCS'}};

MX_SAPS=56; %Max SAPS value

%Convert SAPS info into tables [min range, max range, SAPS score;...]
%Also convert the units in the table to match the units of the data
Age=[0,45,0;46,55,1;56,65,2;66,75,3;76,200,4];
HR=[70,109,0;55,69,2;110,139,2;40,54,3;140,179,3;180,250,4;10,39,4];
SysABP=[80,149,0;55,79,2;150,189,2;190,300,4;20,54,4];
Temp=[36,38.4,0;34,35.9,1;38.5,38.9,1;32,33.9,2;30,31.9,3;39,40.9,3;41,45,4;15,29.9,4];
RespRate=[12,24,0;10,11,1;25,34,1;6,9,2;35,49,3;2,5,4;50,80,4];
MechVent=49; %Equivalent to a RespRate that will yield a SAPS value of 3
Urine=[5,20,0.002;3.5,4.999,0.001;0.7,3.499,0;0.5,0.699,0.002;0.2,0.499,0.003;0,0.199,0.004].*1000; %Convert from L to mL
BUN=[55,100;36,54.9;29,35.9;7.5,28.9;3.5,7.4;1,3.4].*2.8; %Convert to mg/dL
BUN(:,3)=[4;3;2;1;0;1];
BUN(:,1)=[BUN(2:end,2)+eps;BUN(end,1)];
HCT=[60,80,4;50,59.9,2;46,49.9,1;30,45.9,0;20,29.9,2;5,19.9,4];
WBC=[40,200,4;20,39.9,2;15,19.9,1;3,14.9,0;1,2.9,2;0.100,0.9,4];
Glucose=[44.5,1000;27.8,44.4;14,27.7;3.9,13.9;2.8,3.8;1.6,2.7;0.5,1.5].*18;%Convert to mg/dL
Glucose(:,3)=[4;3;1;0;2;3;4];
Glucose(:,1)=[Glucose(2:end,2)+eps;Glucose(end,1)];
K=[7,20,4;6,6.9,3;5.5,5.9,2;3.5,5.4,0;3,3.4,1;2.5,2.9,2;0.5,2.4,4];
Na=[180,200,4;161,179,3;156,160,2;151,155,1;130,150,0;120,129,2;110,119,3;50,109,4];
HCO3=[40,100,4;30,39.9,1;20,29.9,0;10,19.9,1;5,9.9,3;2,4.9,4];
GCS=[13,15,0;10,12,1;7,9,2;4,6,3;3,3,4];

%Only use data for the first 24 hrs (standard SAPS)
tm=cell2mat(tm);
fr_data=find(str2num(tm(:,1:2))<24);
val=val(fr_data);
tm=tm(fr_data);
category=category(fr_data);

%Loop through all SAPS variables, adding risk points to SAPS_SCORE according to their tables
SAPS_SCORE=NaN;
for s=1:length(SAPS)
    %Get data for the selected category only (If more than one name exist for the variables, merge data)
    saps_var=SAPS{s};
    sig_ind= val.*0;
    eval(['table=' regexprep(saps_var{1},' ','_') ';'])
    for i=1:length(saps_var)
        sig_ind=sig_ind | strcmp(saps_var(i),category);
    end
    tmp_data=val(sig_ind);
    
    if(strcmp(saps_var{1},'RespRate'))
        %For Respiration, check Mechanical Ventilation flag
        %If subject is on ventillation, this is equivalent to having a
        %RespRate that will yield a SAPS value of 3
        tmp_category=category(sig_ind);
        mech_vent_ind=find(strcmp(saps_var(2),tmp_category)==1);
        if(~isempty(mech_vent_ind) && any(mech_vent_ind))
            tmp_data=MechVent;
        end
    end
    if(strcmp(saps_var{1},'Urine'))
        %For Urine output, get cumulative over 24hrs
        tmp_data=sum(tmp_data);
    end
    
    %Set out of bound values to NaN
    tmp_data(tmp_data < min(table(:,1)))=NaN;
    tmp_data(tmp_data > max(table(:,2)))=NaN;
    %Get the lowest and highest values for the measured variable
    if(all(isnan(tmp_data)) || isempty(tmp_data))
        if(~truncated)
            warning(['No SAPS score for variable ' saps_var{:} '... exiting.'])
            SAPS_SCORE=NaN;
            break
        else
            warning(['No variable ' saps_var{:} '... attempting truncated SAPS.'])
        end
    else
        mn=nanmin(tmp_data);
        mn_saps_ind=find( ( mn(1) >= table(:,1) & mn(1) <= table(:,2) ) ==1);
        mn_saps_val=table(mn_saps_ind,3);
        mx=nanmax(tmp_data);
        mx_saps_ind=find( ( mx(1) >= table(:,1) & mx(1) <= table(:,2) ) ==1);
        mx_saps_val=table(mx_saps_ind,3);
        %Add worst case scenario to SAPS SCORE
        if(isnan(SAPS_SCORE))
            SAPS_SCORE= nanmax(mn_saps_val,mx_saps_val);
        else
            SAPS_SCORE= SAPS_SCORE + nanmax(mn_saps_val,mx_saps_val);
        end
    end
end


