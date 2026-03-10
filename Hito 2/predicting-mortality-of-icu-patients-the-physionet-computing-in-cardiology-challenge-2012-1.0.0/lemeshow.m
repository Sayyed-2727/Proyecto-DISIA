function [H]=lemeshow(data,show)
%[H]=lemeshow(data,show)
%
%data -(Nx2) where N is the number of cases (4000 for each of sets A, B, and C),
%	     the first column is the true outcome (IHD from the Outcomes file),
%            and the second column is the estimated risk (from the challenge
%            entry outputs for event2)
%
%show -(logical) If true (non-zero), this function will plot the percentiles
%
%H    -(scalar) normalized Hosmer-Lemeshow H statistic
%
%Written by Ikaro Silva, 2012
%
%Version 1.4 (25 March 2012)
%
%Reference: Hosmer, David W., Lemeshow, Stanley (2000). Applied Logistic
%Regression, New York : Wiley
%
% http://en.wikipedia.org/wiki/Hosmer–Lemeshow_test
%

%Change Log
%1.0 -> 1.1 Fixed issue where Obs variable was being overestimated
%

D=10; % Use D-iles (i.e., deciles as in Lemeshow's original paper)
N=length(data(:,1));
M=floor(N/D); % M should always be 400 for the challenge
data=sortrows(data,2); % rearrange the matrix by estimated risk
CM=zeros(D,3)+NaN;
centroid=zeros(D-1,1)+NaN;
for i=1:D
    %Set number of deaths proportional to Y true value
    ind2=i*M;      % highest bin in decile D
    ind1=ind2-M+1; % lowest bin
    Obs=sum([data(ind1:ind2,1)]); % number of observed deaths within decile D
    centroid(i)=mean(data(ind1:ind2,2)); % mean estimated risk within decile
    Exp=M*centroid(i); % expected number of deaths within decile
    Htemp=(Obs-Exp)^2/(M*centroid(i)*(1-centroid(i)) + 0.001);
    CM(i,:)=[Obs Exp Htemp];
end

% Hosmer-Lemeshow H statistic, normalized by range of decile risk
H=sum(CM(:,3))/(centroid(10)-centroid(1));

if(show)
    plot(centroid,CM(:,1),'bo','MarkerSize',10)
    grid on;xlim([0 1]);hold on
    plot(centroid,CM(:,2),'rx','MarkerSize',10)
    xlabel('Predicted Risk')
    ylabel('Number of Deaths')
    legend('Observed','Predicted')
end

