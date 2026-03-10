function [risk,prediction]=physionet2012(time,param,value)
% [risk,prediction]=physionet2012(time,param,value)
% 
% Sample Submission for the PhysioNet 2012 Challenge. Variables are:
% 
% time       - (Nx1 Cell Array) Cell array containing time of measurement
% param      - (Nx1 Cell Array) Cell array containing type (category) of
%               measurement
% value      - (Nx1 Double Array) Double array containing value of measurement
%
% 
% risk       - (Scalar) estimate of the risk of the patient dying in hospital
% prediction - (Logical)Binary classification if the patient is going to die
%               in the hospital (1 - Died, 0 - Survived)
% 
% Example:
% [risk,prediction]=physionet2012(time,param,value)
% 
% Written by Ikaro Silva, 2012
% 
% Version 1.1
%
%The sample submission calculates the in hospital death 
%probability based on Bayesian Rule conditioned on SAPS_SCORE
%and PDF fitting based on training Set A. 
%
%    P(dying | SAPS_SCORE) = P(dying)*P(SAPS_SCORE|Dying)/
%    (P(dying)*P(SAPS_SCORE|dying) + P(living)*P(SAPS_SCORE|living))
%
%Calculate likelihood, P(SAPS_SCORE|Dying), based polynomial fitting of the CDF
%of the training data A conditioned on those that died (values of the fit
%are hardcoded below)
%
% saps_died=SAPS_SCORE(ihd==1); %ihd is logical vector where 1 = in hospital death
% MX_SAPS=4*14;
% [Ndied,xx]=hist(saps_died,[0:MX_SAPS]);
% md=nanmean(saps_died);
% st_d=nanstd(saps_died)
% pdf_died=normpdf(xx,md,st_d);
% plot(xx,Ndied./sum(Ndied));hold on;grid on;plot(xx,pdf_died,'r') %Check fit
%
% Repeat to get conditional probability on those that lived using Extreme
% Value Distribution
% saps_alive=SAPS_SCORE(ihd==0); %ihd is logical vector where 1 = in hospital death
% [Nalive,xx]=hist(saps_alive,[0:MX_SAPS]);
% parmhat = evfit(saps_alive(~isnan(saps_alive)));
% pdf_alive=evpdf(xx,parmhat(1),parmhat(2));
% figure
% %This fitting is not very good, but for the purpose of an example will do
% plot(xx,Nalive./sum(Nalive));hold on;grid on;plot(xx,pdf_alive,'r')
 
TH=0.2038; %Threshold for classifying the patient as non-survivor
           %Determined through maximization of the min(PPV,Sensitivity) on
           %training Set A

%Calculate SAPS SCORE
truncated=1; %Attempt to calculate SAPS even if some variables are missing
[SAPS_SCORE]=saps_score(time,param,value,truncated);

%Hard-coded values based on the analysis commented above
md=017.644699140401148; 
st_d=4.677514903313259;
likelihood_saps_dying=normpdf(SAPS_SCORE,md,st_d);

parmhat =[15.927414293093566 4.930958270510013];
likelihood_saps_living=evpdf(SAPS_SCORE,parmhat(1),parmhat(2));
pdf_dying= 0.1385;%Based on set A

%Calcuate the Posteriori Probability given the SAPS_SCORE
posteriori_num=  (pdf_dying*likelihood_saps_dying);
posteriori_den=  (pdf_dying*likelihood_saps_dying + (1-pdf_dying)*likelihood_saps_living);

risk=posteriori_num/posteriori_den;
prediction=risk>=TH;
