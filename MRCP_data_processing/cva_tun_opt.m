function [COM,PWGR,V,vp,DISC]=cva_tun_opt(pat,label);
%
% Outputs: V. eigenvectors
%          DISC. transformed data w/o labels in the fist column (There are
%          classes-1 variates).
%          PWGR. Structure matrix: Within groups correlation matrix
%                between components and original features (features*components)
%          COM. Index 'Discriminability Power' (%) that shows the contribution
%                of each feature in the canonical space construction. 
%                 
%
% Inputs: data. Original data with labels in the first column.


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%CVA transformation

% Edited by M. Tavella <michele.tavella@epfl.ch> on 05/08/09 12:17:59
labels_old = unique(label);
labels_new = 1:1:length(labels_old);
for j = 1:length(labels_old)
    label(find(label == labels_old(j)),1) = labels_new(j);
end

nFeature=size(pat,2);
nClass=max(label);

% Within covariance matrix
COV = zeros(nFeature,nFeature,nClass);
for k=1:nClass
    classdata=pat(label==k,:);%segmentacio de patrons de cada classe
    COV(:,:,k)= (size(classdata,1)-1)*(cov(classdata));
end
C=sum(COV,3);
M=mean(pat);

% Between covariance matrix
Bg = zeros(nFeature,nFeature,nClass);
for k=1:nClass
    classdata=pat(label==k,:);%segmentacio de patrons de cada classe
    Centg=mean(classdata)-M;
    Bg(:,:,k)=size(classdata,1)*(Centg'*Centg);
end
[V,vp]=svd(pinv(C)*sum(Bg,3));
vp=diag(vp);


DISC=pat*V(:,1:nClass-1);

%Within groups correlation matrix (Correlation beween components and originalfeatures)
WGCOV = zeros(nClass-1+nFeature,nClass-1+nFeature,nClass);
for k=1:nClass
    classdata=[DISC(label==k,:),pat(label==k,:)];%segmentacio de patrons de cada classe
    WGCOV(:,:,k) = (size(classdata,1)-1)*(cov(classdata));
end
PWGCov=sum(WGCOV,3);
PWGR = PWGCov(nClass:end,1:nClass-1);
for i = 1:size(PWGR,1)
    for u = 1:size(PWGR,2)
        PWGR(i,u)=PWGR(i,u)/(sqrt(PWGCov(i+nClass-1,i+nClass-1)*PWGCov(u,u)));
    end
end

%Discriminability Power (%)
vp=vp(1:nClass-1,1)./sum(vp(1:nClass-1,1));
COM=100.*(((PWGR.^2)*(vp))./(sum((PWGR.^2)*(vp))));
