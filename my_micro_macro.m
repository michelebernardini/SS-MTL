function [fmacro, macroprecision, macrorecall] = my_micro_macro( pred_label, orig_label)

%compute macro precision, recall and fscore

mat=confusionmat(orig_label, pred_label);
len=size(mat,1);
macroTP=zeros(len,1);
macroFP=zeros(len,1);
macroFN=zeros(len,1);
macroP=zeros(len,1);
macroR=zeros(len,1);
macroF=zeros(len,1);
for i=1:len
    macroTP(i)=mat(i,i);
    macroFP(i)=sum(mat(:, i))-mat(i,i);
    macroFN(i)=sum(mat(i,:))-mat(i,i);
    if macroTP(i)==0 && macroFP(i)==0 
        macroP(i)=0;
    else    
    macroP(i)=macroTP(i)/(macroTP(i)+macroFP(i)+eps);
    end
    if macroTP(i)==0 && macroFN(i)==0 
        macroR(i)=0;
    else    
    macroR(i)=macroTP(i)/(macroTP(i)+macroFN(i)+eps);
    end
    if macroP(i)==0 && macroR(i)==0 
        macroF(i)=0;
    else    
    macroF(i)=2*macroP(i)*macroR(i)/(macroP(i)+macroR(i));
    end
end
fmacro=mean(macroF);
macroprecision=mean(macroP);
macrorecall=mean(macroR);
end