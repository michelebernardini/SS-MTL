%% FUNCTION combine_input
%   transform X, Y cell array input into matrix/vector input with
%   ind and ssize. 
%
%% INPUT
%   X: {[n_i, d]}*t
%   Y: {[n_i, 1]}*t
%
%% OUTPUT
%   Xcmb: [\sum_i n_i, d]
%   Ycmb: [\sum_i n_i, 1]
%   ind: index
%   ssize: the array of size


function [Xcmb, Ycmb, ind, ssize] = combine_input(X, Y)

t = length(X);

Xcmb = [];
Ycmb = [];

ind = 1;
ssize = zeros(t, 1);

for i = 1: t
    Xcmb = cat(1, Xcmb,  X{i});
    Ycmb = cat(1, Ycmb,  Y{i});
    ind  = cat(1, ind,   ind(end) + size(X{i},1));
    ssize(i) = size(X{i},1);
end

end