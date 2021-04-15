function X = multi_transpose (X)

for i = 1:length(X)
    X{i} = X{i}';
end

end