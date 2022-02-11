load('Synthetic.mat');

view_num = size(X,2);
indices = crossvalind('Kfold',Y,5);
X_cross = cell(1,5);
Y_cross = cell(1,5);
for k=1:5
    test = (indices == k);
    train = ~test;

    for i=1:view_num
        X_subCross{1,i} = X{1,i}(test,:);
    end
    Y_subCross = Y(test);
    
    X_cross{1,k} = X_subCross;
    Y_cross{1,k} = Y_subCross;
end

save('exCrossSynthetic.mat','X_cross','Y_cross');