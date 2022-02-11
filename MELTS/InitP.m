function [ P ] = InitP( X_train,Z,Zi,view_num,H)
P = cell(1,view_num);

for i = 1:view_num 
    Xi=X_train{1,i};
    P{1,i} = pinv(Xi*H*Xi')*(Xi*H*[Z Zi{1,i}]);
end

end

