function [ P ] = InitP( X_train,Z,Zi,view_num,H)
%UNTITLED5 此处显示有关此函数的摘要
% 保存P1-Pk
P = cell(1,view_num);

for i = 1:view_num
    %Xi维度：di*n     
    Xi=X_train{1,i};
    %Pi维度：di*(d+ds)
    P{1,i} = pinv(Xi*H*Xi')*(Xi*H*[Z Zi{1,i}]);
%     P{1,i} = lsqminnorm(Xi*H*Xi',Xi*H*[Z Zi{1,i}]);
end

end

