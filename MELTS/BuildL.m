function [ L,S] = BuildL( Y_train,N )
%UNTITLED2 此处显示有关此函数的摘要
S = zeros(N,N);
D = zeros(N,N);

% 计算相似度矩阵
for i=1:N
    for j=1:N
        if Y_train(i)==Y_train(j)
            S(i,j)=1;
%             S(i,j)=6;
        else
%             S(i,j)=-1;
            S(i,j)=0;
        end 
    end
end

% 计算度矩阵
for i=1:N
    for j=1:N
        D(i,i)=D(i,i)+S(i,j);
    end
end

% 计算拉普拉斯矩阵
L=D-S;
end

