function [ L,S] = BuildL( Y_train,N )
%UNTITLED2 �˴���ʾ�йش˺�����ժҪ
S = zeros(N,N);
D = zeros(N,N);

% �������ƶȾ���
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

% ����Ⱦ���
for i=1:N
    for j=1:N
        D(i,i)=D(i,i)+S(i,j);
    end
end

% ����������˹����
L=D-S;
end

