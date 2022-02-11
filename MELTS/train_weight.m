
function [W0,P0,b0,bi_0,Zi_0,Z0]=train_weight(X_train, Y_train, view_num, alpha,beta,gamma,theta,ds,d,iters, epsilon,L,Sim,N,S)

c = max(Y_train);

P0 = cell(1,view_num);
Zi_0= cell(1,view_num);
Z_0 = zeros(N,ds);

for i=1:view_num
   [COEFF SCORE latent]=pca(X_train{1,i}');
   Zi_0{1,i} = SCORE(:,1:d);
   Z_0 = Z_0 + SCORE(:,1:ds);
end
Z_0 = Z_0/view_num;

Z0_0 = [Z_0];
for i=1:view_num
    Z0_0 = horzcat(Z0_0,Zi_0{1,i});
end

Y = -1.0 * ones(N, c);

for i = 1 : N
    Y(i, Y_train(i)) = 1.0;
end

I = eye(N);
one_martix = ones(N,N);
H=I - one_martix/N;

W0 = pinv(Z0_0'*H*Z0_0 + (theta/gamma)*eye(ds+view_num*d))*(Z0_0'*H*Y); 

[P0] = InitP( X_train,Z_0,Zi_0,view_num,H);

b0 = Y'*ones(N,1)/N - W0'*Z0_0'*ones(N,1)/N;
for j=1:view_num
    bi_0{1,j}=[Z_0 Zi_0{1,j}]'*ones(N,1)/N - P0{1,j}'*X_train{1,j}*ones(N,1)/N;
end

W = W0;
P =P0;
Zi=Zi_0;
Z=Z_0;
Z0 = Z0_0;
b = b0;
bi = bi_0;

obj = 0;
for j=1:view_num
    obj1 = X_train{1,j}'*P{1,j}+ones(N,1)*bi{1,j}'-[Z Zi{1,j}];
    obj =obj + trace(obj1' * obj1);
end

obj2 = 0;
for j=1:view_num
    for k =1:view_num
        obj2 = obj2 + trace((Zi{1,j}-Zi{1,k})'*(Zi{1,j}-Zi{1,k}))* Sim(j,k);
    end
end
obj2 = alpha/(2*view_num*(view_num-1))*obj2;
obj = obj + obj2;

obj3 = 2*beta/N^2*trace(Z0'*L*Z0);
obj = obj + obj3;

obj4 = gamma*trace((Z0*W+ones(N,1)*b' - Y)'*(Z0*W+ones(N,1)*b' - Y));
obj = obj + obj4;


obj5 = theta*trace(W'*W);
obj = obj + obj5;

obj0=obj;


%% Training...
A = cell(1,view_num);

for i = 1: iters
    
    % optimize P
    for j=1:view_num
        P{1,j}=-pinv(X_train{1,j}*X_train{1,j}')*X_train{1,j}*(ones(N,1)*bi{1,j}'-[Z Zi{1,j}]);
    end
    
    % optimize W
    W = -pinv(Z0'*Z0 + (theta/gamma)*eye(ds+view_num*d))*Z0'*(ones(N,1)*b'-Y);

    % optimize Z
    B = ones(N,1)*b'-Y;
    E_z = Z0(:,ds+1:ds+view_num*d)* W(ds+1:ds+view_num*d,:)+B;
    A_1_sum =zeros(N,ds);
    for j = 1:view_num
        A{1,j} = X_train{1,j}'*P{1,j}+ones(N,1)*bi{1,j}';
        A_1_sum = A_1_sum + A{1,j}(:,1:ds);
    end
    a1 = (2*beta/N^2)*L;
    b1 = view_num * eye(ds) + gamma * W(1:ds,:) * W(1:ds,:)';  
    r1 = A_1_sum - gamma * E_z * W(1:ds,:)';
    Z= sylvester(a1,b1,r1);
    Z0(:,1:ds) = Z;

    % optimize Zi
    for j=1:view_num
        E_zi = Z0 * W - Zi{1,j} * W(ds+(j-1)*d+1:ds+j*d,:) + B;

        Sim_sum = 0;
        Sim_temp = zeros(N,d);
        for o = 1:view_num
            Sim_sum = Sim_sum + Sim(j,o);
            Sim_temp = Sim_temp + Sim(j,o) * Zi{1,o};
        end

        a2 = (2*beta/N^2)*L;
        b2 = eye(d) + alpha/(view_num*(view_num-1))*Sim_sum*eye(d) + gamma * W(ds+(j-1)*d+1:ds+j*d,:) * W(ds+(j-1)*d+1:ds+j*d,:)';
        r2 = A{1,j}(:,ds+1:ds+d) + alpha/(view_num*(view_num-1))*Sim_temp - gamma * E_zi * W(ds+(j-1)*d+1:ds+j*d,:)';
        Zi{1,j}=sylvester(a2,b2,r2);
        Z0(:,ds+(j-1)*d+1:ds+j*d) = Zi{1,j};
    end

   % optimize bi,b
    b = Y'*ones(N,1)/N - W'*Z0'*ones(N,1)/N;
    for j=1:view_num
        bi{1,j}=[Z Zi{1,j}]'*ones(N,1)/N - P{1,j}'*X_train{1,j}*ones(N,1)/N;
    end
    
    P_ch = 0;
    bi_ch = 0;
    b_ch = 0;
    for j=1:view_num
        P_ch = P_ch + trace((P{1,j}-P0{1,j})'*(P{1,j}-P0{1,j}));
        bi_ch = bi_ch + trace((bi{1,j}-bi_0{1,j})'*(bi{1,j}-bi_0{1,j}));
    end
    W_ch = trace((W-W0)'*(W-W0));
    b_ch = trace((b-b0)'*(b-b0));
    para_va = b_ch+P_ch+bi_ch+W_ch;
  
    
    obj = 0;
    for j=1:view_num
        obj1 = X_train{1,j}'*P{1,j}+ones(N,1)*bi{1,j}'-[Z Zi{1,j}];
        obj =obj + trace(obj1' * obj1);
    end
    obj2 = 0;
    for j=1:view_num
        for k =1:view_num
            obj2 = obj2 + trace((Zi{1,j}-Zi{1,k})'*(Zi{1,j}-Zi{1,k}))* Sim(j,k);
        end
    end
    obj2 = alpha/(2*view_num*(view_num-1))*obj2;
    obj = obj + obj2;
    obj3 = 2*beta/N^2*trace(Z0'*L*Z0);
    obj = obj + obj3;
    obj4 = gamma*trace((Z0*W+ones(N,1)*b' - Y)'*(Z0*W+ones(N,1)*b' - Y));
    obj = obj + obj4;
    obj5 = theta*trace(W'*W);
    obj = obj + obj5;
    obj_va = obj0-obj;

%     if (obj_va < epsilon)
%         break; 
%     end
        
    P0 = P; 
    W0 = W; 
    Z_0 = Z;
    b0 = b;
    bi_0 = bi;
    Zi_0=Zi;
    obj0=obj;
    
end

return;
end