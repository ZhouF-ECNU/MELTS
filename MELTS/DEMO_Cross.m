load('../data/myDataCross.mat');
iters =30;
epsilon = 1e-2;
parameters_lists=cell(1,5);
alpha = cell(1,5);
beta = cell(1,5);
gamma = cell(1,5);
theta = cell(1,5);
ds = cell(1,5);
d = cell(1,5);

view_num = size(X_cross{1,1}, 2);

init_d = size(X_cross{1,1}{1,1},2);
for i =1:view_num
    init_d = min(init_d,size(X_cross{1,1}{1,i},2));
end

pred_result = cell(1,5);

for i = 1:5
    alpha{1,i} = [1e-3,1e-2,1e-1,1,10,100,1000];
    beta{1,i} = [1e-3,1e-2,1e-1,1,10,100,1000];
    gamma{1,i} = [1e-3,1e-2,1e-1,1,10,100,1000];
    theta{1,i} = [1e-3,1e-2,1e-1,1,10,100,1000];
    ds{1,i} = [1,75,150];
    d{1,i} = [1,75,150];
end


for i = 1:5
    parameters_lists{1,i} = ones(1,6);
end
    
outCV_acc_lists = [];
for outCv =1:5

    [X_train,Y_train,X_val,Y_val,X_traVal,Y_traVal,X_test,Y_test] = split_traVal_test(outCv,view_num,X_cross,Y_cross);

    for view=1:view_num
        [X_traVal{1,view},ps] = mapstd(X_traVal{1,view}',0,1);
        X_test{1,view} = mapstd('apply',X_test{1,view}',ps);
        X_train{1,view} = mapstd('apply',X_train{1,view}',ps);
        X_val{1,view} = mapstd('apply',X_val{1,view}',ps);
    end

    max_acc = 0;

    %% ---------- train the model ----------
    for j = 1 : size(alpha{1,outCv},2)
        for k = 1 :size(beta{1,outCv},2)
            for l = 1:size(gamma{1,outCv},2)
                for h = 1:size(theta{1,outCv},2)
                    for g = 1:size(ds{1,outCv},2)
                        for f = 1:size(d{1,outCv},2)
                            sum_acc = 0;

                            [~,N_train] = size(X_train{1,1});

                            [L,S] = BuildL(Y_train,N_train);

                            Sim = zeros(view_num,view_num);
                            for i = 1:view_num
                                for o =1:view_num
                                    Sim(i,o) = distcorr(X_train{1,i}',X_train{1,o}');
                                    if i==o
                                        Sim(i,o)=0;
                                    end
                                end
                            end

                            acc=0;

                            [W0,P0,b0,bi_0,Zi_0,Z0]  = ...
                            train_weight(X_train, Y_train, view_num, alpha{1,outCv}(j),beta{1,outCv}(k),gamma{1,outCv}(l),theta{1,outCv}(h),ds{1,outCv}(g),d{1,outCv}(f),iters, epsilon,L,Sim,N_train,S);

                            [~,N] = size(X_val{1,1});
                            Z2 = [];
                            Z1 = zeros(N,ds{1,outCv}(g));
                            for i = 1:view_num
                                Xi=X_val{1,i};          
                                Z=Xi'*P0{1,i}+ones(N,1)*bi_0{1,i}';
                                Z1 = Z1+Z(:,1:ds{1,outCv}(g));
                                Z2 = horzcat(Z2,Z(:,ds{1,outCv}(g)+1:ds{1,outCv}(g)+d{1,outCv}(f)));
                            end

                            Z_list = [Z1/view_num Z2];
                            Y_pred=Z_list*W0 + ones(N,1)*b0';
                            [aa,bb] = max(Y_pred,[],2); 
                            acc = sum(Y_val==bb)/length(Y_val);

                            if acc > max_acc
                                max_acc = acc;
                                parameters_lists{1,outCv}(1,1)=j;
                                parameters_lists{1,outCv}(1,2)=k;
                                parameters_lists{1,outCv}(1,3)=l;
                                parameters_lists{1,outCv}(1,4)=h;
                                parameters_lists{1,outCv}(1,5)=g;
                                parameters_lists{1,outCv}(1,6)=f;
                            end

                        end
                    end
                end
            end
        end
    end
 
    [~,N_train] = size(X_train{1,1});
    
    [L,S] = BuildL(Y_train,N_train);

    Sim = zeros(view_num,view_num);
    for i = 1:view_num
        for o =1:view_num
            Sim(i,o) = distcorr(X_train{1,i}',X_train{1,o}');
            if i==o
                Sim(i,o)=0;
            end
        end
    end

    acc=0;

    [W0,P0,b0,bi_0,Zi_0,Z0]  = ...
    train_weight(X_train, Y_train, view_num, alpha{1,outCv}(parameters_lists{1,outCv}(1,1)),beta{1,outCv}(parameters_lists{1,outCv}(1,2)),gamma{1,outCv}(parameters_lists{1,outCv}(1,3)),theta{1,outCv}(parameters_lists{1,outCv}(1,4)),ds{1,outCv}(parameters_lists{1,outCv}(1,5)),d{1,outCv}(parameters_lists{1,outCv}(1,6)),iters, epsilon,L,Sim,N_train,S);

    [~,N] = size(X_test{1,1});
    Z2 = [];
    Z1 = zeros(N,ds{1,outCv}(parameters_lists{1,outCv}(1,5)));
    for i = 1:view_num
        Xi=X_test{1,i};       
        Z=Xi'*P0{1,i}+ones(N,1)*bi_0{1,i}';
        Z1 = Z1+Z(:,1:ds{1,outCv}(parameters_lists{1,outCv}(1,5)));
        Z2 = horzcat(Z2,Z(:,ds{1,outCv}(parameters_lists{1,outCv}(1,5))+1:ds{1,outCv}(parameters_lists{1,outCv}(1,5))+d{1,outCv}(parameters_lists{1,outCv}(1,6))));
    end

    Z_list = [Z1/view_num Z2];
    Y_pred=Z_list*W0 + ones(N,1)*b0';
    [aa,bb] = max(Y_pred,[],2); 

    Y_pred_p = softmax(Y_pred');
    Y_pred_p = Y_pred_p';

    pred_result{1,outCv} = Y_pred_p;

    acc = sum(Y_test==bb)/length(Y_test);
    fprintf('OutCv:%d -- Accuracy : %.5f\n',outCv,acc);
    outCV_acc_lists = [ outCV_acc_lists,acc];

end
fprintf('Final Accuracy : %.5f\n', mean(outCV_acc_lists));
fprintf('Final Std : %.5f\n', std(outCV_acc_lists));

return;
