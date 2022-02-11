function [X_train,Y_train,X_val,Y_val,X_traVal,Y_traVal,X_test,Y_test] = split_traVal_test(outCv,view_num,X_cross,Y_cross)
X_test = X_cross{1,outCv};
Y_test = Y_cross{1,outCv};
X_traVal = cell(1,view_num);
Y_traVal = [];

X_train = cell(1,view_num);
Y_train = [];

X_val = cell(1,view_num);
Y_val = [];
temp = 0;
for tra_val = 1:5
    if (tra_val ~= outCv)
        for tra_view = 1:view_num
            X_traVal{1,tra_view} = [X_traVal{1,tra_view};X_cross{1,tra_val}{1,tra_view}];   
        end
        Y_traVal = [Y_traVal;Y_cross{1,tra_val}];
        
        if(temp==0)
            for tra_view = 1:view_num
                X_val{1,tra_view} = [X_val{1,tra_view};X_cross{1,tra_val}{1,tra_view}];   
            end
            Y_val = [Y_val;Y_cross{1,tra_val}];
        else
            for tra_view = 1:view_num
                X_train{1,tra_view} = [X_train{1,tra_view};X_cross{1,tra_val}{1,tra_view}];   
            end
            Y_train = [Y_train;Y_cross{1,tra_val}];
        end
        temp = temp+1;
    end
end

end

