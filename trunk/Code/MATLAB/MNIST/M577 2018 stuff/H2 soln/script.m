pca_dimension = Inf;
%Uncomment the next line to bypass PCA
%pca_dimension = Inf;       
num_epochs = 10000;
digit0 = 3
digit1 = 5

[X,T,Y,W,V] = do_digit_pair(digit0,digit1,num_epochs,pca_dimension);

show_misclassified = false;
[Ytest,Ttest] = do_test_data(digit0, digit1, W, V, show_misclassified);

plotconfusion(T',Y','Training Data',...
              Ttest',Ytest','Test Data'),


