% Learn positive rule
X=[-1 0;
   0 -1;
   -1 -1;
   1  0;
   0  1 ];
[N,I]=size(X);
t=[0;0;0;1;1]
for r=1:size(X,1)
    fprintf('%2d %2d --> %6.2f\n', X(r,1),X(r,2),t(r));
end
L=1000;
eta=0.1;
alpha=0.1;
[w,e,o]=backprop(X,t,...
                 'L',L,...
                 'eta',eta,...
                 'alpha',alpha, ...
                 'Display','off' ...
                 );
display(w);
plot(1:L,log2(o));xlabel('Epoch'),ylabel('Log2 Fitness');
   
   