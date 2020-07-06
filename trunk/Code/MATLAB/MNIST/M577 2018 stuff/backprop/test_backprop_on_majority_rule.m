% Learn majority rule
X=[0 0 0;
   1 0 0;
   0 1 0;
   0 0 1;
   1 1 0;
   1 0 1;
   0 1 1;
   1 1 1];
[N,I]=size(X);
t=sum(X,2)>I/2;
disp('Learning majority table:');
for r=1:8
    fprintf('%d %d %d --> %6.2f\n', X(r,1),X(r,2),X(r,3),t(r));
end
L=1000;
eta=0.3;
alpha=0.1;
[w,e,o]=backprop(X,t,...
                 'L',L,...
                 'eta',eta,...
                 'alpha',alpha,...
                 'Display','on' ...
                 );
display(w);
display(o);
   
   