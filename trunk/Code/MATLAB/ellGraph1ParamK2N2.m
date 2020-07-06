figure
hold on
k= 1;
for x = .5:.1:8
labels{k} = strcat('x = ', sprintf('%d',x));
n=1;
ell_x = zeros(1,length(0:.01:1));
for j = 1:-.01:0
ell_x(n) = ell(F(x),[j;1-j]);
n=n+1;
end
plot(1:-.01:0,ell_x);
k=k+1;
end
legend(labels{:})