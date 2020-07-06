G = @(F,p) F./(p'*F);
H_ell = @(F,p,N) -1/N.*G(F,p)*G(F,p)';
options = {12, 'diff', true};
stabPt = @(F,p) stablepoint(F,p,options{:});
rng('default')
F =@(x) [1,x,.001;2,1,.001;.001,.001,1];
F(1)
[K,N] = size(F(0.1));
ell = @(X,y) log(prod(y'*X,2))./N;
tol = .001;
sig = @(K) rand(K,1)*tol;
p_0 = @(K) ones(K,1)./K+sig(K);
H = @(x) sum(x.*log(x));
resolution = .01;
n = 1
for i = 1:-resolution:0
for j = 1-i:-resolution:0
p = [abs(i);abs(j);abs(1-i-j)];
R_H(:,n) = [i,j,H(p)];
R_ell(:,n) = [i,j,ell(F(1),p)];
n = n+1;
end
end
ell = @(X,y) log(prod(y'*X,2))./N;
n = 1
for i = 1:-resolution:0
for j = 1-i:-resolution:0
p = [abs(i);abs(j);abs(1-i-j)];
R_H(:,n) = [i,j,H(p)];
R_ell(:,n) = [i,j,ell(F(1),p)];
n = n+1;
end
end
plot3(R_ell(1,:),R_ell(2,:),R_ell(3,:))
figure
plot3(R_ell(1,:),R_ell(2,:),R_H(3,:))
plot3(R_ell(1,:),R_ell(2,:),-R_H(3,:))
S = [R_ell(1,:);R_ell(2,:); -R_ell(3,:)-.001*R_H(3,:)];
figure
plot3(R_ell(1,:),R_ell(2,:),S(3,:))
S = [R_ell(1,:);R_ell(2,:); -R_ell(3,:)-.05*R_H(3,:)];
plot3(R_ell(1,:),R_ell(2,:),S(3,:))
contour3(R_ell(1,:),R_ell(2,:),S(3,:))
stabPt(F(1),p_0(K))
[elmax,elIdx] = max(R_ell(3,:))
R_ell(1,elIdx)
R_ell(2,elIdx)
S(2,elIdx)
[smax,sIdx] = max(s(3,:))
[smax,sIdx] = max(S(3,:))
S(2,sIdx)
S(1,sIdx)
[smin,sidx] = min(S(3,:))
S(2,sidx)
S(1,sidx)
S = [R_ell(1,:);R_ell(2,:); R_ell(3,:)-.001*R_H(3,:)];
plot3(R_ell(1,:),R_ell(2,:),S(3,:))
n = 1
for i = 1:-resolution:0
for j = 1-i:-resolution:0
p = [abs(i);abs(j);abs(1-i-j)];
R_H(:,n) = [i,j,H(p)];
R_ell(:,n) = [i,j,ell(F(1),p)];
n = n+1;
end
end
S = [R_ell(1,:);R_ell(2,:); R_ell(3,:)-.1*R_H(3,:)];
plot3(R_ell(1,:),R_ell(2,:),S(3,:))
resolution = .001;
n = 1
for i = 1:-resolution:0
for j = 1-i:-resolution:0
p = [abs(i);abs(j);abs(1-i-j)];
R_H(:,n) = [i,j,H(p)];
R_ell(:,n) = [i,j,ell(F(1),p)];
n = n+1;
end
end
S = [R_ell(1,:);R_ell(2,:); R_ell(3,:)-.1*R_H(3,:)];
plot3(R_ell(1,:),R_ell(2,:),S(3,:))
S = [R_ell(1,:);R_ell(2,:); R_ell(3,:)-.5*R_H(3,:)];
plot3(R_ell(1,:),R_ell(2,:),S(3,:))
F =@(x) [1,x,.001;2,1,.001;1,2,5];
stabPt(F(1),p_0(K))
stabPt(F(2),p_0(K))
stabPt(F(3),p_0(K))
stabPt(F(5),p_0(K))
F =@(x) [1,x,.001;2,1,2;1,1,5];
stabPt(F(1),p_0(K))
stabPt(F(2),p_0(K))
stabPt(F(3),p_0(K))
for i = 1:-resolution:0
for j = 1-i:-resolution:0
v(:,n) = [abs(i),abs(j),abs(1-i-j)];
n=n+1;
end
end
[X,Y] = meshgrid(v(1,:),v(2,:));
resolution = .01;
for i = 1:-resolution:0
for j = 1-i:-resolution:0
v(:,n) = [abs(i),abs(j),abs(1-i-j)];
n=n+1;
end
end
[X,Y] = meshgrid(v(1,:),v(2,:));
resolution = .05;
resolution*20
for i = 1:-resolution:0
for j = 1-i:-resolution:0
v(:,n) = [abs(i),abs(j),abs(1-i-j)];
n=n+1;
end
end
[X,Y] = meshgrid(v(1,:),v(2,:));
L = length(v(1,:))
resolution = .01;
x = -1:resolution:1;
y = x;
[X,Y] = meshgrid(x,y);
Z = 1-exp(X)-exp(Y);
contour3(X,Y,Z)
x = -e:resolution:0;
x = -exp(1):resolution:0;
y = x;
[X,Y] = meshgrid(x,y);
Z = 1-exp(X)-exp(Y);
contour3(X,Y,Z)
contour3(exp(X),exp(Y),Z)
contour3(exp(X),exp(Y),Z,50)
x = 1:-resolution:0;
y = x;
[X,Y] = meshgrid(x,y);
Z = X.*Y;
contour3(X,Y,Z)
Z = abs(1-X-Y);
contour3(X,Y,Z)
contour3(X,Y,Z,100)
f = @(X) X.*(max(X)./sum(X))
Z = f([x',y']);
contour3(X,Y,Z,100)
Z = f([x',y'])';
contour3(x,y,Z,100)
contour3(x,y',Z,100)
f = @(X,Y) X.*(max(X,Y)./sum(X,Y))
Z = f([X,Y])';
Z = f(X,Y)';
max(X,Y)
max(X)
f = @(X,Y) [X.*(max(X,Y)./(X+Y)),Y.*(max(X,Y)./(X+Y))]
Z = ell(F(2),[f(X,Y),1-(sum(f(X,Y)))]);
size(1-(sum(f(X,Y))0
size(1-(sum(f(X,Y)))
size(1-(sum(f(X,Y))))
f(X,Y)
f = @(X,Y) cat(2,X.*(max(X,Y)./(X+Y)),Y.*(max(X,Y)./(X+Y)));
f(X,Y)
f = @(X,Y) cat(3,X.*(max(X,Y)./(X+Y)),Y.*(max(X,Y)./(X+Y)));
f(X,Y)
f = @(X,Y) cat(1,X.*(max(X,Y)./(X+Y)),Y.*(max(X,Y)./(X+Y)));
f(X,Y)
f = @(X,Y) cat(3,X.*(max(X,Y)./(X+Y)),Y.*(max(X,Y)./(X+Y)));
Z = ell(F(2),[f(X,Y),1-(sum(f(X,Y)))]);
Z = ell(F(2),[f(X,Y),1-(sum(f(X,Y),3))]);
size(1-(sum(f(X,Y),3)))
Z = ell(F(2),cat(3,f(X,Y),1-(sum(f(X,Y),3))));
Z = ell(F(2),permute(cat(3,f(X,Y),1-(sum(f(X,Y),3))),3,1,2));
permute(cat(3,f(X,Y),1-(sum(f(X,Y),3))),[3,1,2])
simp =permute(cat(3,f(X,Y),1-(sum(f(X,Y),3))),[3,1,2]);
sum(simp)
simp(:,101,101)
simp(:,101,101) =[0;0;0];
simp(:,101,101)
pi = simp(:,101,101)
pi.*max(pi)./sum(pi)
for i = 1:101
for j= 1:101
Z(i,j) = ell(F(2),squeeze(simp(:,i,j)));
end
end
Coords = f(X,Y);
figure
contour3(coords(:,:,1),coords(:,:,2),Z,100)
contour3(Coords(:,:,1),Coords(:,:,2),Z,100)
contour3(Coords(:,:,1),Coords(:,:,2),Z,10000)
contour3(Coords(:,:,1),Coords(:,:,2),Z,1000)
figure
plot3(R_ell(1,:),R_ell(2,:),R_ell(3,:))
plot3(R_ell(1,:),R_ell(2,:),R_ell(3,:),':')
resolution = .001;
n = 1
for i = 1:-resolution:0
for j = 1-i:-resolution:0
p = [abs(i);abs(j);abs(1-i-j)];
R_H(:,n) = [i,j,H(p)];
R_ell(:,n) = [i,j,ell(F(1),p)];
n = n+1;
end
end
plot3(R_ell(1,:),R_ell(2,:),R_ell(3,:),':')
resolution = .001;
n = 1
for i = 1:-resolution:0
for j = 1-i:-resolution:0
p = [abs(i);abs(j);abs(1-i-j)];
R_H(:,n) = [i,j,H(p)];
R_ell(:,n) = [i,j,ell(F(2),p)];
n = n+1;
end
end
plot3(R_ell(1,:),R_ell(2,:),R_ell(3,:),':')
hold on
contour3(Coords(:,:,1),Coords(:,:,2),Z,1000)
[m,i] = max(R_ell(3,:))
stabPt(F(2),p_0(K))
R_ell(2,i)
plot3(R_ell(1,i),R_ell(2,i),R_ell(3,i),'ok','MarkerFaceColor','r')