%% Something about GMMData.m or stablepointNewton.m has problems
% possibly when d=1?!?
rng('default');
K=3;
N=1000;
F=rand(K,N);
p=1/K*ones(K,1);
resolution=.005;
v=zeros(3,10);
k=1;
for i=.4:resolution:.6
for j=.4:resolution:(.4+abs(.6-i))
v(:,k)=[i;j;abs(1-i-j)];
k=k+1;
end
end
[Img,Colors,Orbits]=MeshMapK(F,v,10,true,'newton');
x=v(1,:);
y=v(2,:);
z=v(3,:);
scrsz = get(groot,'ScreenSize');
predot=5;
figure1=figure('OuterPosition',[1 scrsz(4)/2 scrsz(3)/2 scrsz(4)/2]);
axes1 = axes('Parent',figure1);
scatter3(axes1,x,y,z,predot,Colors,'filled');
view(axes1,[135 25]);
grid(axes1,'on');
figure1=figure('OuterPosition',[1 scrsz(4)/2 scrsz(3)/2 scrsz(4)/2]);
axes1 = axes('Parent',figure1);
scatter3(axes1,x,y,z,predot,Colors,'filled');
view(axes1,[135 25]);
grid(axes1,'on');
figure2=figure('OuterPosition',[scrsz(3)/2 scrsz(4)/2 scrsz(3)/2 scrsz(4)/2]);
axes2 = axes('Parent',figure2);
scatter3(axes2,Img(1,:),Img(2,:),Img(3,:),10,1:size(Img,2),'filled');
view(axes2,[135 25]);
grid(axes2,'on');
figure2=figure('OuterPosition',[scrsz(3)/2 scrsz(4)/2 scrsz(3)/2 scrsz(4)/2]);
axes2 = axes('Parent',figure2);
scatter3(axes2,Img(1,:),Img(2,:),Img(3,:),10,1:size(Img,2),'filled');
view(axes2,[135 25]);
grid(axes2,'on');
F
p
stablepointNewton(F,1/3*ones(3,1),10)
stablepointNewton(F,p,10)
stablepoint(F,1/3*ones(3,1),10,'ratio')
stablepoint(F,1/3*ones(3,1),10,'diff')
p
[F,p]=GMMData(3,1000000,1,123876452);
stablepoint(F,1/3*ones(3,1),10,'diff')
stablepointNewton(F,1/3*ones(3,1),14)
sum(ans)
%stablepointNewton(F,1/3*ones(3,1),16)
stablepointNewton(F,1/3*ones(3,1),15)

%% This seems to work okay

[F,p]=GMMData(3,1000000,3,203);
stablepointNewton(F,1/3*ones(3,1),15)
stablepoint(F,1/3*ones(3,1),10,'diff')
p'
q1=stablepointNewton(F,1/3*ones(3,1),15)
q2=stablepoint(F,1/3*ones(3,1),10,'diff')
p'
norm(q1-p')
norm(q2-p')
norm(q2-q1)
q2-q1
[F,p]=GMMData(3,10000000,3,203);
q1=stablepointNewton(F,1/3*ones(3,1),15)
q2=stablepoint(F,1/3*ones(3,1),10,'diff')
p'
norm(q1-p')
norm(q2-p')
norm(q2-q1)