%% Initial setup
%rng(2147483647);
%rng(2017);
rng(203);
N=10;
K=3;
G=rand(K,N);
%a=rand(K-1,1);
%A=a/sum(a);
%A=[.35;.65];
%A=[1.1;-.1]; Don't do this!!!!
%A=ones(3)-diag([.5,.25,.25]);
%F=[G;A'*G];
F=G;

%% Set up the mesh we will be using
fullImg=true;
% V will be the 3 dimensional simplex (option 1)
resolution=.01;

if(fullImg)
    v=zeros(size(F));
    k=1;
    for i=0:resolution:1
        for j=0:resolution:1-i
            v(:,k)=[i;j;abs(1-i-j)];
            k=k+1;
        end
    end
% V as the line between 2 points in the simplex(option 2)
else
    t=0:resolution:1;
    a=[.95;.05;0];
    b=[0;.05;.95];
%     a=[0;.25;.75];
%     b=[.36;.64;0];
    s=(ones(1,size(t,2))-t);
    u=t.*a+s.*b;
    e1=t.*[1;0;0]+s.*[0;1;0];
    e2=t.*[0;1;0]+s.*[0;0;1];
    e3=t.*[0;0;1]+s.*[1;0;0];
    v=[u,e1,e2,e3];
end
%extract the x, y, and z coordinates for plotting
x=v(1,:);
y=v(2,:);
z=v(3,:);

%% Plot the results of the mesh
[img,colors]=SingleMeshMapK(F,v,10);

predot=5;

scrsz = get(groot,'ScreenSize');

% The plot of where the points go
figure1=figure('OuterPosition',[1 scrsz(4)/2 scrsz(3)/2 scrsz(4)/2]);
axes1 = axes('Parent',figure1);
scatter3(axes1,x,y,z,predot,colors,'filled');
view(axes1,[135 25]);
grid(axes1,'on');

% plot of the image
figure2=figure('OuterPosition',[scrsz(3)/2 scrsz(4)/2 scrsz(3)/2 scrsz(4)/2]);
axes2 = axes('Parent',figure2);
scatter3(axes2,img(1,:),img(2,:),img(3,:),predot,1:size(img,2),'filled');
view(axes2,[135 25]);
grid(axes2,'on');