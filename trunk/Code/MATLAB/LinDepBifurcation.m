%% Initial setup
%rng(2147483647);
%rng(2017);
rng(203);
N=10;
K=3;
G=rand(K-1,N);
% a=rand(K-1,1);
% A=a/sum(a);
%A=[.35;.65];
%A=[1.1;-.1]; Don't do this!!!!
%A=ones(3)-diag([.5,.25,.25]);
top=.8:.01:1.2;
sp=rand(1);
A=[sp.*top;(1-sp).*top];
% F=G;

%% Set up the mesh we will be using
fullImg=false;
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
    t=0:resolution:(18/19);
    a=[.05;0;.95];
    b=[.05;.95;0];
    q=[.95;.05;0];
    p=[0;.05;.95];
    %     a=[0;.25;.75];
    %     b=[.36;.64;0];
    s=(ones(1,size(t,2))-t);
    u=t.*a+s.*b;
    w=t.*p+s.*q;
    e1=t.*[1;0;0]+s.*[0;1;0];
    e2=t.*[0;1;0]+s.*[0;0;1];
    e3=t.*[0;0;1]+s.*[1;0;0];
    v=[u,w,e1,e2,e3];
end
%extract the x, y, and z coordinates for plotting
x=v(1,:);
y=v(2,:);
z=v(3,:);

%% Create output files
preimfilepart='pathsmovie_bif2';
imfilepart='imagemovie_bif2';
format='MPEG-4';

PreVW=VideoWriter(preimfilepart,format);
ImVW=VideoWriter(imfilepart,format);

fr=6;

PreVW.FrameRate=fr;
ImVW.FrameRate=fr;%fr<6 is bad!


%% Plot the results of the mesh
r = length(top);
frameMsg = 'Calculating Frame: %d of %d';
W = waitbar(1/r,sprintf(frameMsg,1,r));

predot=5;
scrsz = get(groot,'ScreenSize');

Mov=struct('cdata',[],'colormap',[]);
Imgs=struct('cdata',[],'colormap',[]);

F=[G;A(:,1)'*G];
[img,colors,orbits]=MeshMapK(F,v,10,~fullImg);

if(~fullImg)
    m=size(orbits,1);
    paths=[];
    pathcolors=[];
    formatMsg='Calculating pathcolors: %d of %d';
    h=waitbar(0/m,sprintf(formatMsg,0,m));
    for i=1:m
        tmp=squeeze(orbits(i,:,any(orbits(i,:,:),2)));
        paths=[paths,tmp];
        pathcolors=[pathcolors,colors(i)*ones(1,size(tmp,2))];
        waitbar(i/m,h,sprintf(formatMsg,i,m))
    end
    delete(h);
end

figure2=figure('OuterPosition',[scrsz(3)/2 scrsz(4)/2 scrsz(3)/2 scrsz(4)/2]);
axes2 = axes('Parent',figure2);
scatter3(axes2,img(1,:),img(2,:),img(3,:),predot*10,1:size(img,2),'filled');
view(axes2,[135 25]);
grid(axes2,'on');
Mov(1)=getframe(figure2);
    
if(~fullImg)
    xpaths=paths(1,:); ypaths=paths(2,:); zpaths=paths(3,:);
    figure3=figure('OuterPosition',[1 50 scrsz(3)/2 scrsz(4)/2]);
    axes3 = axes('Parent',figure3);
    scatter3(axes3,xpaths,ypaths,zpaths,predot,pathcolors,'filled');
    view(axes3,[135 25]);
    Imgs(1)=getframe(figure3);
end


for j=2:r
    waitbar(j/r,W,sprintf(frameMsg,j,r))
    F=[G;A(:,j)'*G];
    [img,colors,orbits]=MeshMapK(F,v,10,~fullImg);
    
    
    %extract the paths and colors for each path, if not working with entire
    %image
    if(~fullImg)
        m=size(orbits,1);
        paths=[];
        pathcolors=[];
        formatMsg='Calculating pathcolors: %d of %d';
        h=waitbar(0/m,sprintf(formatMsg,0,m));
        for i=1:m
            tmp=squeeze(orbits(i,:,any(orbits(i,:,:),2)));
            paths=[paths,tmp];
            pathcolors=[pathcolors,colors(i)*ones(1,size(tmp,2))];
            waitbar(i/m,h,sprintf(formatMsg,i,m))
        end
        delete(h);
    end
    
    
    %     % The plot of where the points go
    %     figure1=figure('OuterPosition',[1 scrsz(4)/2 scrsz(3)/2 scrsz(4)/2]);
    %     axes1 = axes('Parent',figure1);
    %     scatter3(axes1,x,y,z,predot,colors,'filled');
    %     view(axes1,[135 25]);
    %     grid(axes1,'on');
    %
    % plot of the image
    %figure2=figure('OuterPosition',[scrsz(3)/2 scrsz(4)/2 scrsz(3)/2 scrsz(4)/2]);
    axes2 = axes('Parent',figure2);
    scatter3(axes2,img(1,:),img(2,:),img(3,:),predot*10,1:size(img,2),'filled');
    view(axes2,[135 25]);
    grid(axes2,'on');
    Mov(j)=getframe(figure2);
    
    % plot of orbits
    if(~fullImg)
        xpaths=paths(1,:); ypaths=paths(2,:); zpaths=paths(3,:);
        %figure3=figure('OuterPosition',[1 50 scrsz(3)/2 scrsz(4)/2]);
        axes3 = axes('Parent',figure3);
        scatter3(axes3,xpaths,ypaths,zpaths,predot,pathcolors,'filled');
        view(axes3,[135 25]);
        Imgs(j)=getframe(figure3);
    end
end

%% Write the movies to file
open(PreVW);
writeVideo(PreVW,Mov);
close(PreVW);

open(ImVW);
writeVideo(ImVW,Imgs);
close(ImVW);
%close('all');