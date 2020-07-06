%% Initial setup
%rng(2147483647);
%rng(2017);
%rng(203,'twister');
N=10;
K=3;
G=randn(K,N);
%a=rand(K-1,1);
%A=a/sum(a);
%A=[.35;.65];
%A=[1.1;-.1]; Don't do this!!!!
%A=ones(3)-diag([.5,.25,.25]);
%F=[G;A'*G];
F=1/sqrt(2*pi).*exp(-(G.^2));

%% Set up the mesh we will be using
fullImg=true;
% V will be the 3 dimensional simplex (option 1)
resolution=.01;

if(fullImg)
    %can the following be vectorized!?!
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

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Create output files
preimfilepart='preimagemovie_partial';
imfilepart='imagemovie_partial';
format='MPEG-4';

PreVW=VideoWriter(preimfilepart,format);
ImVW=VideoWriter(imfilepart,format);

fr=6;

PreVW.FrameRate=fr;
ImVW.FrameRate=fr;%fr<6 is bad!

%% Create the frames
loops=10;
Mov(loops+4)=struct('cdata',[],'colormap',[]);
Imgs(loops+4)=struct('cdata',[],'colormap',[]);
predot=5;

scrsz = get(groot,'ScreenSize');

loopMsg='Creating frame %d of %d';
W=waitbar(1/loops,sprintf(loopMsg,1,loops));
% Draw lines to convergent points
% [img,~]=MeshMapK(F,v,10,'false');
% b=img(:,4);
% t=0:resolution:1;
% s=(ones(1,size(t,2))-t);
% line1=t.*img(:,1)+s.*b;
% line2=t.*img(:,2)+s.*b;
% line3=t.*img(:,3)+s.*b;
% line4=t.*img(:,5)+s.*b;
% line5=t.*img(:,6)+s.*b;
% line7=t.*img(:,7)+s.*b;
% lines=[line1,line2,line3,line4,line5,line7];


[img,colors]=SingleMeshMapK(F,v,8);

%get the first frame for each movie
figure1=figure('OuterPosition',[1 scrsz(4)/2 scrsz(3)/2 scrsz(4)/2]);
axes1 = axes('Parent',figure1);
scatter3(axes1,x,y,z,predot,colors,'filled');
view(axes1,[135 25]);
grid(axes1,'on');
Mov(1)=getframe(figure1);

figure2=figure('OuterPosition',[scrsz(3)/2 scrsz(4)/2 scrsz(3)/2 scrsz(4)/2]);
axes2 = axes('Parent',figure2);
scatter3(axes2,img(1,:),img(2,:),img(3,:),predot,1:size(img,2),'filled');
%hold('on');
%scatter3(axes2,lines(1,:),lines(2,:),lines(3,:),predot,'black','filled');
view(axes2,[135 25]);
grid(axes2,'on');
%hold('off');
Imgs(1)=getframe(figure2);

%write partial images to .mat file
framefile=matfile('frames.mat','Writable', true);
save framefile Mov Imgs;

G=F;

for j=2:loops
    waitbar(j/loops,W,sprintf(loopMsg,j,loops));
    v=img;
    x=v(1,:);
    y=v(2,:);
    z=v(3,:);
    [img,colors]=SingleMeshMapK(F,v,8);
    
    %capture a frame for the preimage movie
    scatter3(axes1,x,y,z,predot,colors,'filled');
    view(axes1,[135 25]);
    grid(axes1,'on');
    Mov(j)=getframe(figure1);
    
    %capture a frame for the image movie
    scatter3(axes2,img(1,:),img(2,:),img(3,:),predot+j/10,1:size(img,2),'filled');
 %   hold('on');
 %  scatter3(axes2,lines(1,:),lines(2,:),lines(3,:),predot,'black','filled');
    view(axes2,[135 25]);
    grid(axes2,'on');
  %  hold('off');
    Imgs(j)=getframe(figure2);
    
    %write partial video to file?
  %  save framefile Mov Imgs;
    
end
delete(W);

[img,colors]=MeshMapK(F,v,10,'false');

scatter3(axes1,x,y,z,predot,colors,'filled');
view(axes1,[135 25]);
grid(axes1,'on');
Mov(loops+1)=getframe(figure1);
Mov(loops+2)=getframe(figure1);
Mov(loops+3)=getframe(figure1);
Mov(loops+4)=getframe(figure1);

scatter3(axes2,img(1,:),img(2,:),img(3,:),15,1:size(img,2),'filled');
%hold('on');
scatter3(axes2,lines(1,:),lines(2,:),lines(3,:),predot,'black','filled');
view(axes2,[135 25]);
grid(axes2,'on');
%hold('off');
Imgs(loops+1)=getframe(figure2);
Imgs(loops+2)=getframe(figure2);
Imgs(loops+3)=getframe(figure2);
Imgs(loops+4)=getframe(figure2);

%% Write the movies to file
open(PreVW);
writeVideo(PreVW,Mov);
close(PreVW);

open(ImVW);
writeVideo(ImVW,Imgs);
close(ImVW);
%close('all');