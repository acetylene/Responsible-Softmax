%% Set up the linear map L and the parameters F
step=1/100;
L=step/2*(ones(3)-eye(3))+(1-step)*eye(3);
seed=131071;
rng(seed);
F=randi(15,3,10);

%% Set up the mesh we will be using

% V will be the 3 dimensional simplex
resolution=.01;
v=zeros(size(F));
k=1;
for i=0:resolution:1
for j=0:resolution:1-i
v(:,k)=[i;j;abs(1-i-j)];
k=k+1;
end
end

%extract the x, y, and z coordinates for plotting
x=v(1,:);
y=v(2,:);
z=v(3,:);


%% Create output files
preimfilepart='preimagemovie_partial';
imfilepart='imagemovie_partial';
format='MPEG-4';

PreVW=VideoWriter(preimfilepart,format);
ImVW=VideoWriter(imfilepart,format);

PreVW.FrameRate=6;
ImVW.FrameRate=6;

%% Create the frames
loops=20;%5*(1/step);this would be 500, but that takes 145 hours! (as is)
Mov(loops+1)=struct('cdata',[],'colormap',[]);
Imgs(loops+1)=struct('cdata',[],'colormap',[]);
predot=5;

scrsz = get(groot,'ScreenSize');

loopMsg='Creating frame %d of %d';
W=waitbar(1/loops,sprintf(loopMsg,1,loops));

[img,colors]=MeshMapK(F,v,8);

%get the first frame for each movie
figure1=figure('OuterPosition',[1 scrsz(4)/2 scrsz(3)/2 scrsz(4)/2]);
axes1 = axes('Parent',figure1);
scatter3(axes1,x,y,z,predot,colors,'filled');
view(axes1,[135 25]);
grid(axes1,'on');
Mov(1)=getframe(figure1);

figure2=figure('OuterPosition',[scrsz(3)/2 scrsz(4)/2 scrsz(3)/2 scrsz(4)/2]);
axes2 = axes('Parent',figure2);
scatter3(axes2,img(1,:),img(2,:),img(3,:),100,1:size(img,2),'filled');
view(axes2,[135 25]);
grid(axes2,'on');
Imgs(1)=getframe(figure2);

%write partial images to .mat file
framefile=matfile('frames.mat','Writable', true);
save framefile Mov Imgs;

G=F;

for j=2:loops
    waitbar(j/loops,W,sprintf(loopMsg,j,loops));
    G=L*G;
    [img,colors]=MeshMapK(G,v,8);
    
    %capture a frame for the preimage movie
    scatter3(axes1,x,y,z,predot,colors,'filled');
    view(axes1,[135 25]);
    grid(axes1,'on');
    Mov(j)=getframe(figure1);
    
    %capture a frame for the image movie
    scatter3(axes2,img(1,:),img(2,:),img(3,:),100,1:size(img,2),'filled');
    view(axes2,[135 25]);
    grid(axes2,'on');
    Imgs(j)=getframe(figure2);
    
    %write partial video to file?
    save framefile Mov Imgs;
    
end
delete(W);

[img,colors]=MeshMapK((L^50)*G,v,8);

scatter3(axes1,x,y,z,predot,colors,'filled');
view(axes1,[135 25]);
grid(axes1,'on');
Mov(loops+1)=getframe(figure1);

scatter3(axes2,img(1,:),img(2,:),img(3,:),100,1:size(img,2),'filled');
view(axes2,[135 25]);
grid(axes2,'on');
Imgs(loops+1)=getframe(figure2);


%% Write the movies to file


open(PreVW);
writeVideo(PreVW,Mov);
close(PreVW);

open(ImVW);
writeVideo(ImVW,Imgs);
close('all');