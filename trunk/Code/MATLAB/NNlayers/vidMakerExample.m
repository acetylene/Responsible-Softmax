%% Greedy assignment
%% Two chinese characters, one broken into
%% two blobs
fname1=fullfile('BWChars','char00035.pbm');
I1=imread(fname1);


fname2=fullfile('BWChars','char01915.pbm');
I2=imread(fname2);

vidMaker = VideoMaker('VideoTitle','fixing_chin_chars',...
                      'Pause',2,...
                      'FrameRate',1);


make_title_frame({'Comparing Two Chinese Characters'
                  'In the Presence of Damage'});
vidMaker.capture_frame(4);

subplot(1,2,1);
imagesc(I1);
title('Unbroken char');
subplot(1,2,2);
imagesc(I2);
title('Broken char');
vidMaker.capture_frame(3);


opts = {'ForegroundColor','black'};

s1=OutlineCycles(I1,opts{:});                   % 1 cycle
s2=OutlineCycles(I2,opts{:});                   % 2 cycles

ob1 = s1.Objects(1);
ob2 = s2.Objects(1);
ob3 = s2.Objects(2);

clf;
subplot(1,2,1);
draw_objects(ob1,'PlotFunction','quiver');
title('Unbroken outer cycle');
subplot(1,2,2);
hold on;
draw_objects(ob2,'PlotFunction','quiver');
draw_objects(ob3,'PlotFunction','quiver');
title('Broken outer cycle');
hold off;
vidMaker.capture_frame(3);


% Normalization based on outer cycles
M1 = mean(ob1.Z);
M2 = mean([ob2.Z;ob3.Z]);

S1 = std(ob1.Z);
S2 = std([ob2.Z;ob3.Z]);

ob1.Z = (ob1.Z-M1)./S1;
ob2.Z = (ob2.Z-M2)./S2;
ob3.Z = (ob3.Z-M2)./S2;

Z1=cellfun(@(Cycle)ob1.Z(Cycle),ob1.Cycles,...
           'UniformOutput',false);
Z2=cellfun(@(Cycle)ob2.Z(Cycle),ob2.Cycles,...
           'UniformOutput',false);
Z3=cellfun(@(Cycle)ob3.Z(Cycle),ob3.Cycles,...
           'UniformOutput',false);


num_ctrl_pts = 64;

n1 = round( num_ctrl_pts * numel(Z1{1}) ./ ob1.Length );
n21 = round(num_ctrl_pts * numel(Z2{1}) ./ ob2.Length);
n22 = round(num_ctrl_pts * numel(Z3{1}) ./ ob3.Length);



W11 = approximate_cycle(Z1{1}, n1);
W21 = approximate_cycle(Z2{1}, n21);
W22 = approximate_cycle(Z3{1}, n22);


[D,W] = dist_from_cycle(W11,{W21,W22});

clf;
hold on;
plot(real(W11),imag(W11),'-o','LineWidth',1);
plot(real(W21),imag(W21),'-o','LineWidth',1);
plot(real(W22),imag(W22),'-o','LineWidth',1);
plot(real(W),imag(W),'-o');
quiver(real(W11),imag(W11),real(W)-real(W11),imag(W)-imag(W11),0,...
        'LineWidth',3,'Color','red');
pbaspect([1,1,1]);
title('Mapping the unbroken outline to two broken ones');
hold off;
vidMaker.capture_frame(5);

cost=@(p)norm(D,p)./numel(D).^(1/p);
clf;
p=linspace(1,100,90);
q=arrayfun(cost,p);
plot(p,q);
title('Mean l^p norm of distance');
vidMaker.capture_frame(5);


[D,W] = dist_from_cycle(W21,W11);

clf;
hold on;
plot(real(W11),imag(W11),'-o','LineWidth',1);
plot(real(W21),imag(W21),'-o','LineWidth',1);
plot(real(W22),imag(W22),'-o','LineWidth',1);
plot(real(W),imag(W),'-o');
quiver(real(W21),imag(W21),real(W)-real(W21),imag(W)-imag(W21),0,...
        'LineWidth',3,'Color','red');
pbaspect([1,1,1]);
title('Mapping first broken outline to the unbroken one');
hold off;
vidMaker.capture_frame(5);

cost=@(p)norm(D,p)./numel(D).^(1/p);
clf;
p=linspace(1,100,90);
q=arrayfun(cost,p);
plot(p,q);
title('Mean l^p norm of distance');
vidMaker.capture_frame(5);


[D,W] = dist_from_cycle(W22,W11);

clf;
hold on;
plot(real(W11),imag(W11),'-o','LineWidth',1);
plot(real(W21),imag(W21),'-o','LineWidth',1);
plot(real(W22),imag(W22),'-o','LineWidth',1);
plot(real(W),imag(W),'-o');
quiver(real(W22),imag(W22),real(W)-real(W22),imag(W)-imag(W22),0,...
        'LineWidth',3,'Color','red');
pbaspect([1,1,1]);
title('Mapping second broken outline to the unbroken one');
hold off;
vidMaker.capture_frame(5);


cost=@(p)norm(D,p)./numel(D).^(1/p);
clf;
p=linspace(1,100,90);
q=arrayfun(cost,p);
plot(p,q);
title('Mean l^p norm of distance');
vidMaker.capture_frame(5);


close(vidMaker);