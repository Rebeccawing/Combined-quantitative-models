Data = csvread('RecordData_v5.csv',1,1); % ignore the datatime string

Data = Data(:,:);
%plot([Data(:,3),Data(:,4)])

[m,n] = size(Data);
[y1,PS] = mapminmax(Data',-1,1); % remapping data to plot them in a sigial figure
y2 = y1';
y2(:,2) = 0.9*y2(:,2); % market position


figure(1)
hold on
plot(y2(:,1)) % last price
plot(y2(:,3)) % nc_indicator
plot(y2(:,2),'color',[1 1 1]*0.7) % market position
legend('last price', 'nc indicator', 'market position')
hold off

figure(2)
hold on 
plot(Data(:,5)) % shp_indicator
plot(Data(:,6)) % ShpVThr
legend('shp indicator','ShpVThr')
hold off


sum(Data(:,5)>Data(:,6))/length(Data(:,5))
sum(Data(:,3)<Data(:,4))/length(Data(:,3))
