function alinea_d()

img_resolution = [32 32];

imdsT = imageDatastore('images_draws','IncludeSubfolders',1,'LabelSource','foldernames');

nfich = length(imdsT.Files) % Number of files found

for i=1:nfich
    
    img = readimage(imdsT,i);
    img = rgb2gray(img);
    imshow(img)
    img = imresize(img,img_resolution);
    binarizedImg = imbinarize(img);
    
    input(:,i) = reshape(binarizedImg, 1, []);
end

size(input)

categoricalTargets = imdsT.Labels;
labels = onehotencode(categoricalTargets,2);

target = labels;
target = target';
% disp(target)
size(target)

% load
load("melhor_net_alinea_c.mat");
net = melhorNet;


out = sim(net, input);

plotconfusion(target, out)

r = 0;
for i=1:size(out,2)               % Para cada classificacao
    [a b] = max(out(:,i));        % b guarda a linha onde encontrou valor mais alto da saida obtida
    [c d] = max(target(:,i));     % d guarda a linha onde encontrou valor mais alto da saida desejada
    if b == d                     % se estao na mesma linha, a classificacao foi correta (incrementa 1)
        r = r+1;
    end
end

accuracy = r/size(out,2)*100;
fprintf('Precisao total de treino %f\n', accuracy)

end