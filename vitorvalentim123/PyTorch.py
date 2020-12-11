import torch
from torch import nn, optim

from torch.utils.data import Dataset, DataLoader
from torch.utils.data import SubsetRandomSampler
from torchvision import transforms

if torch.cuda.is_available():
  device = torch.device('cuda')
else:
  device = torch.device('cpu')

class ThumbData(Dataset):
  def __init__(self, all_url, label_list, nbins):
    self.all_url = all_url
    self.nbins   = nbins
    self.labels  = label_list

  def __len__(self):
    return len(self.all_url)

  def __getitem__(self, idx):

    thumbnail_url = self.all_url[idx]
    img = skimage.img_as_float(io.imread(thumbnail_url))

    histograms = [exposure.histogram(img[:,:,i], nbins=self.nbins, normalize=True)[0] for i in range(img.shape[-1])]

    return np.asarray(histograms).ravel(), self.labels[idx]

data = ThumbData(url_list, label_list, nbins)

train_size = int(0.75*len(data))
idx  = torch.randperm(len(data))
train_sampler = SubsetRandomSampler(idx[0:train_size]) 
test_sampler = SubsetRandomSampler(idx[train_size:])

train_loader = DataLoader(data, sampler=train_sampler,
                          batch_size=10, num_workers=4)

test_loader  = DataLoader(data, sampler=test_sampler,
                          batch_size=10, num_workers=4)

class MinhaRede(nn.Module):

  def __init__(self, tam_entrada):

    super(MinhaRede, self).__init__()

    self.rede = nn.Sequential(
        nn.Linear(tam_entrada, 32),
        nn.ReLU(),
        nn.Linear(32, 2)
    )

  def forward(self, thumbnail):

    saida = self.rede(thumbnail)
    return saida

tam_entrada = 3 * 25
rede = MinhaRede(tam_entrada).to(device).double()

print(rede)

optimizer = optim.Adam(rede.parameters(), lr=1e-3, weight_decay=5e-4)
criterion = nn.CrossEntropyLoss().to(device)

from sklearn import metrics 

def forward(mode, loader):

  if mode == 'train':
    rede.train()
  else:
    rede.eval()
  
  epoch_loss = []
  pred_list, label_list = [], []

  for data, label in loader:

    data = data.to(device)
    label = label.to(device)

    optimizer.zero_grad()
    out = rede(data)
  
    loss = criterion(out, label)
    epoch_loss.append(loss.cpu().data)
  
    pred = out.data.max(dim=1)[1].cpu().numpy()
  
    pred_list.append(pred)
    label_list.append(label.cpu().numpy())

    if mode == 'train':
      loss.backward()
      optimizer.step()

  epoch_loss = np.asarray(epoch_loss)
  pred_list = np.asarray(pred_list).ravel()
  label_list = np.asarray(label_list).ravel()

  acc = metrics.accuracy_score(pred_list, label_list)

  print(mode, 'Loss:', epoch_loss.mean(), '+/-', epoch_loss.std(), 'Accuracy:', acc)

num_epochs = 20
for i in range(num_epochs):
  forward('train', train_loader)
  forward('test', test_loader)
  print('--------------------------------')
