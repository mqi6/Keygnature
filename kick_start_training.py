import copy
import random
import torch
import torch.nn as nn
from torch.nn import functional as tfunc
import numpy as np
from torch.utils.data import DataLoader, Dataset
from torch.autograd import Variable
from torch.nn.utils import clip_grad_norm_
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.metrics import roc_curve
from pytorch_metric_learning import miners, losses, samplers
import time
import timm
from tqdm import tqdm
import matplotlib.pyplot as plt
from utils.configs import configs

scenario = 'chunk'

batches_per_epoch_train = 256
batches_per_epoch_val = 16
batch_size_train = 512
batch_size_val = 512
sequence_length = 128
embedding_size = 512


EPOCHS = 5


def contrastive_loss(x1, x2, label, margin: float = 1.0):
    dist = nn.functional.pairwise_distance(x1, x2)
    loss = (1 - label) * torch.pow(dist, 2) + label * torch.pow(torch.clamp(margin - dist, min=0.0), 2)
    loss = torch.mean(loss)
    return loss


def norm_embeddings(embeddings):
    return embeddings / torch.sqrt((embeddings ** 2).sum(dim=-1, keepdims=True))


class Kick_Start_Model(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, sequence_length = 100):
        super().__init__()
        self.batch_norm = nn.BatchNorm1d(num_features=sequence_length)
        self.dropout = nn.Dropout(p=0.2)
        self.rnn = torch.nn.RNN(input_size, hidden_size, nonlinearity='relu', batch_first=True)
        self.linear = torch.nn.Linear(hidden_size * sequence_length, output_size)
        self.softmax = torch.nn.Softmax(dim=1)

    def forward(self, x):
        x = self.batch_norm(x)
        x, _ = self.rnn(x)
        # x = self.linear(x)
        # embedding = self.softmax(x)[:, -1, :]
        embedding = torch.flatten(x, start_dim=1)
        embedding = self.dropout(embedding)
        embedding = self.linear(embedding)
        # print(embedding.shape)
        # embedding = embedding / torch.sqrt((embedding ** 2).sum(dim=-1, keepdims=True))
        return embedding


class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, sequence_length = 100):
        super().__init__()
        self.network = nn.Sequential(
            nn.BatchNorm1d(num_features=sequence_length),
            nn.Flatten(),
            nn.Linear(input_size * sequence_length, hidden_size),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(hidden_size, output_size)
        )
    
    def forward(self, x):
        embedding = self.network(x)
        embedding = embedding / torch.sqrt((embedding ** 2).sum(dim=-1, keepdims=True))
        return embedding


def conv_bn_relu(inp, outp):
    return nn.Sequential(
        nn.Conv1d(inp, outp, kernel_size=3, padding=1),
        nn.BatchNorm1d(outp),
        nn.PReLU(outp),
    )


def repeat_nn(num_repeat, fn, *args, **kwargs):
    return nn.Sequential(*[fn(*args, **kwargs)])


class Sum(nn.Module):
    def __init__(self, *modules):
        super().__init__()
        self.forward_modules = nn.Sequential(*modules)
    
    def forward(self, x):
        return sum([m(x) for m in self.forward_modules])


class ConvModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, sequence_length=100, num_convs=(3, 18, 18, 18)):
        super().__init__()
        self.norm = nn.BatchNorm1d(num_features=sequence_length)
        self.network = nn.Sequential(
            nn.Conv1d(input_size, hidden_size, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_size),
            nn.PReLU(hidden_size),
            Sum(
                nn.Sequential(
                    repeat_nn(num_convs[0], conv_bn_relu, hidden_size, hidden_size),
                    nn.AvgPool1d(2),
                ),
                nn.MaxPool1d(2)
            ),
            nn.BatchNorm1d(hidden_size),
            nn.PReLU(hidden_size),
            Sum(
                nn.Sequential(
                    repeat_nn(num_convs[1], conv_bn_relu, hidden_size, hidden_size),
                    nn.AvgPool1d(2),
                ),
                nn.MaxPool1d(2)
            ),
            nn.BatchNorm1d(hidden_size),
            nn.PReLU(hidden_size),
            nn.Conv1d(hidden_size, hidden_size * 2, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_size * 2),
            nn.PReLU(hidden_size * 2),
            Sum(
                nn.Sequential(
                    repeat_nn(num_convs[2], conv_bn_relu, hidden_size * 2, hidden_size * 2),
                    nn.AvgPool1d(2),
                ),
                nn.MaxPool1d(2)
            ),
            nn.BatchNorm1d(hidden_size * 2),
            nn.PReLU(hidden_size * 2),
            Sum(
                repeat_nn(num_convs[3], conv_bn_relu, hidden_size * 2, hidden_size * 2),
                nn.Identity()
            ),
            nn.BatchNorm1d(hidden_size * 2),
            nn.PReLU(hidden_size * 2),
            nn.Flatten(),
            # nn.Linear(hidden_size * sequence_length, hidden_size),
            nn.Dropout(p=0.2),
            nn.Linear(hidden_size * sequence_length * 2 // 8, output_size)
        )
    
    def forward(self, x):
        x = self.norm(x)
        x = x.transpose(2, 1)
        embedding = self.network(x)
        embedding = embedding / (torch.sqrt((embedding ** 2).sum(dim=-1, keepdims=True)))
        return embedding


class CompareModule(nn.Module):
    def __init__(self, feature_extractor, embedding_size):
        super().__init__()
        self.feature_extractor = feature_extractor
        self.embedding_size = embedding_size
        self.network = nn.Sequential(
            nn.BatchNorm1d(2 * embedding_size),
            nn.Linear(2 * embedding_size, embedding_size),
            nn.PReLU(embedding_size),
            nn.BatchNorm1d(embedding_size),
            nn.Linear(embedding_size, 64),
            nn.PReLU(64),
            Sum(
                nn.Identity(),
                nn.Sequential(*[nn.Sequential(
                    nn.BatchNorm1d(64),
                    nn.Linear(64, 64),
                    nn.PReLU(64)
                ) for _ in range(10)])
            ),
            nn.Linear(64, 1)
        )
    
    def forward(self, x_left, x_right):
        features_left = self.feature_extractor(x_left)
        features_right = self.feature_extractor(x_right)
        return self.network(torch.cat((features_left, features_right), dim=1))


def extract_features(session):
    bg = session[0, 0]
    diff = np.cos((session[:, 1] - session[:, 0]) / 1E1)
    ascii_f = session[:, 2] / 256
    cos_bg = np.cos((session[:, 0] - bg) / 1e3)
    sin_bg = np.sin((session[:, 0] - bg) / 1e3)
    cos_ed = np.cos((session[:, 1] - bg) / 1e3)
    sin_ed = np.sin((session[:, 1] - bg) / 1e3)
    return np.stack((diff, cos_bg, sin_bg, cos_ed, sin_ed, ascii_f), axis=1)


def convert_features_to_2d(features):
    assert len(features.shape) == 2
    session_length, num_features = features.shape
    side_size = int(np.sqrt(session_length))
    if side_size ** 2 != session_length:
        size_size += 1
        features = 0
        raise NotImplementedError
    converted_features = features.reshape((side_size, side_size, num_features)).transpose((2, 0, 1))
    return converted_features


def get_session_fixed_length(session, sequence_length, start_zero=True):
    num_tile = max(1, sequence_length // session.shape[0]) + 2
    tiled = np.tile(session, (num_tile, 1))
    if start_zero:
        start_idx = 0
    else:
        start_idx = np.random.randint(0, session.shape[0] - 1)
    # print(tiled.shape, start_idx, sequence_length, tiled[start_idx:start_idx+sequence_length].shape)
    return tiled[start_idx:start_idx+sequence_length]


def augment_session(session):
    new_session = []
    for cur_event in session.copy():
        if random.random() < 0.02:
            # drop the event
            continue
        if random.random() < 0.1:
            # copy the event
            new_session.append(cur_event.copy())
        if random.random() < 0.1:
            # change the start time
            cur_event[0] += (random.random() - 0.5) * 100
        if random.random() < 0.1:
            # change the end time
            cur_event[1] += (random.random() - 0.5) * 100
        if random.random() < 0.1:
            # change the symbol
            cur_event[2] = random.randint(10, 255)
        # if random.random() < 0.1:
        #     # zero the event
        #     cur_event = np.zeros_like(cur_event)
        new_session.append(cur_event)
    if len(new_session) < 3:
        return session
    return np.array(new_session)


class PrepareData:
    def __init__(self, dataset, sequence_length, samples_considered_per_epoch, augment=False):
        self.data = dataset
        self.len = samples_considered_per_epoch
        self.sequence_length = sequence_length
        self.user_keys = list(self.data.keys())
        self.augment = augment

    def __getitem__(self, index):
        user_num = random.randint(0, len(self.user_keys)-1)
        user_idx = self.user_keys[user_num]
        session_idx = random.choice(list(self.data[user_idx].keys()))
        session_1 = self.data[user_idx][session_idx]
        if self.augment:
            session_1 = augment_session(session_1)
        session_1 = get_session_fixed_length(session_1, self.sequence_length, not self.augment)
        # diff_1 = np.reshape((session_1[:, 1] - session_1[:, 0]) / 1E3, (np.shape(session_1)[0], 1))
        # ascii_1 = np.reshape(session_1[:, 2] / 256, (np.shape(session_1)[0], 1))
        session_1_processed = extract_features(session_1)

        # label = random.choice([0, 1]
        user_num_2 = user_num
        if random.random() < 0.5:
            label = 0
            session_idx_2 = random.choice([x for x in list(self.data[user_idx].keys()) if x != session_idx])
            session_2 = self.data[user_idx][session_idx_2]
        else:
            label = 1
            while user_num_2 == user_num:
                user_num_2 = random.randint(0, len(self.user_keys)-1)
            user_idx_2 = self.user_keys[user_num_2]
            session_idx_2 = random.choice(list(self.data[user_idx_2].keys()))
            session_2 = self.data[user_idx_2][session_idx_2]
        if self.augment:
            session_2 = augment_session(session_2)
        session_2 = get_session_fixed_length(session_2, self.sequence_length, not self.augment)
        # diff_2 = np.reshape((session_2[:, 1] - session_2[:, 0]) / 1E3, (np.shape(session_2)[0], 1))
        # ascii_2 = np.reshape(session_2[:, 2] / 256, (np.shape(session_2)[0], 1))
        session_2_processed = extract_features(session_2)
        # print('1', session_1_processed.shape, session_2_processed.shape, label, user_num, user_num_2)
        # session_1_processed = convert_features_to_2d(session_1_processed)
        # session_2_processed = convert_features_to_2d(session_2_processed)
        return (session_1_processed, session_2_processed), label, (user_num, user_num_2)

    def __len__(self):
        return self.len


class ClassificationData(Dataset):
    def __init__(self, dataset, sequence_length, samples_per_epoch):
        self.data = dataset
        self.user_keys = list(self.data.keys())
        self.sequence_length = sequence_length
        self.samples_per_epoch = samples_per_epoch

    def __len__(self):
        return len(self.data) * self.samples_per_epoch
    
    def __getitem__(self, idx):
        idx = idx % len()
        user_idx = self.user_keys[idx]
        session_idx = random.choice(list(self.data[user_idx].keys()))
        session_1 = self.data[user_idx][session_idx]
        session_1 = np.concatenate((session_1, np.zeros((self.sequence_length, np.shape(session_1)[1]))))[:self.sequence_length]
        # diff_1 = np.reshape((session_1[:, 1] - session_1[:, 0]) / 1E3, (np.shape(session_1)[0], 1))
        # ascii_1 = np.reshape(session_1[:, 2] / 256, (np.shape(session_1)[0], 1))
        session_1_processed = extract_features(session_1)
        return session_1_processed, idx


class TimmModule(nn.Module):
    def __init__(self, backbone, dropout, embedding_size):
        super().__init__()
        self.backbone = backbone
        self.pooling2d = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(p=dropout)
        self.linear = nn.Linear(self.backbone.num_features, embedding_size, bias=False)
    
    def forward(self, x):
        features = self.backbone.forward_features(x)
        if len(features.shape) == 3:
            features = features.mean(1)
        elif len(features.shape) == 2:
            features = features
        else:
            features = self.pooling2d(features)
        if len(features.shape) > 1:
            features = torch.flatten(features, 1)
        features = self.dropout(features)
        embedding = self.linear(features)
        embedding = embedding / (torch.sqrt((embedding ** 2).sum(dim=-1, keepdims=True)))
        return embedding


file_loc = 'KVC_data/{}/{}_dev_set.npy'.format(scenario, scenario)
data = np.load(file_loc, allow_pickle=True).item()
to_delete = []
for user_key, sessions in data.items():
    for sess_key, sess in sessions.items():
        if len(sess) < 5:
            to_delete.append((user_key, sess_key))

for user_key, sess_key in to_delete:
    del data[user_key][sess_key]

to_delete = []
for user_key, sessions in data.items():
    if len(sessions) < 2:
        to_delete.append(user_key)

for user_key in to_delete:
    del data[user_key]


dev_set_users = list(data.keys())
random.shuffle(dev_set_users)
train_val_division = 0.95
train_users = dev_set_users[:int(len(dev_set_users)*train_val_division)]
val_users = dev_set_users[int(len(dev_set_users)*train_val_division):]
print('Train num users', len(train_users))
print('Validation num users', len(val_users))


# train_data = copy.deepcopy(data)
# for user in list(data.keys()):
#     if user not in train_users:
#         del train_data[user]
# val_data = copy.deepcopy(data)
# for user in list(data.keys()):
#     if user not in val_users:
#         del val_data[user]
# del data
train_data = {u: data[u] for u in train_users}
val_data = {u: data[u] for u in val_users}



ds_t = PrepareData(train_data, sequence_length=sequence_length, samples_considered_per_epoch=batches_per_epoch_train*batch_size_train, augment=True)
ds_v = PrepareData(val_data, sequence_length=sequence_length, samples_considered_per_epoch=batches_per_epoch_val*batch_size_val)

train_dataloader = DataLoader(ds_t, batch_size=batch_size_train, num_workers=0, persistent_workers=False)
val_dataloader = DataLoader(ds_v, batch_size=batch_size_val, num_workers=0, persistent_workers=False)

device = torch.device('cuda:0')  #"cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

# extractor = ConvModel(input_size=6, hidden_size=4, output_size=embedding_size, sequence_length=sequence_length, num_convs=(1, 3, 13, 3))
# extractor = ConvModel(input_size=6, hidden_size=64, output_size=embedding_size, sequence_length=sequence_length, num_convs=(1, 3, 13, 3))
# extractor = ConvModel(input_size=6, hidden_size=64, output_size=embedding_size, sequence_length=sequence_length, num_convs=(10, 30, 50, 50))
extractor = ConvModel(input_size=6, hidden_size=128, output_size=embedding_size, sequence_length=sequence_length, num_convs=(30, 50, 100, 100))
# model = CompareModule(extractor, embedding_size)
model = extractor
# timm_model = 'xcit_nano_12_p8_224'
# model = TimmModule(timm.create_model(timm_model, pretrained=False, in_chans=6, img_size=16), dropout=0.2, embedding_size=embedding_size)
model.to(device)
# loss_fn = nn.BCEWithLogitsLoss()
additional_loss = losses.ArcFaceLoss(num_classes=len(train_data), embedding_size=embedding_size).to(device)
# miner = miners.BatchEasyHardMiner(miners.BatchEasyHardMiner.HARD, miners.BatchEasyHardMiner.HARD)
# additional_loss = losses.MultipleLosses([
#     losses.TupletMarginLoss(),
#     losses.IntraPairVarianceLoss()
# ], weights=[1.0, 0.5])
lr = 1e-2
optimizer = torch.optim.AdamW(params=[
    {'params': model.parameters(), 'lr': lr}, 
    {'params': additional_loss.parameters(), 'lr': lr * (10 ** 0)},
    ]
    , lr=lr,
    # momentum=0.8, 
    betas=(0.95, 0.999)
    )
scheduler = CosineAnnealingLR(optimizer=optimizer, T_max=EPOCHS,
                              eta_min=1e-8, last_epoch=-1)
grad_amp = torch.cuda.amp.grad_scaler.GradScaler(growth_interval=1)
MODEL_NAME = f'conv_bigger_f2_{embedding_size}_{EPOCHS}_{sequence_length}_{scenario}_20_10'


def train_one_epoch():
    running_loss = 0.
    batch_eers = []
    batch_losses = []
    for i, (input_data, labels, user_indices) in enumerate(tqdm(train_dataloader, total=len(train_dataloader)), 0):
        input_data[0] = Variable(input_data[0]).float()
        input_data[1] = Variable(input_data[1]).float()
        user_indices = [u.to(device) for u in user_indices]
        with torch.cuda.amp.autocast(True):
            input_data[0] = input_data[0].to(device)
            input_data[1] = input_data[1].to(device)
            pred1 = model(input_data[0])
            pred2 = model(input_data[1])
            loss = 0  # contrastive_loss(pred1, pred2, labels.float().to(device)) * 30 # \
                # + (loss_func(pred1, user_indices[0]) + loss_func(pred2, user_indices[1]))
            whole_pred = torch.cat((pred1, pred2), dim=0)
            whole_indices = torch.cat(user_indices, dim=0)
            # indices_tuple = miner(whole_pred, whole_indices)
            # loss += additional_loss(whole_pred, whole_indices, indices_tuple)
            loss += additional_loss(whole_pred, whole_indices)
            # preds = model(*input_data).squeeze()
            # loss = loss_fn(preds, labels.float().to(device))
        optimizer.zero_grad()
        # loss.backward(retain_graph=True)
        # optimizer.step()
        grad_amp.scale(loss).backward()
        grad_amp.unscale_(optimizer)
        clip_grad_norm_(model.parameters(), max_norm=1, norm_type=2)
        grad_amp.step(optimizer)
        grad_amp.update()
        scheduler.step()
        running_loss = loss.item()
        with torch.inference_mode():
            pred1 = pred1.detach()
            pred2 = pred2.detach()
            dists = nn.functional.pairwise_distance(norm_embeddings(pred1), norm_embeddings(pred2))
            # dists = preds.detach()
        fpr, tpr, threshold = roc_curve(labels, dists.detach().cpu().numpy(), pos_label=1)
        # print(f'Batch {i} |> batch size {len(labels)} | num positive {labels.detach().cpu().numpy().sum()} | min dist {dists.detach().cpu().numpy().min()}')
        fnr = 1 - tpr
        EER1 = fpr[np.nanargmin(np.absolute((fnr - fpr)))]
        EER2 = fnr[np.nanargmin(np.absolute((fnr - fpr)))]
        batch_eers.append(np.mean([EER1, EER2])*100)
        batch_losses.append(running_loss)
    return np.mean(batch_losses), np.mean(batch_eers)

loss_t_list, eer_t_list = [], []
loss_v_list, eer_v_list = [], []

best_v_eer = 100.
for epoch in range(EPOCHS):
    start = time.time()
    print('EPOCH:', epoch)
    model.train()
    epoch_loss, epoch_eer = train_one_epoch()
    loss_t_list.append(epoch_loss)
    eer_t_list.append(epoch_eer)
    model.eval()
    running_loss_v = 0.
    with torch.inference_mode():
        batch_eers_v = []
        batch_losses_v = []
        for i, (input_data, labels, user_indices) in enumerate(tqdm(val_dataloader, total=len(val_dataloader)), 0):
            input_data[0] = Variable(input_data[0]).float()
            input_data[1] = Variable(input_data[1]).float()
            input_data[0] = input_data[0].to(device)
            input_data[1] = input_data[1].to(device)
            pred1 = model(input_data[0])
            pred2 = model(input_data[1])            # criterion = torch.jit.script(nn.BCELoss())
            loss_v = contrastive_loss(pred1, pred2, labels.float().to(device))
            # preds = model(*input_data).squeeze()
            # loss_v = loss_fn(preds, labels.float().to(device))
            running_loss_v = loss_v.item()
            pred1 = pred1.cpu().detach()
            pred2 = pred2.cpu().detach()
            dists = nn.functional.pairwise_distance(norm_embeddings(pred1), norm_embeddings(pred2))
            # dists = preds.detach().cpu().numpy()
            fpr, tpr, threshold = roc_curve(labels, dists, pos_label=1)
            fnr = 1 - tpr
            EER1_v = fpr[np.nanargmin(np.absolute((fnr - fpr)))]
            EER2_v = fnr[np.nanargmin(np.absolute((fnr - fpr)))]
            batch_eers_v.append(np.mean([EER1_v, EER2_v])*100)
            batch_losses_v.append(running_loss_v)
    eer_v_list.append(np.mean(batch_eers_v))
    loss_v_list.append(np.mean(batch_losses_v))
    print("Epoch Loss on Training Set: " + str(epoch_loss))
    print("Epoch Loss on Validation Set: " + str(np.mean(batch_losses_v)))
    print("Epoch EER [%] on Training Set: "+ str(epoch_eer))
    print("Epoch EER [%] on Validation Set: " + str(np.mean(batch_eers_v)))
    if eer_v_list[-1] < best_v_eer:
        torch.save(model.state_dict(), MODEL_NAME + '.pt')
        print("New Best Epoch EER [%] on Validation Set: " + str(eer_v_list[-1]))
        best_v_eer = eer_v_list[-1]
    TRAIN_INFO_LOG_FILENAME = MODEL_NAME + '_log.txt'
    log_list = [loss_t_list, loss_v_list, eer_t_list, eer_v_list]
    with open(TRAIN_INFO_LOG_FILENAME, "w") as output:
        output.write(str(log_list))
    end = time.time()
    print('Time for last epoch [min]: ' + str(np.round((end-start)/60, 2)))

with open(TRAIN_INFO_LOG_FILENAME, "r") as file:
    log_list = eval(file.readline())
figure, axis = plt.subplots(3)
figure.suptitle('Scenario: {}'.format(scenario))
axis[0].plot(log_list[0])
axis[0].set_title("Training Loss")
axis[0].set_ylabel('Loss')
axis[0].grid()
axis[1].plot(log_list[1])
axis[1].set_title("Validation Loss")
axis[1].set_ylabel('Loss')
axis[1].grid()
axis[2].plot(log_list[2], label='Training')
axis[2].plot(log_list[3], label='Validation')
axis[2].set_title("Training and Validation EER (%)")
axis[2].set_xlabel('Epochs')
axis[2].set_ylabel('EER (%)')
axis[2].legend()
axis[2].grid()
plt.savefig(f'train_plots_{MODEL_NAME}.pdf')
# plt.show()
