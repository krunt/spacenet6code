import sys
sys.path.append('utilities')
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.autograd import Variable

try:
    from itertools import ifilterfalse
except ImportError:  # py3k
    from itertools import filterfalse


import warnings
from pathlib import Path
import tempfile
import csv
import os
import datetime
import json
import shutil
import time
import sys

from tqdm import tqdm
from skimage.color import label2rgb
from skimage import measure
from multiprocessing import Pool
import lightgbm as lgb
from sklearn.model_selection import KFold, GroupShuffleSplit
from sklearn.neighbors import KDTree
from skimage.morphology import watershed
import warnings

from skimage import measure
from skimage.morphology import dilation, square, watershed
import scipy.sparse as ss
import numpy as np
import pandas as pd
import attr
import click
import tqdm
import cv2
import rasterio
import skimage.io
import skimage.measure
from sklearn.utils import Bunch

from torch import nn
import torch
from torch.optim import Adam
import torchvision.models
from torchvision.models import vgg16
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import SequentialSampler, RandomSampler
from albumentations.pytorch.transforms import img_to_tensor
from albumentations import (
    Normalize, Compose, HorizontalFlip, RandomRotate90, RandomCrop, CenterCrop,
    ElasticTransform, VerticalFlip, RandomBrightnessContrast, GaussNoise
    )

from multiprocessing import Pool
from functools import partial
from scipy.ndimage import binary_erosion

import spacenetutilities.labeltools.coreLabelTools as cLT
from spacenetutilities import geoTools as gT
from shapely.geometry import shape
from shapely.wkt import dumps
import geopandas as gpd

from scipy.ndimage.morphology import distance_transform_edt

import segmentation_models_pytorch as smp

from zoo.unet import DensenetUnet

SAR_MAX = 80.0
SAR_MEAN = np.array([25.301306, 28.817173, 26.368433]) / SAR_MAX
SAR_STD = np.array([12.33574 , 13.619368, 12.610607]) / SAR_MAX

#SAR_MEAN = np.array([25.301306, 28.817173, 26.368433, 23.118353]) / SAR_MAX
#SAR_STD = np.array((12.33574 , 13.619368, 12.610607, 11.541024)) / SAR_MAX

warnings.simplefilter(action='ignore', category=FutureWarning)


rot_df = None


def readrotationfile(path):
    rotationdf = pd.read_csv(path,
                             sep=' ',
                             index_col=0,
                             names=['strip', 'direction'],
                             header=None)
    rotationdf['direction'] = rotationdf['direction'].astype(int)
    return rotationdf


def lookuprotation(tilepath, rotationdf):
    tilename = os.path.splitext(os.path.basename(tilepath))[0]
    stripname = '_'.join(tilename.split('_')[-4:-2])
    rotation = rotationdf.loc[stripname].squeeze()
    return rotation


def transform_model_keys(state_dict):
    return state_dict
#    ret = dict()
#    for key in state_dict.keys():
#        if key.startswith('module.'):
#            ret[key[7:]] = state_dict[key]
#    return ret

def orient_sar(img, rotFlag, direction=0):
    if rotFlag == 1:
        return np.fliplr(np.flipud(img))
    if rotFlag == 0:
        return img
    assert(0)
    if rotFlag == 2:
        return np.rot90(img, axes=(0,1) if direction == 0 else (1,0))

class conv_relu(nn.Module):
    def __init__(self, in_, out):
        super().__init__()
        self.conv = nn.Conv2d(in_, out, 3, padding=1)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.activation(x)
        return x


class decoder_block(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels):
        super(decoder_block, self).__init__()
        self.in_channels = in_channels
        self.block = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear'),
            conv_relu(in_channels, middle_channels),
            conv_relu(middle_channels, out_channels),
        )

    def forward(self, x):
        return self.block(x)

#def unet_vgg16(pretrained=False):
    #model = smp.Unet('densenet161', encoder_weights='imagenet')
    #return model


class orig_unet_vgg16(nn.Module):
    def __init__(self, num_filters=32, num_classes=3, pretrained=False):
        super().__init__()

        self.encoder = vgg16(pretrained=pretrained).features
        self.pool = nn.MaxPool2d(2, 2)

        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Sequential(
            #nn.Conv2d(4, 3, 1),
            self.encoder[0], self.relu, self.encoder[2], self.relu)
        self.conv2 = nn.Sequential(
            self.encoder[5], self.relu, self.encoder[7], self.relu)
        self.conv3 = nn.Sequential(
            self.encoder[10], self.relu, self.encoder[12], self.relu,
            self.encoder[14], self.relu)
        self.conv4 = nn.Sequential(
            self.encoder[17], self.relu, self.encoder[19], self.relu,
            self.encoder[21], self.relu)
        self.conv5 = nn.Sequential(
            self.encoder[24], self.relu, self.encoder[26], self.relu,
            self.encoder[28], self.relu)

        self.center = decoder_block(512, num_filters * 8 * 2, num_filters * 8)
        self.dec5 = decoder_block(
            512 + num_filters * 8, num_filters * 8 * 2, num_filters * 8)
        self.dec4 = decoder_block(
            512 + num_filters * 8, num_filters * 8 * 2, num_filters * 8)
        self.dec3 = decoder_block(
            256 + num_filters * 8, num_filters * 4 * 2, num_filters * 2)
        self.dec2 = decoder_block(
            128 + num_filters * 2, num_filters * 2 * 2, num_filters)
        self.dec1 = conv_relu(64 + num_filters, num_filters)
        self.final = nn.Conv2d(num_filters, num_classes, kernel_size=1)

    def forward(self, x):
        conv1 = self.conv1(x)
        conv2 = self.conv2(self.pool(conv1))
        conv3 = self.conv3(self.pool(conv2))
        conv4 = self.conv4(self.pool(conv3))
        conv5 = self.conv5(self.pool(conv4))
        center = self.center(self.pool(conv5))
        dec5 = self.dec5(torch.cat([center, conv5], 1))
        dec4 = self.dec4(torch.cat([dec5, conv4], 1))
        dec3 = self.dec3(torch.cat([dec4, conv3], 1))
        dec2 = self.dec2(torch.cat([dec3, conv2], 1))
        dec1 = self.dec1(torch.cat([dec2, conv1], 1))
        x_out = self.final(dec1)
        return x_out


def conv3x3(in_, out):
    return nn.Conv2d(in_, out, 3, padding=1)

class ConvRelu(nn.Module):
    def __init__(self, in_, out):
        super().__init__()
        self.conv = conv3x3(in_, out)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.activation(x)
        return x


class NoOperation(nn.Module):
    def forward(self, x):
        return x


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels):
        super().__init__()

        self.block = nn.Sequential(
            ConvRelu(in_channels, middle_channels),
            nn.ConvTranspose2d(middle_channels, out_channels, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)

class DecoderBlockV2(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels, is_deconv=True):
        super(DecoderBlockV2, self).__init__()
        self.in_channels = in_channels

        if is_deconv:
            """
                Paramaters for Deconvolution were chosen to avoid artifacts, following
                link https://distill.pub/2016/deconv-checkerboard/
            """

            self.block = nn.Sequential(
                ConvRelu(in_channels, middle_channels),
                nn.ConvTranspose2d(middle_channels, out_channels, kernel_size=4, stride=2,
                                   padding=1),
                nn.ReLU(inplace=True)
            )
        else:
            self.block = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear'),
                ConvRelu(in_channels, middle_channels),
                ConvRelu(middle_channels, out_channels),
            )

    def forward(self, x):
        return self.block(x)


class old_unet_vgg16(nn.Module):
    def __init__(self, num_classes=3, encoder_depth=152, num_filters=32, dropout_2d=0.2,
                 pretrained=False, is_deconv=False):
        super().__init__()
        self.num_classes = num_classes
        self.dropout_2d = dropout_2d

        if encoder_depth == 34:
            self.encoder = torchvision.models.resnet34(pretrained=pretrained)
            bottom_channel_nr = 512
        elif encoder_depth == 50:
            self.encoder = torchvision.models.resnet50(pretrained=pretrained)
            bottom_channel_nr = 2048
        elif encoder_depth == 101:
            self.encoder = torchvision.models.resnet101(pretrained=pretrained)
            bottom_channel_nr = 2048
        elif encoder_depth == 152:
            self.encoder = torchvision.models.resnet152(pretrained=pretrained)
            bottom_channel_nr = 2048
        else:
            raise NotImplementedError('only 34, 101, 152 version of Resnet are implemented')

        self.pool = nn.MaxPool2d(2, 2)

        self.relu = nn.ReLU(inplace=True)

        self.conv1 = nn.Sequential(self.encoder.conv1,
                                   self.encoder.bn1,
                                   self.encoder.relu,
                                   self.pool)

        self.conv2 = self.encoder.layer1

        self.conv3 = self.encoder.layer2

        self.conv4 = self.encoder.layer3

        self.conv5 = self.encoder.layer4

        self.center = DecoderBlockV2(bottom_channel_nr, num_filters * 8 * 2, num_filters * 8, is_deconv)
        self.dec5 = DecoderBlockV2(bottom_channel_nr + num_filters * 8, num_filters * 8 * 2, num_filters * 8, is_deconv)
        self.dec4 = DecoderBlockV2(bottom_channel_nr // 2 + num_filters * 8, num_filters * 8 * 2, num_filters * 8,
                                   is_deconv)
        self.dec3 = DecoderBlockV2(bottom_channel_nr // 4 + num_filters * 8, num_filters * 4 * 2, num_filters * 2,
                                   is_deconv)
        self.dec2 = DecoderBlockV2(bottom_channel_nr // 8 + num_filters * 2, num_filters * 2 * 2, num_filters * 2 * 2,
                                   is_deconv)
        self.dec1 = DecoderBlockV2(num_filters * 2 * 2, num_filters * 2 * 2, num_filters, is_deconv)
        self.dec0 = ConvRelu(num_filters, num_filters)
        self.final = nn.Conv2d(num_filters, num_classes, kernel_size=1)

    def forward(self, x):
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)
        conv5 = self.conv5(conv4)

        pool = self.pool(conv5)
        center = self.center(pool)

        dec5 = self.dec5(torch.cat([center, conv5], 1))

        dec4 = self.dec4(torch.cat([dec5, conv4], 1))
        dec3 = self.dec3(torch.cat([dec4, conv3], 1))
        dec2 = self.dec2(torch.cat([dec3, conv2], 1))
        dec1 = self.dec1(dec2)
        dec0 = self.dec0(dec1)

        return self.final(F.dropout2d(dec0, p=self.dropout_2d))


def unet_vgg16(pretrained=True):
    return DensenetUnet(seg_classes=3, backbone_arch='densenet161')


def get_image(imageid, basepath='wdata/dataset', rgbdir='train_rgb', train=True):
    fn = f'{basepath}/{rgbdir}/SN6_Train_AOI_11_Rotterdam_SAR-Intensity_{imageid}.tif' if train else f'{basepath}/{rgbdir}/SN6_Test_Public_AOI_11_Rotterdam_SAR-Intensity_{imageid}.tif'
    #img = cv2.imread(fn, cv2.IMREAD_COLOR)
    #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    #return img
    img = skimage.io.imread(fn)
    return img


def _get_distance_weights(d):
    w1 = 1
    w0 = 50
    sigma = 10

    weights = np.zeros_like(d, dtype=np.float)
    weights = w1 + w0 * np.exp(-(d ** 2) / (sigma ** 2))
    weights[d == 0] = 1

    return weights


class AtlantaDataset(Dataset):
    def __init__(self, image_ids, aug=None, aug2=None, basepath='wdata/dataset'):
        self.image_ids = image_ids
        self.aug = aug
        self.aug2 = aug2
        self.basepath = basepath

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        imageid = self.image_ids[idx]
        im = get_image(imageid, basepath=self.basepath, rgbdir='train_rgb', train=True)
        assert im is not None

        mask = cv2.imread(f'{self.basepath}/masks/full_mask_{imageid}.png')
        assert mask is not None

        rotFlag = lookuprotation(imageid, rot_df)
        if rotFlag:
            for i in range(3):
                im[:, :, i] = orient_sar(im[:, :, i], rotFlag)
                mask[:, :, i] = orient_sar(mask[:, :, i], rotFlag)

        augmented = self.aug(image=im, mask=mask)
        augmented2 = self.aug2(image=augmented['image'].copy())

        mask_ = (augmented['mask'] > 0).astype(np.uint8)
        mask_ = torch.from_numpy(mask_.transpose((2,0,1))).float()

        return (img_to_tensor(np.concatenate((augmented2['image'],augmented2['image'][:,:,2].reshape(512,512,1)),axis=2)), mask_, imageid)

        #return (img_to_tensor(augmented2['image']), mask_, imageid)


class AtlantaTestDataset(Dataset):
    def __init__(self, image_ids, aug=None, rgbdir='test_rgb', basepath='wdata/dataset'):
        self.image_ids = image_ids
        self.aug = aug
        self.basepath = basepath
        self.rgbdir = rgbdir

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        imageid = self.image_ids[idx]
        im = get_image(imageid, basepath=self.basepath, rgbdir=self.rgbdir, train=self.rgbdir=='valid_rgb')
        assert im is not None

        rotFlag = lookuprotation(imageid, rot_df)
        if rotFlag:
            for i in range(3):
                im[:, :, i] = orient_sar(im[:, :, i], rotFlag)

        augmented = self.aug(image=im)

        return img_to_tensor(np.concatenate((augmented['image'],augmented['image'][:,:,2].reshape(512,512,1)),axis=2)), imageid

        #return img_to_tensor(augmented['image']), imageid


@click.group()
def cli():
    pass


@cli.command()
@click.option('--inputs', '-i', default='/data/test',
              help='input directory')
def check(inputs):
    systemcheck_train()
    # TODO: check training images


@cli.command()
@click.option('--inputs', '-i', default='/data/test',
              help='input directory')
@click.option('--working_dir', '-w', default='wdata',
              help="working directory")
def preproctrain(inputs, working_dir):
    """
    * Making 8bit rgb train images
    """
    # preproc images
    Path(f'{working_dir}/dataset/train_rgb').mkdir(parents=True,
                                                   exist_ok=True)

    pool = Pool(1)
    wqueue = []

    src_imgs = list(sorted(Path(inputs).glob('./SAR-Intensity/*.tif')))
    for src in tqdm.tqdm(src_imgs, total=len(src_imgs)):
        dst = f'{working_dir}/dataset/train_rgb/{src.name}'
        if not Path(dst).exists():
            #pan_to_bgr(str(src), dst)
            wqueue.append(pool.apply_async(partial(pan_to_bgr, str(src), dst)))

    for item in tqdm.tqdm(wqueue, total=len(wqueue)):
        item.wait()

    wqueue = []

    # prerpoc masks
    (Path(working_dir) / Path('dataset/masks')).mkdir(parents=True,
                                                      exist_ok=True)
    geojson_dir = Path(inputs) / Path('geojson_buildings')
    mask_dir = Path(working_dir) / Path('dataset/masks')
    geojson_files = list(geojson_dir.glob('./*.geojson'))
    for geojson_fn in tqdm.tqdm(geojson_files, total=len(geojson_files)):
        masks_from_geojson(mask_dir, inputs, geojson_fn)


def create_separation(labels):
    tmp = dilation(labels > 0, square(12))
    tmp2 = watershed(tmp, labels, mask=tmp, watershed_line=True) > 0
    tmp = tmp ^ tmp2
    tmp = dilation(tmp, square(3))

    props = measure.regionprops(labels)

    msk1 = np.zeros_like(labels, dtype='bool')

    for y0 in range(labels.shape[0]):
        for x0 in range(labels.shape[1]):
            if not tmp[y0, x0]:
                continue
            if labels[y0, x0] == 0:
                sz = 5
            else:
                sz = 7
                if props[labels[y0, x0] - 1].area < 300:
                    sz = 5
                elif props[labels[y0, x0] - 1].area < 2000:
                    sz = 6
            uniq = np.unique(labels[max(0, y0 - sz):min(labels.shape[0], y0 + sz + 1),
                             max(0, x0 - sz):min(labels.shape[1], x0 + sz + 1)])
            if len(uniq[uniq > 0]) > 1:
                msk1[y0, x0] = True
    return msk1

def masks_from_geojson(mask_dir, inputs, geojson_fn):
    chip_id = geojson_fn.name.lstrip('SN6_Train_AOI_11_Rotterdam_Buildings_').rstrip('.geojson')
    mask_fn = mask_dir / f'mask_{chip_id}.tif'
    full_mask_fn = mask_dir / f'full_mask_{chip_id}.png'

    if not mask_fn.exists():
        ref_fn = str(Path(inputs) / Path(
            f'SAR-Intensity/SN6_Train_AOI_11_Rotterdam_SAR-Intensity_{chip_id}.tif'))
        cLT.createRasterFromGeoJson(str(geojson_fn), ref_fn, str(mask_fn))
    
    mask = cv2.imread(str(mask_fn), cv2.IMREAD_UNCHANGED)
    assert mask is not None
    
    labels, num_labels = skimage.measure.label(mask, return_num=True)

    full_mask = np.zeros((labels.shape[0], labels.shape[1], 3))
    if num_labels == 0:
        cv2.imwrite(str(full_mask_fn), full_mask)
        return

    for i in range(1, num_labels + 1):
        building_mask = np.zeros_like(labels, dtype='bool')
        building_mask[labels == i] = 1
        area = np.sum(building_mask)
        if area < 200:
            contour_size = 1
        elif area < 500:
            contour_size = 2
        else:
            contour_size = 3
        eroded = binary_erosion(building_mask, iterations=contour_size)
        countour_mask = building_mask ^ eroded
        full_mask[..., 0] += building_mask
        full_mask[..., 1] += countour_mask
    full_mask[..., 2] = create_separation(labels)
    full_mask = np.clip(full_mask * 255, 0, 255)
    cv2.imwrite(str(full_mask_fn), full_mask)


def read_cv_splits(inputs):
    fn = 'working/cv.txt'
    if not Path(fn).exists():
        train_imageids = list(sorted(
            Path(inputs).glob('SAR-Intensity/SN6_Train_AOI_11_Rotterdam_SAR-Intensity_*.tif')))

        # split 4 folds
        df_fold = pd.DataFrame({
            'filename': train_imageids,
            'catid': [path.parent.parent.name for path in train_imageids],
        })
        df_fold.loc[:, 'fold_id'] = np.random.randint(0, 3, len(df_fold))
        df_fold.loc[:, 'ImageId'] = df_fold.filename.apply(
            lambda x: x.name[len('SN6_Train_AOI_11_Rotterdam_SAR-Intensity_'):-4])

        df_fold[[
            'ImageId', 'filename', 'catid', 'fold_id',
        ]].to_csv(fn, index=False)

    return pd.read_csv(fn)


@cli.command()  # noqa: C901
@click.option('--inputs', '-i', default='/data/test',
              help='input directory')
@click.option('--working_dir', '-w', default='./working',
              help="working directory")
@click.option('--fold_id', '-f', default=0, help='fold id')
def train(inputs, working_dir, fold_id):
    start_epoch, step = 0, 0

    global rot_df
    rot_df = readrotationfile(str(Path(inputs) / Path('SummaryData/SAR_orientations.txt')))

    # TopCoder
    num_workers, batch_size = 4, 2 #8, 4 * 8
    gpus = [0] #[0, 1, 2, 3]

    # My machine
    # num_workers, batch_size = 8, 2 * 3
    # gpus = [0, 1]

    patience, n_epochs = 8, 150
    lr, min_lr, lr_update_rate = 5e-5, 5e-6, 0.5
    #lr, min_lr, lr_update_rate = 1e-6, 5e-7, 0.5
    training_timelimit = 60 * 60 * 24 * 2  # 2 days
    st_time = time.time()

    model = unet_vgg16(pretrained=True)
    model = nn.DataParallel(model, device_ids=gpus).cuda()

        #VerticalFlip(p=0.25), 
        #RandomRotate90(p=0.5),
        #ElasticTransform(p=0.25),
        #RandomBrightnessContrast(p=0.1), 
        #GaussNoise(p=0.2, var_limit=2000),

    train_transformer = Compose([
        HorizontalFlip(p=0.5),
        RandomCrop(512, 512, p=1.0),
    ], p=1.0)

    train_transformer2 = Compose([
        Normalize(mean=SAR_MEAN, std=SAR_STD, max_pixel_value=SAR_MAX),
    ], p=1.0)

    val_transformer = Compose([
        CenterCrop(512, 512, p=1.0),
    ], p=1.0)

    val_transformer2 = Compose([
        Normalize(mean=SAR_MEAN, std=SAR_STD, max_pixel_value=SAR_MAX),
    ], p=1.0)

    # train/val loadrs
    df_cvfolds = read_cv_splits(inputs)
    trn_loader, val_loader = make_train_val_loader(
        train_transformer, train_transformer2, 
        val_transformer, val_transformer2, df_cvfolds, fold_id,
        batch_size, num_workers)

    # train
    criterion = binary_loss()
    optimizer = Adam(model.parameters(), lr=lr)

    model_name = f'v12_f{fold_id}'

    ################KUTS
#    del model
#    model = unet_vgg16(pretrained=False)
#    #path = f'working/models/{model_name}/{model_name}_best'
#    path = f'working/models/v12_f0/v12_f0_best'
#    #path = f'wdata/models/v12_f1/v12_f1_ep45_25622'
#    
#    cp = torch.load(path)
#    model = nn.DataParallel(model).cuda()
#    start_epoch = cp['epoch']
#    #start_epoch = 0
#    model.load_state_dict(transform_model_keys(cp['model']))
#    model = model.module
#    model = nn.DataParallel(model, device_ids=gpus).cuda()
#    optimizer = Adam(model.parameters(), lr=lr)
    ##################KUTS

    report_epoch = 10

    fh = open_log(model_name)

    # vers for early stopping
    best_score = 0
    not_improved_count = 0

    for epoch in range(start_epoch, n_epochs):
        model.train()

        tl = trn_loader  # alias
        trn_metrics = Metrics()

        try:
            tq = tqdm.tqdm(total=(len(tl) * trn_loader.batch_size))
            tq.set_description(f'Ep{epoch:>3d}')
            for i, (inputs, targets, names) in enumerate(trn_loader):
                inputs = inputs.cuda()
                targets = targets.cuda()

                outputs = model(inputs)
                loss = criterion(outputs, targets)
                optimizer.zero_grad()

                # Increment step counter
                batch_size = inputs.size(0)
                loss.backward()
                optimizer.step()
                step += 1
                tq.update(batch_size)

                # Update eval metrics
                trn_metrics.loss.append(loss.item())
                #trn_metrics.bce.append(criterion._stash_bce_loss.item())
                trn_metrics.jaccard.append(criterion._stash_jaccard.item())

                if i > 0 and i % report_epoch == 0:
                    report_metrics = Bunch(
                        epoch=epoch,
                        step=step,
                        trn_loss=np.mean(trn_metrics.loss[-report_epoch:]),
                        #trn_bce=np.mean(trn_metrics.bce[-report_epoch:]),
                        trn_jaccard=np.mean(trn_metrics.jaccard[-report_epoch:]),
                    )
                    write_event(fh, **report_metrics)
                    tq.set_postfix(
                        loss=f'{report_metrics.trn_loss:.5f}',
                        #bce=f'{report_metrics.trn_bce:.5f}',
                        jaccard=f'{report_metrics.trn_jaccard:.5f}'
                        )

            # End of epoch
            report_metrics = Bunch(
                epoch=epoch,
                step=step,
                trn_loss=np.mean(trn_metrics.loss[-report_epoch:]),
                #trn_bce=np.mean(trn_metrics.bce[-report_epoch:]),
                trn_jaccard=np.mean(trn_metrics.jaccard[-report_epoch:]),
            )
            write_event(fh, **report_metrics)
            tq.set_postfix(
                loss=f'{report_metrics.trn_loss:.5f}',
                #bce=f'{report_metrics.trn_bce:.5f}',
                jaccard=f'{report_metrics.trn_jaccard:.5f}')
            tq.close()
            save(model, epoch, step, model_name)

            # Run validation
            val_metrics = validation(model,
                                     criterion,
                                     val_loader,
                                     epoch,
                                     step,
                                     fh)
            report_val_metrics = Bunch(
                epoch=epoch,
                step=step,
                val_loss=np.mean(val_metrics.loss[-report_epoch:]),
                #val_bce=np.mean(val_metrics.bce[-report_epoch:]),
                val_jaccard=np.mean(val_metrics.jaccard[-report_epoch:]),
            )
            write_event(fh, **report_val_metrics)

            if time.time() - st_time > training_timelimit:
                tq.close()
                break

            if best_score < report_val_metrics.val_jaccard:
                best_score = report_val_metrics.val_jaccard
                not_improved_count = 0
                copy_best(model, epoch, model_name, step)
            else:
                not_improved_count += 1

            if not_improved_count >= patience:
                # Update learning rate and optimizer

                lr *= lr_update_rate
                # Stop criterion
                if lr < min_lr:
                    tq.close()
                    break

                not_improved_count = 0

                # Load best weight
                del model
                model = unet_vgg16(pretrained=False)
                path = f'working/models/{model_name}/{model_name}_best'
                cp = torch.load(path)
                model = nn.DataParallel(model).cuda()
                epoch = cp['epoch']
                model.load_state_dict(transform_model_keys(cp['model']))
                model = model.module
                model = nn.DataParallel(model, device_ids=gpus).cuda()

                # Init optimizer
                optimizer = Adam(model.parameters(), lr=lr)

        except KeyboardInterrupt:
            save(model, epoch, step, model_name)
            tq.close()
            fh.close()
            sys.exit(1)
        except Exception as e:
            raise e
            break

    fh.close()


def validation(model, criterion, val_loader,
               epoch, step, fh):
    report_epoch = 10
    val_metrics = Metrics()

    with torch.no_grad():
        model.eval()

        vl = val_loader

        tq = tqdm.tqdm(total=(len(vl) * val_loader.batch_size))
        tq.set_description(f'(val) Ep{epoch:>3d}')
        for i, (inputs, targets, names) in enumerate(val_loader):
            inputs = inputs.cuda()
            targets = targets.cuda()

            outputs = model(inputs)
            loss = criterion(outputs, targets)
            tq.update(inputs.size(0))

            val_metrics.loss.append(loss.item())
            #val_metrics.bce.append(criterion._stash_bce_loss.item())
            val_metrics.jaccard.append(criterion._stash_jaccard.item())

            if i > 0 and i % report_epoch == 0:
                report_metrics = Bunch(
                    epoch=epoch,
                    step=step,
                    val_loss=np.mean(val_metrics.loss[-report_epoch:]),
                    #val_bce=np.mean(val_metrics.bce[-report_epoch:]),
                    val_jaccard=np.mean(val_metrics.jaccard[-report_epoch:]),
                )
                tq.set_postfix(
                    loss=f'{report_metrics.val_loss:.5f}',
                    #bce=f'{report_metrics.val_bce:.5f}',
                    jaccard=f'{report_metrics.val_jaccard:.5f}'
                    )

        # End of epoch
        report_metrics = Bunch(
            epoch=epoch,
            step=step,
            val_loss=np.mean(val_metrics.loss[-report_epoch:]),
            #val_bce=np.mean(val_metrics.bce[-report_epoch:]),
            val_jaccard=np.mean(val_metrics.jaccard[-report_epoch:]),
        )
        tq.set_postfix(
            loss=f'{report_metrics.val_loss:.5f}',
            #bce=f'{report_metrics.val_bce:.5f}',
            jaccard=f'{report_metrics.val_jaccard:.5f}')
        tq.close()

    return val_metrics


@attr.s
class Metrics(object):
    loss = attr.ib(default=[])
    bce = attr.ib(default=[])
    jaccard = attr.ib(default=[])


def dice_round(preds, trues):
    preds = preds.float()
    return soft_dice_loss(preds, trues)


def soft_dice_loss(outputs, targets, per_image=False):
    batch_size = outputs.size()[0]
    eps = 1e-5
    if not per_image:
        batch_size = 1
    dice_target = targets.contiguous().view(batch_size, -1).float()
    dice_output = outputs.contiguous().view(batch_size, -1)
    intersection = torch.sum(dice_output * dice_target, dim=1)
    union = torch.sum(dice_output, dim=1) + torch.sum(dice_target, dim=1) + eps
    loss = (1 - (2 * intersection + eps) / union).mean()

    return loss


def jaccard(outputs, targets, per_image=False, non_empty=False, min_pixels=5):
    batch_size = outputs.size()[0]
    eps = 1e-3
    if not per_image:
        batch_size = 1
    dice_target = targets.contiguous().view(batch_size, -1).float()
    dice_output = outputs.contiguous().view(batch_size, -1)
    target_sum = torch.sum(dice_target, dim=1)
    intersection = torch.sum(dice_output * dice_target, dim=1)
    losses = 1 - (intersection + eps) / (torch.sum(dice_output + dice_target, dim=1) - intersection + eps)
    if non_empty:
        assert per_image == True
        non_empty_images = 0
        sum_loss = 0
        for i in range(batch_size):
            if target_sum[i] > min_pixels:
                sum_loss += losses[i]
                non_empty_images += 1
        if non_empty_images == 0:
            return 0
        else:
            return sum_loss / non_empty_images

    return losses.mean()


class DiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True, per_image=False):
        super().__init__()
        self.size_average = size_average
        self.register_buffer('weight', weight)
        self.per_image = per_image

    def forward(self, input, target):
        return soft_dice_loss(input, target, per_image=self.per_image)


class JaccardLoss(nn.Module):
    def __init__(self, weight=None, size_average=True, per_image=False, non_empty=False, apply_sigmoid=False,
                 min_pixels=5):
        super().__init__()
        self.size_average = size_average
        self.register_buffer('weight', weight)
        self.per_image = per_image
        self.non_empty = non_empty
        self.apply_sigmoid = apply_sigmoid
        self.min_pixels = min_pixels

    def forward(self, input, target):
        if self.apply_sigmoid:
            input = torch.sigmoid(input)
        return jaccard(input, target, per_image=self.per_image, non_empty=self.non_empty, min_pixels=self.min_pixels)

class StableBCELoss(nn.Module):
    def __init__(self):
        super(StableBCELoss, self).__init__()

    def forward(self, input, target):
        input = input.float().view(-1)
        target = target.float().view(-1)
        neg_abs = - input.abs()
        # todo check correctness
        loss = input.clamp(min=0) - input * target + (1 + neg_abs.exp()).log()
        return loss.mean()

class FocalLoss2d(nn.Module):
    def __init__(self, gamma=2, ignore_index=255):
        super().__init__()
        self.gamma = gamma
        self.ignore_index = ignore_index

    def forward(self, outputs, targets):
        outputs = outputs.contiguous()
        targets = targets.contiguous()
        eps = 1e-8
        non_ignored = targets.view(-1) != self.ignore_index
        targets = targets.view(-1)[non_ignored].float()
        outputs = outputs.contiguous().view(-1)[non_ignored]
        outputs = torch.clamp(outputs, eps, 1. - eps)
        targets = torch.clamp(targets, eps, 1. - eps)
        pt = (1 - targets) * (1 - outputs) + targets * outputs
        return (-(1. - pt) ** self.gamma * torch.log(pt)).mean()

class ComboLoss(nn.Module):
    def __init__(self, weights, per_image=False, channel_weights=[1, 0.5, 0.5], channel_losses=None):
        super().__init__()
        self.weights = weights
        self.bce = StableBCELoss()
        self.dice = DiceLoss(per_image=False)
        self.jaccard = JaccardLoss(per_image=False)
        self.focal = FocalLoss2d()
        self.mapping = {'bce': self.bce,
                        'focal': self.focal,
                        'dice': self.dice,
                        'jaccard': self.jaccard}
        self.expect_sigmoid = {'dice', 'jaccard'}
        self.per_channel = {'dice', 'jaccard'}
        self.values = {}
        self.channel_weights = channel_weights
        self.channel_losses = channel_losses

    def forward(self, outputs, targets):
        loss = 0
        weights = self.weights
        sigmoid_input = torch.sigmoid(outputs)
        for k, v in weights.items():
            if not v:
                continue
            val = 0
            if k in self.per_channel:
                channels = targets.size(1)
                for c in range(channels):
                    if not self.channel_losses or k in self.channel_losses[c]:
                        val += self.channel_weights[c] * self.mapping[k](sigmoid_input[:, c, ...] if k in self.expect_sigmoid else outputs[:, c, ...],
                                               targets[:, c, ...])

            else:
                val = self.mapping[k](sigmoid_input if k in self.expect_sigmoid else outputs, targets)

            self.values[k] = val
            loss += self.weights[k] * val
        return loss.clamp(min=1e-5)


class binary_loss(object):
    def __init__(self):
        self._stash_jaccard = 0
        #self.combo_loss = ComboLoss(weights={"bce": 1, "jaccard" : 1}, channel_weights=[1, 0.1, 0.5], channel_losses=[["bce", "jaccard"], ["bce", "jaccard"], ["bce", "jaccard"]], per_image=False)
        self.combo_loss = ComboLoss(weights={"focal": 1, "dice" : 1}, channel_weights=[1, 0.1, 0.5], channel_losses=[["focal", "dice"], ["focal", "dice"], ["focal", "dice"]], per_image=False)

    def __call__(self, outputs, targets):
        with torch.no_grad():
            eps = 1e-8
            jaccard_target = (targets[:,0,:,:] == 1).float()
            jaccard_output = torch.sigmoid(outputs[:,0,:,:])

            intersection = (jaccard_output * jaccard_target).sum()
            union = jaccard_output.sum() + jaccard_target.sum()

            jaccard_score = ((intersection + eps) / (union - intersection + eps))
            self._stash_jaccard = jaccard_score
        
        return self.combo_loss(outputs, targets)


def save(model, epoch, step, model_name):
    path = f'wdata/models/{model_name}/{model_name}_ep{epoch}_{step}'
    Path(path).parent.mkdir(parents=True, exist_ok=True)

    torch.save({
        'model': model.state_dict(),
        'epoch': epoch,
        'step': step,
    }, path)


def copy_best(model, epoch, model_name, step):
    path = f'wdata/models/{model_name}/{model_name}_ep{epoch}_{step}'
    best_path = f'working/models/{model_name}/{model_name}_best'
    shutil.copy(path, best_path)


def write_event(log, **data):
    data['dt'] = datetime.datetime.now().isoformat()
    log.write(json.dumps(data, sort_keys=True))
    log.write('\n')
    log.flush()


def open_log(model_name):
    time_str = datetime.datetime.now().strftime('%Y%m%d.%H%M%S')
    path = f'wdata/models/{model_name}/{model_name}.{time_str}.log'
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fh = open(path, 'at', encoding='utf8')
    return fh


def make_train_val_loader(train_transformer, train_transformer2,
                          val_transformer, val_transformer2,
                          df_cvfolds,
                          fold_id,
                          batch_size,
                          num_workers):
    trn_dataset = AtlantaDataset(
        df_cvfolds[df_cvfolds.fold_id != fold_id].ImageId.tolist(),
        aug=train_transformer, aug2=train_transformer2)
    trn_loader = DataLoader(
        trn_dataset,
        sampler=RandomSampler(trn_dataset),
        batch_size=batch_size,
        drop_last=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available())

    val_dataset = AtlantaDataset(
        df_cvfolds[df_cvfolds.fold_id == fold_id].ImageId.tolist(),
        aug=val_transformer, aug2=val_transformer2)
    val_loader = DataLoader(
        val_dataset,
        sampler=SequentialSampler(val_dataset),
        batch_size=batch_size,
        drop_last=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available())
    return trn_loader, val_loader


@cli.command()
@click.option('--inputs', '-i', default='/data/test',
              help='input directory')
@click.option('--working_dir', '-w', default='wdata',
              help="working directory")
@click.option('--output', '-o', default='out.csv',
              help="output filename")
def inference(inputs, working_dir, output):
    print('Collect filenames...')
    test_collection = []

    global rot_df
    rot_df = readrotationfile(str(Path(inputs) / Path('SummaryData/SAR_orientations.txt')))

    src_imgs = list(sorted(
        Path(inputs).glob('SAR-Intensity/SN6_Test_Public_AOI_11_Rotterdam*.tif')))

    for src in src_imgs:
        test_collection.append(src.name)

    print(f'Found {len(test_collection)} test images.')
    assert len(test_collection) > 0

    print(f'Check preprocessed 8bit rgb images')
    for src_img in test_collection:
        assert (
            Path(working_dir) / Path(f'dataset/test_rgb/{src_img}')).exists()

    model_names = [
        'v12_f0_best',
        'v12_f1_best',
        #'v12_f2_best',
    ]
#    for model_name in model_names:
#        inference_by_model(model_name, test_collection)

    model_names = [
        'v12_f0_best',
        'v12_f1_best',
        #'v12_f2_best',
    ]

#    for model_name in model_names:
#        prefix = '_'.join(model_name.split('_')[:2])
#        pred_mask_dir = f'wdata/models/{prefix}/test_{model_name}/'
#        inp = get_inputs('20190822135802_20190822140055_tile_1772.png', str(pred_mask_dir))

#    steps = [step for step in range(0, 100, step_size)]
#    for step in steps:
#        process_images(step)

    # merge prediction masks and write submission file
#    output_fn = str(Path(working_dir) / output)
#    make_sub(model_names, test_collection, output_fn)

@cli.command()
@click.option('--inputs', '-i', default='/data/valid',
              help='input directory')
@click.option('--working_dir', '-w', default='wdata',
              help="working directory")
@click.option('--output', '-o', default='out.csv',
              help="output filename")
def validinference(inputs, working_dir, output):
    print('Collect filenames...')
    test_collection = []

    global rot_df
    rot_df = readrotationfile(str(Path(inputs) / Path('SummaryData/SAR_orientations.txt')))

    src_imgs = list(sorted(
        Path(inputs).glob('Valid-SAR-Intensity/SN6_Train_AOI_11_Rotterdam_SAR-Intensity*.tif')))

    for src in src_imgs:
        test_collection.append(src.name)

    print(f'Found {len(test_collection)} test images.')
    assert len(test_collection) > 0

    print(f'Check preprocessed 8bit rgb images')
    for src_img in test_collection:
        assert (
            Path(working_dir) / Path(f'dataset/valid_rgb/{src_img}')).exists()

    model_names = [
        'v12_f0_best',
        'v12_f1_best',
        #'v12_f2_best',
    ]
#    for model_name in model_names:
#        inference_by_model_valid(model_name, test_collection)

    # merge prediction masks and write submission file
    output_fn = str(Path(working_dir) / output)
    make_sub_valid(model_names, test_collection, output_fn)


pixels_threshold = 76
sep_count = 3
sep_thresholds = [0.6, 0.7, 0.8]

#def get_inputs(filename, pred_folder, truth_folder=None):
def get_inputs(pred):
    inputs = []
    #pred = cv2.imread(os.path.join(pred_folder, filename), cv2.IMREAD_COLOR)

    pred_msk = pred / 255.
    pred_msk = pred_msk[..., 0] * (1 - pred_msk[..., 2])
    pred_msk = 1 * (pred_msk > 0.7)

    pred_msk = pred_msk.astype(np.uint8)

    y_pred = measure.label(pred_msk, neighbors=8, background=0)
    props = measure.regionprops(y_pred)
    for i in range(len(props)):
        if props[i].area < 20:
            y_pred[y_pred == i + 1] = 0
    y_pred = measure.label(y_pred, neighbors=8, background=0)

    nucl_msk = (255 - pred[..., 0])
    nucl_msk = nucl_msk.astype('uint8')
    y_pred = watershed(nucl_msk, y_pred, mask=((pred[..., 0] > pixels_threshold)), watershed_line=True)

    props = measure.regionprops(y_pred)

    for i in range(len(props)):
        if props[i].area < 50:
            y_pred[y_pred == i + 1] = 0

    pred_labels = measure.label(y_pred, neighbors=8, background=0)
    pred_props = measure.regionprops(pred_labels)

    #    img = cv2.imread(path.join(img_folder, filename), cv2.IMREAD_COLOR)

    coords = np.array([pr.centroid for pr in pred_props])
    if len(coords) > 0:
        t = KDTree(coords)
        neighbors100 = t.query_radius(coords, r=50)
        neighbors200 = t.query_radius(coords, r=100)
        neighbors300 = t.query_radius(coords, r=150)
        neighbors400 = t.query_radius(coords, r=200)
    med_area = np.median(np.asarray([pr.area for pr in pred_props]))

    lvl2_labels = [np.zeros_like(pred_labels, dtype=np.int) for i in range(sep_count)]

    separated_regions = [[] for i in range(sep_count)]
    main_regions = [[] for i in range(sep_count)]

    for i in range(len(pred_props)):
        msk_reg = pred_labels[pred_props[i].bbox[0]:pred_props[i].bbox[2], pred_props[i].bbox[1]:pred_props[i].bbox[3]] == i + 1
        pred_reg = pred[pred_props[i].bbox[0]:pred_props[i].bbox[2], pred_props[i].bbox[1]:pred_props[i].bbox[3]]

        contours = cv2.findContours((msk_reg * 255).astype(dtype=np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours[0]) > 0:
            cnt = contours[0][0]
            min_area_rect = cv2.minAreaRect(cnt)

        inp = []
        inp.append(10)
        inp.append(pred_props[i].area)
        inp.append(0)
        if len(contours[0]) > 0:
            inp.append(cv2.isContourConvex(cnt) * 1.0)
            inp.append(min(min_area_rect[1]))
            inp.append(max(min_area_rect[1]))
            if max(min_area_rect[1]) > 0:
                inp.append(min(min_area_rect[1]) / max(min_area_rect[1]))
            else:
                inp.append(0)
            inp.append(min_area_rect[2])
        else:
            inp.append(0)
            inp.append(0)
            inp.append(0)
            inp.append(0)
            inp.append(0)
        inp.append(pred_props[i].convex_area)
        inp.append(pred_props[i].solidity)
        inp.append(pred_props[i].eccentricity)
        inp.append(pred_props[i].extent)
        inp.append(pred_props[i].perimeter)
        inp.append(pred_props[i].major_axis_length)
        inp.append(pred_props[i].minor_axis_length)
        if (pred_props[i].minor_axis_length > 0):
            inp.append(pred_props[i].minor_axis_length / pred_props[i].major_axis_length)
        else:
            inp.append(0)

        pred_values = pred_reg[..., 0][msk_reg]

        inp.append(pred_values.mean())
        inp.append(pred_values.std())
        #
        inp.append(neighbors100[i].shape[0])
        median_area = med_area
        if neighbors100[i].shape[0] > 0:
            neighbors_areas = np.asarray([pred_props[j].area for j in neighbors100[i]])
            median_area = np.median(neighbors_areas)
        inp.append(median_area)
        inp.append(pred_props[i].area / median_area)

        inp.append(neighbors200[i].shape[0])
        median_area = med_area
        if neighbors200[i].shape[0] > 0:
            neighbors_areas = np.asarray([pred_props[j].area for j in neighbors200[i]])
            median_area = np.median(neighbors_areas)
        inp.append(median_area)
        inp.append(pred_props[i].area / median_area)

        inp.append(neighbors300[i].shape[0])
        median_area = med_area
        if neighbors300[i].shape[0] > 0:
            neighbors_areas = np.asarray([pred_props[j].area for j in neighbors300[i]])
            median_area = np.median(neighbors_areas)
        inp.append(median_area)
        inp.append(pred_props[i].area / median_area)

        inp.append(neighbors400[i].shape[0])
        median_area = med_area
        if neighbors400[i].shape[0] > 0:
            neighbors_areas = np.asarray([pred_props[j].area for j in neighbors400[i]])
            median_area = np.median(neighbors_areas)
        inp.append(median_area)
        inp.append(pred_props[i].area / median_area)

        bst_j = 0
        pred_reg[~msk_reg] = 0
        pred_reg0 = pred_reg / 255.
        pred_reg0 = pred_reg0[..., 0] * (1 - pred_reg0[..., 1])
        max_regs = 1
        for j in range(1, sep_count + 1):
            sep_regs = []
            if bst_j > 0:
                separated_regions[j - 1].append(sep_regs)
                continue
            pred_reg2 = 255 * (pred_reg0 > sep_thresholds[j - 1])
            pred_reg2 = pred_reg2.astype(np.uint8)
            if j > sep_count - 1:
                kernel = np.ones((3, 3), np.uint8)
                pred_reg2 = cv2.erode(pred_reg2, kernel, iterations=2)
            lbls = measure.label(pred_reg2, neighbors=4, background=False)
            num_regs = lbls.max()
            if num_regs > 1:
                bst_j = j
                max_regs = num_regs
            if num_regs > 1 or (j < sep_count and num_regs > 0):
                lbls = lbls.astype(np.int32)
                labels_ws = watershed((255 - pred_reg[..., 0]), lbls, mask=msk_reg)
                start_num = len(main_regions[j - 1])
                labels_ws += start_num
                labels_ws[labels_ws == start_num] = 0
                for k in range(num_regs):
                    sep_regs.append(k + start_num)
                    main_regions[j - 1].append(i)
                lvl2_labels[j - 1][pred_props[i].bbox[0]:pred_props[i].bbox[2], pred_props[i].bbox[1]:pred_props[i].bbox[3]] += labels_ws
            separated_regions[j - 1].append(sep_regs)

        inp.append(bst_j)
        inp.append(max_regs)
        inp.append(1)
        inp.append(0)

        inputs.append(np.asarray(inp))
    inputs = np.asarray(inputs)
    all_sep_props = []
    all_sep_inputs = []
    for j in range(sep_count):
        inputs_lvl2 = []
        pred_props2 = measure.regionprops(lvl2_labels[j])
        for i in range(len(pred_props2)):
            msk_reg = lvl2_labels[j][pred_props2[i].bbox[0]:pred_props2[i].bbox[2], pred_props2[i].bbox[1]:pred_props2[i].bbox[3]] == i + 1
            pred_reg = pred[pred_props2[i].bbox[0]:pred_props2[i].bbox[2], pred_props2[i].bbox[1]:pred_props2[i].bbox[3], 0]

            contours = cv2.findContours((msk_reg * 255).astype(dtype=np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            if len(contours[0]) > 0:
                cnt = contours[0][0]
                min_area_rect = cv2.minAreaRect(cnt)

            inp = []
            inp.append(10)
            inp.append(pred_props2[i].area)
            main_area = inputs[main_regions[j][i]][0]
            inp.append(pred_props2[i].area / main_area)
            if len(contours[0]) > 0:
                inp.append(cv2.isContourConvex(cnt) * 1.0)
                inp.append(min(min_area_rect[1]))
                inp.append(max(min_area_rect[1]))
                if max(min_area_rect[1]) > 0:
                    inp.append(min(min_area_rect[1]) / max(min_area_rect[1]))
                else:
                    inp.append(0)
                inp.append(min_area_rect[2])
            else:
                inp.append(0)
                inp.append(0)
                inp.append(0)
                inp.append(0)
                inp.append(0)
            inp.append(pred_props2[i].convex_area)
            inp.append(pred_props2[i].solidity)
            inp.append(pred_props2[i].eccentricity)
            inp.append(pred_props2[i].extent)
            inp.append(pred_props2[i].perimeter)
            inp.append(pred_props2[i].major_axis_length)
            inp.append(pred_props2[i].minor_axis_length)
            if (pred_props2[i].minor_axis_length > 0):
                inp.append(pred_props2[i].minor_axis_length / pred_props2[i].major_axis_length)
            else:
                inp.append(0)

            pred_values = pred_reg[msk_reg]

            inp.append(pred_values.mean())
            inp.append(pred_values.std())
            #
            inp.append(inputs[main_regions[j][i]][-16])
            median_area = inputs[main_regions[j][i]][-15]
            inp.append(median_area)
            inp.append(pred_props2[i].area / median_area)

            inp.append(inputs[main_regions[j][i]][-13])
            median_area = inputs[main_regions[j][i]][-12]
            inp.append(median_area)
            inp.append(pred_props2[i].area / median_area)

            inp.append(inputs[main_regions[j][i]][-10])
            median_area = inputs[main_regions[j][i]][-9]
            inp.append(median_area)
            inp.append(pred_props2[i].area / median_area)

            inp.append(inputs[main_regions[j][i]][-7])
            median_area = inputs[main_regions[j][i]][-6]
            inp.append(median_area)
            inp.append(pred_props2[i].area / median_area)

            bst_j = inputs[main_regions[j][i]][-4]
            max_regs = inputs[main_regions[j][i]][-3]

            inp.append(bst_j)
            inp.append(max_regs)
            inp.append(len(separated_regions[j][main_regions[j][i]]))
            inp.append(j + 1)

            inputs_lvl2.append(np.asarray(inp))
        all_sep_props.append(pred_props2)
        inputs_lvl2 = np.asarray(inputs_lvl2)
        all_sep_inputs.append(inputs_lvl2)

    return inputs, pred_labels, all_sep_inputs, lvl2_labels, separated_regions


num_split_iters = 1
folds_count = 30

pixels_threshold = 76
sep_count = 3
sep_thresholds = [0.6, 0.7, 0.8]

lgbm_models_folder = 'lgbm_models'

test_out_folders = ['lgbm_test_sub1']
color_out_folders = ['color_test_sub1']

best_thrs = [0.3]
step_size=20

def process_images(img):
    gbm_models = []

    for it in range(num_split_iters):
        for it2 in range(folds_count):
            gbm_models.append(
                lgb.Booster(model_file=os.path.join(lgbm_models_folder, 'gbm_model_{0}_{1}.txt'.format(it, it2))))

    paramss = [img]
#    files = list(reversed(sorted(listdir(test_pred_folder))))
#    for filename in files[step:step + step_size]:
#        if path.isfile(path.join(test_pred_folder, filename)) and '.png' in filename:
#            paramss.append((filename, test_pred_folder, None))

    inputs = []
    inputs2 = []
    labels = []
    labels2 = []
    separated_regions = []
    results = [get_inputs(param) for param in paramss]

    for i in range(len(results)):
        inp, lbl, inp2, lbl2, sep_regs = results[i]
        inputs.append(inp)
        inputs2.append(inp2)
        labels.append(lbl)
        labels2.append(lbl2)
        separated_regions.append(sep_regs)
    for sub_id in range(1):
        bst_k = np.zeros((sep_count + 1))
        removed = 0
        replaced = 0
        total_cnt = 0
        im_idx = 0

        empty_cnt = 0

        #for filename in files[step:step + step_size]:
        for iii in range(1):
            #if path.isfile(path.join(test_pred_folder, filename)) and '.png' in filename:
            if True:
                #img_id = filename

                inp = inputs[im_idx]
                pred = np.zeros((inp.shape[0]))
                pred2 = [np.zeros((inp2.shape[0])) for inp2 in inputs2[im_idx]]

                for m in gbm_models:
                    if pred.shape[0] > 0:
                        pred += m.predict(inp)
                    for k in range(len(inputs2[im_idx])):
                        if pred2[k].shape[0] > 0:
                            pred2[k] += m.predict(inputs2[im_idx][k])
                if pred.shape[0] > 0:
                    pred /= len(gbm_models)
                for k in range(len(pred2)):
                    if pred2[k].shape[0] > 0:
                        pred2[k] /= len(gbm_models)

                pred_labels = np.zeros_like(labels[im_idx], dtype='uint16')

                clr = 1

                for i in range(pred.shape[0]):
                    max_sep = -1
                    max_pr = pred[i]
                    for k in range(len(separated_regions[im_idx])):
                        if len(separated_regions[im_idx][k][i]) > 0:
                            pred_lvl2 = pred2[k][separated_regions[im_idx][k][i]]
                            if len(pred_lvl2) > 1 and pred_lvl2.mean() > max_pr:
                                max_sep = k
                                max_pr = pred_lvl2.mean()
                                break
                            if len(pred_lvl2) > 1 and pred_lvl2.max() > max_pr:
                                max_sep = k
                                max_pr = pred_lvl2.max()

                    if max_sep >= 0:
                        pred_lvl2 = pred2[max_sep][separated_regions[im_idx][max_sep][i]]
                        replaced += 1
                        for j in separated_regions[im_idx][max_sep][i]:
                            if pred2[max_sep][j] > best_thrs[sub_id]:
                                pred_labels[labels2[im_idx][max_sep] == j + 1] = clr
                                clr += 1
                            else:
                                removed += 1
                    else:
                        if pred[i] > best_thrs[sub_id]:
                            pred_labels[labels[im_idx] == i + 1] = clr
                            clr += 1
                        else:
                            removed += 1
                    bst_k[max_sep + 1] += 1

                return pred_labels.astype(np.int16)

                clr_labels = label2rgb(pred_labels, bg_label=0)
                clr_labels *= 255
                clr_labels = clr_labels.astype('uint8')

                return clr_labels

                cv2.imwrite(path.join(pred_folder, color_out_folders[sub_id], img_id), clr_labels, [cv2.IMWRITE_PNG_COMPRESSION, 9])

                total_cnt += pred_labels.max()

                cv2.imwrite(path.join(pred_folder, test_out_folders[sub_id], filename[:-4]+".tif"), pred_labels)
                im_idx += 1

        print('total_cnt', total_cnt, 'removed', removed, 'replaced', replaced, 'empty:', empty_cnt)
        print(bst_k)


def make_sub(model_names, test_collection, output_fn):  # noqa: C901
    chip_summary_list = []
    with tempfile.TemporaryDirectory() as tempdir:
        tq = tqdm.tqdm(total=(len(test_collection)))
        tq.set_description(f'(avgfolds)')
        for name in test_collection:
            tq.update(1)
            y_pred_avg = np.zeros((900, 900), dtype=np.float32)

            imageid = name.lstrip('SN6_Test_Public_AOI_11_Rotterdam_SAR-Intensity_').rstrip('.tif')
            for model_name in model_names:
                # Prediction mask
                prefix = '_'.join(model_name.split('_')[:2])
                pred_mask_dir = f'wdata/models/{prefix}/test_{model_name}/'
                y_pred = np.array(ss.load_npz(
                    str(Path(pred_mask_dir) / Path(f'{imageid}.npz'))
                ).todense() / 255.0)
                y_pred_avg += y_pred
            y_pred_avg /= len(model_names)

            rotFlag = lookuprotation(name, rot_df)

            #import pdb
            #pdb.set_trace()

            if rotFlag:
                y_pred_avg = orient_sar(y_pred_avg, rotFlag, direction=1)

            # Remove small objects
            y_pred = (y_pred_avg > 0.5)
            y_pred_label = skimage.measure.label(y_pred)

            min_area_thresh = 80

            for lbl_idx in np.unique(y_pred_label):
                if (y_pred_label == lbl_idx).sum() < min_area_thresh:
                    y_pred_label[y_pred_label == lbl_idx] = 0

            # to_summary
            simplification_threshold = 0
            preds_test = (y_pred_label > 0).astype('uint8')
            pred_geojson_path = str(Path(tempdir) / Path(f'{name}.json'))

            geotiff_path = f'/mnt/extdisk/contests/spacenet6/test/test_public/AOI_11_Rotterdam/SAR-Intensity'
            im_fname = f'SN6_Test_Public_AOI_11_Rotterdam_SAR-Intensity_{imageid}.tif'

            try:
                raw_test_im = rasterio.open(
                    os.path.join(geotiff_path, im_fname))
                shapes = cLT.polygonize(
                    preds_test,
                    raw_test_im.profile['transform'])
                geom_list = []
                raster_val_list = []
                for s in shapes:
                    geom_list.append(shape(s[0]).simplify(
                        tolerance=simplification_threshold,
                        preserve_topology=False))
                    raster_val_list.append(s[1])
                feature_gdf = gpd.GeoDataFrame({
                    'geometry': geom_list,
                    'rasterVal': raster_val_list})
                feature_gdf.crs = raw_test_im.profile['crs']
                feature_gdf['conf'] = 1
                gT.exporttogeojson(pred_geojson_path, feature_gdf)
            except ValueError:
                # print(f'Warning: Empty prediction array for {name}')
                pass

            chip_summary = {
                'chipName': im_fname,
                'geoVectorName': pred_geojson_path,
                'imageId': imageid,
                'geotiffPath': geotiff_path,
            }
            chip_summary_list.append(chip_summary)

        tq.close()
        __createCSVSummaryFile(chip_summary_list, output_fn, pixPrecision=2)

def make_sub_valid(model_names, test_collection, output_fn):  # noqa: C901
    chip_summary_list = []
    with tempfile.TemporaryDirectory() as tempdir:
        tq = tqdm.tqdm(total=(len(test_collection)))
        tq.set_description(f'(avgfolds)')
        for name in test_collection:
            tq.update(1)
            y_pred_avg = np.zeros((900, 900, 3), dtype=np.float32)

            imageid = name.lstrip('SN6_Train_AOI_11_Rotterdam_SAR-Intensity_').rstrip('.tif')
            for model_name in model_names:
                # Prediction mask
                prefix = '_'.join(model_name.split('_')[:2])
                pred_mask_dir = f'wdata/models/{prefix}/valid_{model_name}/'
#                y_pred = np.array(ss.load_npz(
#                    str(Path(pred_mask_dir) / Path(f'{imageid}.png'))
#                ).todense() / 255.0)
                y_pred = cv2.imread(str(Path(pred_mask_dir) / Path(f'{imageid}.png')), 
                        cv2.IMREAD_COLOR) / 255.0
                #import pdb
                #pdb.set_trace()
                y_pred_avg += y_pred
            y_pred_avg /= len(model_names)

            rotFlag = lookuprotation(name, rot_df)

            #import pdb
            #pdb.set_trace()

            if rotFlag:
                y_pred_avg = orient_sar(y_pred_avg, rotFlag, direction=1)

            y_pred_avg = process_images(y_pred_avg * 255)

            # Remove small objects
            #y_pred = (y_pred_avg > 0.5)
            y_pred = y_pred_avg > 0
            y_pred_label = skimage.measure.label(y_pred)

            min_area_thresh = 20

            for lbl_idx in np.unique(y_pred_label):
                if (y_pred_label == lbl_idx).sum() < min_area_thresh:
                    y_pred_label[y_pred_label == lbl_idx] = 0

            # to_summary
            simplification_threshold = 0
            preds_test = (y_pred_label > 0).astype('uint8')
            pred_geojson_path = str(Path(tempdir) / Path(f'{name}.json'))

            geotiff_path = f'/mnt/extdisk/contests/spacenet6/train/AOI_11_Rotterdam/Valid-SAR-Intensity/'
            im_fname = f'SN6_Train_AOI_11_Rotterdam_SAR-Intensity_{imageid}.tif'

            try:
                raw_test_im = rasterio.open(
                    os.path.join(geotiff_path, im_fname))
                shapes = cLT.polygonize(
                    preds_test,
                    raw_test_im.profile['transform'])
                geom_list = []
                raster_val_list = []
                for s in shapes:
                    geom_list.append(shape(s[0]).simplify(
                        tolerance=simplification_threshold,
                        preserve_topology=False))
                    raster_val_list.append(s[1])
                feature_gdf = gpd.GeoDataFrame({
                    'geometry': geom_list,
                    'rasterVal': raster_val_list})
                feature_gdf.crs = raw_test_im.profile['crs']
                feature_gdf['conf'] = 1
                gT.exporttogeojson(pred_geojson_path, feature_gdf)
            except ValueError:
                # print(f'Warning: Empty prediction array for {name}')
                pass

            chip_summary = {
                'chipName': im_fname,
                'geoVectorName': pred_geojson_path,
                'imageId': imageid,
                'geotiffPath': geotiff_path,
            }
            chip_summary_list.append(chip_summary)

        tq.close()
        __createCSVSummaryFile(chip_summary_list, output_fn, pixPrecision=2)


def __createCSVSummaryFile(chipSummaryList, outputFileName, pixPrecision=2):
    with open(outputFileName, 'w') as csvfile:
        writerTotal = csv.writer(csvfile, delimiter=',', lineterminator='\n')
        writerTotal.writerow([
            'ImageId', 'BuildingId', 'PolygonWKT_Pix', 'Confidence'])

        # TODO: Add description=createCSVSummaryFile
        for chipSummary in tqdm.tqdm(chipSummaryList,
                                     total=len(chipSummaryList),
                                     desc='createCSVSummaryFile'):
            chipName = chipSummary['chipName']
            geoVectorName = chipSummary['geoVectorName']
            rasterChipDirectory = chipSummary['geotiffPath']
            imageId = chipSummary['imageId']

            buildingList = gT.geoJsonToPixDF(
                geoVectorName,
                rasterName=os.path.join(rasterChipDirectory, chipName),
                affineObject=[],
                gdal_geomTransform=[],
                pixPrecision=pixPrecision)
            buildingList = gT.explodeGeoPandasFrame(buildingList)

            if len(buildingList) > 0:
                for idx, building in buildingList.iterrows():
                    tmpGeom = dumps(building.geometry,
                                    rounding_precision=pixPrecision)
                    writerTotal.writerow([imageId, idx, tmpGeom, 1])
            else:
                imageId = chipSummary['imageId']
                writerTotal.writerow([imageId, -1,
                                      'LINESTRING EMPTY', 1])


def inference_by_model(model_name, filenames,
                       batch_size=2,
                       num_workers=0,
                       fullsize_mode=False):
    # TODO: Optimize parameters for p2.xlarge
    print(f'Inrefernce by {model_name}')
    prefix = '_'.join(model_name.split('_')[:2])
    model_checkpoint_file = f'working/models/{prefix}/{model_name}'

    pred_mask_dir = f'wdata/models/{prefix}/test_{model_name}/'
    Path(pred_mask_dir).mkdir(parents=True, exist_ok=True)

    model = unet_vgg16(pretrained=False)
    cp = torch.load(model_checkpoint_file)
    if 'module.final.weight' in cp['model']:
        model = nn.DataParallel(model).cuda()
        epoch = cp['epoch']
        model.load_state_dict(transform_model_keys(cp['model']))
        model = model.module
        model = model.cuda()
    else:
        epoch = cp['epoch']
        model.load_state_dict(transform_model_keys(cp['model']))
        model = model.cuda()

    image_ids = [
        Path(path).name.lstrip('SN6_Test_Public_AOI_11_Rotterdam_SAR-Intensity_').rstrip('.tif')
        for path in Path('wdata/dataset/test_rgb/').glob(
            '*.tif')]

    tst_transformer = Compose([
        #Normalize()
        Normalize(mean=SAR_MEAN, std=SAR_STD, max_pixel_value=SAR_MAX),
    ], p=1.0)
    tst_dataset = AtlantaTestDataset(image_ids, aug=tst_transformer)
    tst_loader = DataLoader(
        tst_dataset,
        sampler=SequentialSampler(tst_dataset),
        batch_size=batch_size,
        drop_last=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available())

    with torch.no_grad():
        tq = tqdm.tqdm(total=(len(tst_loader) * tst_loader.batch_size))
        tq.set_description(f'(test) Ep{epoch:>3d}')
        for X, names in tst_loader:
            tq.update(X.size(0))

            # TODO
            if fullsize_mode:
                pass
            else:
                pass

            for j, name in enumerate(names):
                # Image level inference
                # 900 -> 512 crop
                X_ = torch.stack([
                    X[j, :, :512, :512],
                    X[j, :, -512:, :512],
                    X[j, :, :512, -512:],
                    X[j, :, -512:, -512:],
                ])

                y_pred = np.zeros((3, 900, 900), dtype=np.float32)
                y_pred_weight = np.zeros((3, 900, 900), dtype=np.uint8)
                inputs = X_.cuda()
                outputs = model(inputs)
                y_pred_sigmoid = np.clip(torch.sigmoid(
                    torch.squeeze(outputs)
                ).detach().cpu().numpy(), 0.0, 1.0)

                y_pred[:, :512, :512] += y_pred_sigmoid[0]
                y_pred_weight[:, :512, :512] += 1
                y_pred[:, -512:, :512] += y_pred_sigmoid[1]
                y_pred_weight[:, -512:, :512] += 1
                y_pred[:, :512, -512:] += y_pred_sigmoid[2]
                y_pred_weight[:, :512, -512:] += 1
                y_pred[:, -512:, -512:] += y_pred_sigmoid[3]
                y_pred_weight[:, -512:, -512:] += 1
                y_pred = y_pred / y_pred_weight

                # Save quanlized values
                #y_pred_mat = ss.csr_matrix(
                    #np.round(y_pred * 255).astype(np.uint8))
#                ss.save_npz(
#                    str(Path(pred_mask_dir) / Path(f'{name}.npz')),
#                    y_pred_mat)

                y_pred_mat = np.round(y_pred * 255).astype(np.uint8)

                y_pred_mat = y_pred_mat.transpose((1, 2, 0))

                cv2.imwrite(str(Path(pred_mask_dir) / Path(f'{name}.png')), y_pred_mat,
                        [cv2.IMWRITE_PNG_COMPRESSION, 9])

        tq.close()


def inference_by_model_valid(model_name, filenames,
                       batch_size=2,
                       num_workers=0,
                       fullsize_mode=False):
    # TODO: Optimize parameters for p2.xlarge
    print(f'Inrefernce by {model_name}')
    prefix = '_'.join(model_name.split('_')[:2])
    model_checkpoint_file = f'working/models/{prefix}/{model_name}'

    pred_mask_dir = f'wdata/models/{prefix}/valid_{model_name}/'
    Path(pred_mask_dir).mkdir(parents=True, exist_ok=True)

    model = unet_vgg16(pretrained=False)
    cp = torch.load(model_checkpoint_file)
    if 'module.final.weight' in cp['model']:
        model = nn.DataParallel(model).cuda()
        epoch = cp['epoch']
        model.load_state_dict(transform_model_keys(cp['model']))
        model = model.module
        model = model.cuda()
    else:
        epoch = cp['epoch']
        model.load_state_dict(transform_model_keys(cp['model']))
        model = model.cuda()

    image_ids = [
        Path(path).name.lstrip('SN6_Train_AOI_11_Rotterdam_SAR-Intensity_').rstrip('.tif')
        for path in Path('wdata/dataset/valid_rgb/').glob(
            '*.tif')]

    tst_transformer = Compose([
        #Normalize()
        Normalize(mean=SAR_MEAN, std=SAR_STD, max_pixel_value=SAR_MAX),
    ], p=1.0)
    tst_dataset = AtlantaTestDataset(image_ids, aug=tst_transformer, rgbdir='valid_rgb')
    tst_loader = DataLoader(
        tst_dataset,
        sampler=SequentialSampler(tst_dataset),
        batch_size=batch_size,
        drop_last=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available())

    with torch.no_grad():
        tq = tqdm.tqdm(total=(len(tst_loader) * tst_loader.batch_size))
        tq.set_description(f'(test) Ep{epoch:>3d}')
        for X, names in tst_loader:
            tq.update(X.size(0))

            # TODO
            if fullsize_mode:
                pass
            else:
                pass

            for j, name in enumerate(names):
                # Image level inference
                # 900 -> 512 crop
                X_ = torch.stack([
                    X[j, :, :512, :512],
                    X[j, :, -512:, :512],
                    X[j, :, :512, -512:],
                    X[j, :, -512:, -512:],
                ])

                y_pred = np.zeros((3, 900, 900), dtype=np.float32)
                y_pred_weight = np.zeros((3, 900, 900), dtype=np.uint8)
                inputs = X_.cuda()
                outputs = model(inputs)
                y_pred_sigmoid = np.clip(torch.sigmoid(
                    torch.squeeze(outputs)
                ).detach().cpu().numpy(), 0.0, 1.0)

                y_pred[:, :512, :512] += y_pred_sigmoid[0]
                y_pred_weight[:, :512, :512] += 1
                y_pred[:, -512:, :512] += y_pred_sigmoid[1]
                y_pred_weight[:, -512:, :512] += 1
                y_pred[:, :512, -512:] += y_pred_sigmoid[2]
                y_pred_weight[:, :512, -512:] += 1
                y_pred[:, -512:, -512:] += y_pred_sigmoid[3]
                y_pred_weight[:, -512:, -512:] += 1
                y_pred = y_pred / y_pred_weight

                # Save quanlized values
#                y_pred_mat = ss.csr_matrix(
#                    np.round(y_pred * 255).astype(np.uint8))
#                ss.save_npz(
#                    str(Path(pred_mask_dir) / Path(f'{name}.npz')),
#                    y_pred_mat)
#
                y_pred_mat = np.round(y_pred * 255).astype(np.uint8)

                y_pred_mat = y_pred_mat.transpose((1, 2, 0))

                cv2.imwrite(str(Path(pred_mask_dir) / Path(f'{name}.png')), y_pred_mat,
                        [cv2.IMWRITE_PNG_COMPRESSION, 9])
        tq.close()

@cli.command()
@click.option('--inputs', '-i', default='/data/test',
              help='input directory')
@click.option('--working_dir', '-w', default='wdata',
              help="working directory")
def preproctest(inputs, working_dir):
    """
    * Making 8bit rgb test images
    """
    (Path(working_dir) / Path('dataset/test_rgb')).mkdir(parents=True,
                                                         exist_ok=True)

    # rgb images
    src_imgs = list(sorted(Path(inputs).glob('./SAR-Intensity/*.tif')))
    for src in tqdm.tqdm(src_imgs, total=len(src_imgs)):
        dst = f'{working_dir}/dataset/test_rgb/{src.name}'
        if not Path(dst).exists():
            pan_to_bgr(str(src), dst)

@cli.command()
@click.option('--inputs', '-i', default='/data/valid',
              help='input directory')
@click.option('--working_dir', '-w', default='wdata',
              help="working directory")
def preprocvalid(inputs, working_dir):
    """
    * Making 8bit rgb test images
    """
    (Path(working_dir) / Path('dataset/valid_rgb')).mkdir(parents=True,
                                                         exist_ok=True)

    # rgb images
    src_imgs = list(sorted(Path(inputs).glob('./Valid-SAR-Intensity/*.tif')))
    for src in tqdm.tqdm(src_imgs, total=len(src_imgs)):
        dst = f'{working_dir}/dataset/valid_rgb/{src.name}'
        if not Path(dst).exists():
            pan_to_bgr(str(src), dst)


def pan_to_bgr(src, dst, thresh=3000):
    bands = [1,2,4]
    with rasterio.open(src, 'r') as reader:
        img = np.empty((reader.height,
                        reader.width,
                        len(bands)))
        for i, band in enumerate(bands):
            banddata = reader.read(band)
            img[:, :, i] = banddata

    img = np.clip(img[:, :, :3], None, thresh)
    img = np.floor(img).astype('uint8')
    cv2.imwrite(dst, img)

def pan_to_bgr_rgb(src, dst, thresh=3000):
    bands = [1,2,3]
    with rasterio.open(src, 'r') as reader:
        img = np.empty((reader.height,
                        reader.width,
                        len(bands)))
        for i, band in enumerate(bands):
            img[:, :, i] = reader.read(band)
    img = np.clip(img[:, :, :3], None, thresh)
    img = np.floor(img).astype('uint8')
    cv2.imwrite(dst, img)
    #os.symlink(src, dst)


@cli.command()
@click.option('--inputs', '-i', default='./test',
              help="input directory")
@click.option('--working_dir', '-w', default='wdata',
              help="working directory")
def filecheck(inputs, working_dir):
    # check test images generated by sp4 baseline code
    filecheck_inference_models(working_dir)
    systemcheck_inference()
    # filecheck_inference_images(working_dir)

    # check train images generated by sp4 baseline code
    # check train masks generated by sp4 baseline code
    # print("Something is wrong. Contact with the author.")


def filecheck_inference_models(working_dir):
    checklist = [
        'working/models/v12_f0/v12_f0_best',
        #'working/models/v12_f1/v12_f1_best',
        #'working/models/v12_f2/v12_f2_best',
    ]

    is_ok = True
    for path in checklist:
        is_ok &= __filecheck(Path(path))

    is_warn = True
    cp = torch.load('working/models/v12_f0/v12_f0_best')
    #is_warn &= helper_assertion_check("Check v12_f0_best.step == 80206",
                                      #cp['step'] == 80206)
#    cp = torch.load('working/models/v12_f1/v12_f1_best')
#    is_warn &= helper_assertion_check("Check v12_f1_best.step == 92874",
#                                      cp['step'] == 92874)
#    cp = torch.load('working/models/v12_f2/v12_f2_best')
#    is_warn &= helper_assertion_check("Check v12_f2_best.step == 95034",
#                                      cp['step'] == 95034)


def filecheck_inference_images(working_dir):
    # inputs: dataset directory
    checklist = [
        "dataset/test_rgb/",
    ]

    is_ok = True
    for path_fmt in checklist:
        path = Path(working_dir) / Path(path_fmt)
        is_ok &= __filecheck(path)


def __filecheck(path, max_length=80):
    print(path, end='')
    if len(str(path)) > max_length - 6:
        print('\t', end='')
    else:
        space_size = max_length - 6 - len(str(path))
        print(space_size * ' ', end='')

    if path.exists():
        print('[ \x1b[6;32;40m' + 'OK' + '\x1b[0m ]')
        return True
    else:
        print('[ \x1b[6;31;40m' + 'NG' + '\x1b[0m ]')
        return False


def systemcheck_inference():
    assert helper_assertion_check("Check CUDA device is available",
                                  torch.cuda.is_available())


def systemcheck_train():
    assert helper_assertion_check("Check CUDA device is available",
                                  torch.cuda.is_available())
    assert helper_assertion_check("Check CUDA device count == 1",
                                  torch.cuda.device_count() == 1)


def helper_assertion_check(msg, res, max_length=80):
    print(msg, end='')
    if len(msg) > max_length - 6:
        print('\t', end='')
    else:
        space_size = max_length - 6 - len(msg)
        print(space_size * ' ', end='')

    if res:
        print('[ \x1b[6;32;40m' + 'OK' + '\x1b[0m ]')
        return True
    else:
        print('[ \x1b[6;31;40m' + 'NG' + '\x1b[0m ]')
        return False


if __name__ == "__main__":
    cli()
