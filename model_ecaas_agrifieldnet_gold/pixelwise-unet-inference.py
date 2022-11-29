#!/usr/bin/env python
# coding: utf-8

import warnings, logging
warnings.filterwarnings("ignore")
import os
import glob, json, math
import getpass, random, gc
import rasterio
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold
from time import sleep
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold, KFold
from sklearn.preprocessing import LabelEncoder, LabelBinarizer
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt
from sklearn.metrics import log_loss, f1_score, precision_score, recall_score, accuracy_score, roc_auc_score
import multiprocessing
import joblib
from joblib import Parallel, delayed
from scipy.stats import gmean
import operator
from functools import reduce
from collections import Counter
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.optim.lr_scheduler import LambdaLR
from torchvision import transforms as TT
from accelerate import Accelerator


def seed_all(seed = 3407):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = True
        
    return seed

if __name__ == '__main__':

    seed_all()


    path = os.environ['INPUT_DATA']
    model_path = os.path.join(path, 'checkpoint/unet-model/')
    output_path = os.environ['OUTPUT_DATA']

    main = 'ref_agrifieldnet_competition_v1'
    assets = ['field_ids','raster_labels']
    source_collection = f'{main}_source'
    test_label_collection = f'{main}_labels_test'

    selected_bands = ['B01', 'B02', 'B03', 'B04','B05', 'B06', 'B07', 'B08','B8A', 'B09', 'B11', 'B12']
    n_selected_bands = len(selected_bands)

    mapper = {
      1: 0, 2: 1,
      3: 2, 4: 3,
      5: 4, 6: 5,
      8: 6, 9: 7,
      13: 8, 14: 9,
      15: 10, 16: 11,
      36: 12
    }

    n_splits = 13
    img_sh = 256

    mean_ = [43.37711395, 38.76292241, 37.587551  , 39.3978952 , 42.61577028,
            54.78574474, 63.25995855, 59.99860118, 69.70862906, 13.36703553,
            69.21308285, 48.3234563 ,  0.22776309,  0.37217344,  0.30184635,
             0.1163135 ,  0.22690262,  0.99895458, 0.]
    std_ = [ 3.33574659,  4.16081291,  5.43403711,  9.23910118,  8.01432914,
             6.74542607,  8.07007347,  7.8449207 ,  9.25897753,  2.56338205,
            16.96799286, 15.59297373,  0.25121888,  0.37174745,  0.33190777,
             0.47001154,  0.24938266,  2.71884689, 1.0]

    ###############################################################################################
    ######################################## Utilities ############################################
    ###############################################################################################

    def get_folder_ids(label_collection):
      with open (f'{path}/{main}/{label_collection}/collection.json') as f:
          train_json = json.load(f)

      folder_ids = [i['href'].split('_')[-1].split('.')[0] for i in train_json['links'][4:]]

      return folder_ids


    def build_collection_df(label_collection):
      folder_ids = get_folder_ids(label_collection)

      data = [
          {
            'unique_folder_id': i,
            'field_paths': f'{path}/{main}/{label_collection}/{label_collection}_{i}/field_ids.tif',
            'label_paths': f'{path}/{main}/{label_collection}/{label_collection}_{i}/raster_labels.tif'
          }

          for i in folder_ids
      ]

      competition_data = pd.DataFrame(data)

      return competition_data

    def build_field2label_data(data):
      field_ids = []
      crop_type = []

      for i in tqdm(range(len(data))):
          with rasterio.open(data['field_paths'].iloc[i]) as src:
              field_data = src.read()[0]
          with rasterio.open(data['label_paths'].iloc[i]) as src:
              crop_data = src.read()[0]

          for field_id in np.unique(field_data)[1:]:
              ind = np.where(field_data == field_id)
              field_ids.append(field_id)
              crop_type.append(np.unique(crop_data[ind])[-1])

      df = pd.DataFrame(np.array([field_ids, crop_type]).transpose(),
                        columns=['field_id', 'crop_id'])
      return df


    def extract_field_crop_data(row):
      with rasterio.open(row.field_paths) as src:
          field_data = src.read()[0].astype('int16')

      try:
        with rasterio.open(row.label_paths) as src:
            label_data = src.read()[0].astype('int16')
      except:
        label_data = np.zeros((256, 256))

      return field_data, label_data

    def get_tile_label(label_paths):
      with rasterio.open(label_paths) as src:
          label_data = src.read()[0].astype('int16')

      ulabel = [mapper[x] for x in np.unique(label_data.flatten()) if x != 0 ]

      mlabels = np.zeros((13))
      mlabels[ulabel] = 1

      return mlabels


    def band_extractor(row):
      bands_src = [
          rasterio.open(f'{path}/{main}/{source_collection}/{source_collection}_{row.unique_folder_id}/{band}.tif') for band in selected_bands
      ]
      bands = [ band.read(1).reshape(1, img_sh, img_sh) for band in bands_src]
      bands = np.vstack(bands)

      return bands

    def compute_band_indices(bands):
        b01 = bands[0]
        b02 = bands[1]
        b03 = bands[2]
        b04 = bands[3]
        b05 = bands[4]
        b06 = bands[5]
        b07 = bands[6]
        b08 = bands[7]
        b8a = bands[8]
        b09 = bands[9]
        b11 = bands[10]
        b12 = bands[11]

        eps = 0

        NDVI = ((b08 - b04)/(b08 + b04 + eps))
        EVI2 = 2.5 * (b08 - b04) / (b08 + 2.4*b04 + 1)
        SAVI = ((b08 - b04)/(b08 + b04 + 0.33)) * (1 + 0.33)
        NDRE = (b08 - b06)/(b08 + b06 + eps)
        OSAVI = (b08 - b04) / (b08 + b04 + 0.35)
        FIDET = b12 / (b8a * b09  + eps)


        return np.stack([NDVI, EVI2, SAVI, NDRE, OSAVI, FIDET], axis=0)


    def read_band_data(row, add_mask=False):
      field_data, label_data = extract_field_crop_data(row)
      bands = band_extractor(row)
      indices = compute_band_indices(bands)

      data = [bands, indices]
      if add_mask:
        data.append( (field_data != 0).astype(float).reshape(1, *bands[0].shape) )

      data = np.vstack(data)

      return field_data, data, label_data

    def paralellize(fct, data, size, verbose=0, with_tqdm=False):
        fn = map(delayed(fct), data)
        if with_tqdm:
            fn = tqdm(fn, total=size)
        return Parallel(n_jobs=-1, verbose=verbose, backend="multiprocessing")(fn)

    def flatAcc(l):
      return reduce(operator.add, l)

    ###############################################################################
    ######################### Unet ################################################
    ###############################################################################

    class ConvLayer(nn.Sequential):
      def __init__(self, in_channels, out_channels):
        conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=1)
        bn = nn.BatchNorm2d(out_channels)
        relu = nn.ReLU(inplace=True)

        super(ConvLayer, self).__init__(conv, bn, relu)  

    class ConvBlock(nn.Sequential):
      def __init__(self, in_channels, out_channels):
        conv1 = ConvLayer(in_channels, out_channels)
        conv2 = ConvLayer(out_channels, out_channels)

        super(ConvBlock, self).__init__(conv1, conv2)

    class PermuteLayer(nn.Module):
      def __init__(self, p):
        super().__init__()

        self.p = p

      def forward(self, x):
        return x.permute(self.p)

    class PadLayer(nn.Module):
      def forward(self, x, skip):
        w = skip.size()[2] - x.size()[2]
        h = skip.size()[3] - x.size()[3]

        return F.pad(x, [w // 2, w - w // 2, h // 2, h - h // 2])


    class SEBlock(nn.Module):
      def __init__(self, in_channels, factor=2):
        super(SEBlock, self).__init__()

        self.block = nn.Sequential(
            nn.AdaptiveMaxPool2d((1,1)),
            PermuteLayer((0,2,3,1)),
            nn.Linear(in_channels, in_channels//2),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels//2, in_channels),
            PermuteLayer((0,3,2,1))
        )


      def forward(self, x):
        return x * torch.sigmoid(self.block(x))

    class Downsample(nn.Sequential):
      def __init__(self, in_channels, out_channels, se=False):

        max_pool = nn.MaxPool2d(2)
        conv = ConvBlock(in_channels, out_channels)
        sed = SEBlock(out_channels) if se else nn.Identity()

        super(Downsample, self).__init__(max_pool, conv, sed)

    class Upsample(nn.Module):
      def __init__(self, in_channels, out_channels):
        super(Upsample, self).__init__()

        self.up = nn.ConvTranspose2d(in_channels, in_channels//2, kernel_size=2, stride=2)
        self.pad = PadLayer()
        self.conv = ConvBlock(in_channels, out_channels)

      def forward(self, x, skip):
        x = self.up(x)
        x = self.pad(x, skip)
        x = torch.cat([skip, x], dim=1)
        x = self.conv(x)

        return x

    class Unet(nn.Module):
      def __init__(self, n_classes, in_channels, depth=2, first_conv_out = 64):
        super(Unet, self).__init__()

        out_channels = [ first_conv_out ] + [
            64 * 2 * (i+1) for i in range(depth)
        ]
        encoder_blocks = [
            Downsample(in_, out_, True) for in_, out_ in zip(out_channels[:-1], out_channels[1:])
        ]
        decoder_blocks = [
            Upsample(in_ * 2, out_ * 2) for in_, out_ in zip(out_channels[::-1][:-1], out_channels[::-1][1:])
        ]

        self.encoder = nn.ModuleList(
            [ ConvBlock(in_channels, out_channels[0]) ] + encoder_blocks
        )

        self.center = Downsample(out_channels[-1], out_channels[-1] * 2, False)

        self.decoder = nn.ModuleList(
            decoder_blocks + [ Upsample(out_channels[1], out_channels[0]) ]
        )

        self.head = nn.Conv2d(out_channels[0], n_classes, kernel_size=1)

      def forward(self, x):
        skips = []
        for i, block in enumerate(self.encoder):
          x = block(x)
          skips.append(x)

        x = self.center(x)

        for i, block in enumerate(self.decoder):
          skip = skips.pop()
          x = block(x, skip)

        x = self.head(x)

        return x


    ################################################################################
    ########################## Loaders & Models definition #########################
    ################################################################################


    class CustomDataset(Dataset):
        def __init__(self, df, tfms=None, n_channels=19, **kwargs):
            super(CustomDataset, self).__init__()

            self.tfms = tfms
            self.df = df.reset_index(drop=True)
            self.default_tfms = TT.Compose([
              TT.Normalize(mean=mean_, std=std_),
            ])
            self.label_mapper = {
              1: 0, 2: 1,
              3: 2, 4: 3,
              5: 4, 6: 5,
              8: 6, 9: 7,
              13: 8, 14: 9,
              15: 10, 16: 11,
              36: 12, 0: -100
            }

        def __len__(self):
            return len(self.df)

        def map_label(self, label):
            return np.vectorize(lambda x: self.label_mapper[x])(label)

        def __getitem__(self, index):
            row = self.df.iloc[index]

            field, image, label = read_band_data(row, add_mask=True)

            image = torch.tensor(image, dtype=torch.float).nan_to_num(nan=0., posinf=0., neginf=0.)
            image = self.default_tfms(image)

            label = self.map_label(label)

            output = {
                'field': torch.tensor(field, dtype=torch.long),
                'image': image,
                'target': torch.tensor(label, dtype=torch.long)
              }

            return output

    class CustomModel(nn.Module):
        def __init__(self, args):
            super().__init__()
            self.model = Unet(args.n_classes, args.n_channels, args.depth, args.first_conv_out)
            self.args = args
            # self.loss_fn = get_loss(self.args)

        def forward(self, x, labels=None):
            logits = self.model(x)

            if labels is not None:
                loss = self.loss_fn(logits, labels)

                return (logits, loss)
            return logits

    class TensorProcessor:
      def __init__(self):
        self.p = PermuteLayer((1, 2, 0))
        self.p_hat = PermuteLayer((0, 2, 3, 1))

      def _train_processor(self, y_hat, y_true, to_numpy):
        bs, nc, h, w = y_hat.size()

        y_true = self.p(y_true).reshape(bs * h * w)
        y_hat = self.p_hat(y_hat).reshape(bs * h * w, nc)

        mask = y_true != -100
        y_true = y_true[mask]
        y_hat = y_hat[mask]
        y_hat = F.softmax(y_hat, dim=1)

        if to_numpy:
          return y_true.numpy(), y_hat.numpy()

        return y_true, y_hat

      def _infer_processor(self, y_hat, to_numpy):
        bs, nc, h, w = y_hat.size()

        y_hat = self.p_hat(y_hat).reshape(bs * h * w, nc)
        y_hat = F.softmax(y_hat, dim=1)

        if to_numpy:
          return y_hat.numpy()

        return y_hat

      def __call__(self, y_hat, y_true=None, to_numpy=True):
        if y_true is not None:
          return self._train_processor(y_hat, y_true, to_numpy)

        return self._infer_processor(y_hat, to_numpy)

    def compute_agg_prediction(args, df, models, istest=False, agg='mean'):
      predictions, all_fields, all_labels = cls_inference(df, args, models, istest)

      chunk_size = len(all_fields)

      agg_predictions = []
      agg_labels = []
      agg_fields = []

      for chunk in tqdm(range(chunk_size), total=chunk_size, desc='chunk'):
        pred_chunk = predictions[chunk]
        field_chunk = all_fields[chunk]
        if not istest:
          label_chunk = all_labels[chunk]

        for f in tqdm(np.unique(field_chunk)):
          if f == 0:
            continue

          mask = field_chunk == f
          if agg == 'gmean':
            f_data = gmean(pred_chunk[mask], axis=0)
          else:
            f_data = np.mean(pred_chunk[mask], axis=0)

          if not istest:
            l = np.unique(label_chunk[mask])[0]

          agg_predictions.append(f_data)
          agg_fields.append(f)
          if not istest:
            agg_labels.append(l)

      agg_predictions = np.array(agg_predictions)
      agg_fields = np.array(agg_fields)
      if not istest:
        agg_labels = np.array(agg_labels)

      return agg_fields, agg_predictions, agg_labels

    def get_accelerator():
      try:
        accelerator = Accelerator(fp16=torch.cuda.is_available())
      except:
        accelerator = Accelerator()

      return accelerator

    def get_testloader(df, size, bs, tta=False):
        tfms = None

        test_ds = CustomDataset(df, tfms=tfms)
        test_dl = DataLoader(test_ds, bs, shuffle=False)

        accelerator = get_accelerator()
        test_dl = accelerator.prepare(test_dl)

        return test_dl

    def cls_inference(df, args, MODELS, istest):
        all_outputs = []
        all_fields = []
        all_labels = []

        testloader = get_testloader(df, args.size, args.ebs)
        processor = TensorProcessor()

        loader_size = len(testloader)
        n_model = len(MODELS)

        with torch.no_grad():
          for i, data in enumerate(tqdm(testloader)):

            x = data['image']
            field = data['field']
            if not istest:
              label = data['target']

            m_logits = []
            for m, model in enumerate(MODELS):
                m_logits.append(
                    processor(model(x).detach().cpu())
                )

            logits = gmean(m_logits, axis=0)

            field = field.reshape(logits.shape[0])
            all_fields.append( field.detach().cpu().numpy() )

            if not istest:
              label = label.reshape(logits.shape[0])
              all_labels.append( label.detach().cpu().numpy() )

            all_outputs.append( logits )

            del x, field, m_logits, logits, data
            # clear_gpu()

            # if i % 10 == 0:
            #     sleep(2)
            #     clear_gpu()
            #     sleep(2)

        return all_outputs, all_fields, all_labels

    def clear_gpu():
        torch.cuda.empty_cache()
        gc.collect()

    def load_models(args, path='models/'):
        accelerator = get_accelerator()

        MODELS = []

        for i in tqdm(args.use_folds, desc='Loading models...'):
            model = CustomModel(args)
            model.load_state_dict(
              torch.load(f'{path}unet-model-{i}.pt', map_location=torch.device(accelerator.device)),
              strict=False
            )
            model.eval()

            model = accelerator.prepare(model)

            MODELS.append( model )

        return MODELS

    class Logger:
        def __init__(self):
            handler = logging.StreamHandler()
            handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
            logger = logging.getLogger("agrifield")
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)

            self.logger = logger

        def log(self, msg):
            self.logger.info(msg)

    class Config:
        use_folds = list(range(n_splits))
        pretrained = True

        device = 'cuda'
        bs = 64
        ebs = 32
        size = 300
        grad_accumulation = 1

        tta = True
        num_tta = 5

        n_classes = 13

        lr = 3e-4
        epochs = 10

        n_channels = 19
        depth = 2
        first_conv_out = 64

        loss = 'ce'
        loss_type = 'focal'
        alpha = 0.
        gamma = 2.0
        loss_weight = None

        schedule = False
        warmup_epochs = 0.2
        factor = 0.8
        patience = 0
        min_lr = 1e-6
        verbose = True

        def __init__(self, **kwargs):
            self.__update(**kwargs)
            self.__asserts()

        def __update(self, **kwargs):
            for k,v in kwargs.items():
                setattr(self, k, v)

        def __asserts(self):
          assert self.grad_accumulation >= 1, 'grad_accumulation should be greater or equal than 1'

        @classmethod
        def clone(cls, obj, **kwargs):
            new_kwargs = obj.to_json()
            new_kwargs.update(kwargs)

            return cls(**new_kwargs)

        def to_json(self):
            return {k: getattr(self, k) for k in dir(self) if not k.startswith('__') and str(type(getattr(self, k))) != "<class 'method'>"}

    ######################################################################################
    ####################################### Inference ####################################
    ######################################################################################
    test_competition_data = build_collection_df(test_label_collection)


    logger = Logger()
    args = Config(
      bs = 32, ebs = 64, lr=1e-4, epochs = 20, schedule='none', depth=2, n_channels=19,
    )

    MODELS = load_models(args, model_path)


    args.ebs = 32 #Inference batch size
    args.tta = False
    test_fields, test_predictions, _ = compute_agg_prediction(args, test_competition_data, MODELS, istest=True)   

    crop_names = [
    'Wheat','Mustard','Lentil','No Crop','Green pea','Sugarcane','Garlic','Maize','Gram','Coriander','Potato','Bersem','Rice'
    ]


    subs = pd.DataFrame(test_fields, columns=['field_id'])
    subs[crop_names] = test_predictions
    subs = subs.groupby('field_id').mean().reset_index()


    subs.to_csv(f'{output_path}/pixelwise-unet.csv', index=False)
