import os
os.environ["KERAS_BACKEND"]        = "mxnet"

import glob, re
import warnings, time, datetime, logging
import sys,subprocess,multiprocessing

import numpy as np, pandas as pd, matplotlib.pyplot as plt
import sklearn, sklearn.pipeline, sklearn.model_selection, sklearn.preprocessing
import mxnet as mx
import central_arena_view as cav
# import gluon_wrappers as gw
# import keras
# from keras import backend as K

import tqdm

model_directory = os.path.dirname(os.path.realpath(__file__))

DEFAULT_MODEL_SUFFIX = 'default'

log = logging.getLogger('model_base_mx')

class Transform():
    def __init__(self, logger, name='nop-transform'):
        self.logger = logger
        self.name = name
        pass

    def get_name(self):
        return self.name

    def in_game_transform(self, game_state, action):
        pass

    def batch_transform(self, id, use_cached=True, augment_with_penalty_moves=False):
        pass

    def batch_transform_X_y(self, in_df):
        pass

action_options = [
    ('WAIT',(1,0,0,0,0,0)),
    ('UP',(0,1,0,0,0,0)),
    ('LEFT',(0,0,1,0,0,0)),
    ('DOWN',(0,0,0,1,0,0)),
    ('RIGHT',(0,0,0,0,1,0)),
    ('BOMB',(0,0,0,0,0,1))
    ]



action_options_df = pd.DataFrame([(1, 0, 0, 0, 0, 0),
                                  (0, 1, 0, 0, 0, 0),
                                  (0, 0, 1, 0, 0, 0),
                                  (0, 0, 0, 1, 0, 0),
                                  (0, 0, 0, 0, 1, 0),
                                  (0, 0, 0, 0, 0, 1)],
                                 columns=cav.DFIDs.A_ONE_HOT)

class BaseTransform(Transform):
    def __init__(self, logger, name='base-transform', size=5):
        super().__init__(logger, name=name)
        logger.debug('setup')
        self.size = size
        self.action_option_dict = dict(action_options)

        self.input_columns = cav.DFIDs.A_ONE_HOT + \
                             self.transform_nearest_object_info_columns(cav.PACAV.nearest_other_agent_info_columns) + \
                             self.transform_nearest_object_info_columns(cav.PACAV.nearest_coin_info_columns) + \
                             self.transform_nearest_object_info_columns(cav.PACAV.nearest_crate_info_columns) + \
                             self.transform_nearest_object_info_columns(cav.PACAV.mid_of_map_info_columns) + \
                             cav.TC.get_transformation(size)

        self.df0 = pd.DataFrame(columns=self.input_columns)

    def transform_nearest_object_info_columns(self, columns):
        e0   = columns[0]
        rest = columns[1:]
        r = [e0 + 'x', e0 + 'y'] + rest
        return r

    def get_name(self):
        return '{}-size{}'.format(self.name, self.size)

    def in_game_transform(self, game_state, action=None, validate=False):
        self.logger.debug('in_game_transform: start')

        # 15ms:
        av = cav.PandasAugmentedCentralArenaView(game_state)
        ldf = av.to_df()
        self.logger.debug('in_game_transform: PandasAugmentedCentralArenaView done')

        t = cav.FeatureSelectionTransformation0(ldf, size=self.size)
        self.logger.debug('in_game_transform: FeatureSelectionTransformation0 created')
        out_npa = t.in_game_transform(av)
        self.logger.debug('in_game_transform: out_npa created')
        out_npa_ = out_npa.reshape(1,-1)
        # out_df = self.df0.copy(deep=True)
        # out_df.loc[0] = out_npa
        out_df  = pd.DataFrame(out_npa_, columns=self.input_columns)
        if validate:
            out_df_ = t.transform()[self.input_columns].copy()
            lds = out_df.iloc[0,:] == out_df_.iloc[0,:]
            if not lds.all():
                raise Exception('Validation between in_game_transform and transform failed: lds: {}'.format(lds))
        self.logger.debug('in_game_transform: FeatureSelectionTransformation0 done')

        # 23ms
        if action is not None:
            action_one_hot = self.action_option_dict[action]
            out_df.loc[0, cav.DFIDs.A_ONE_HOT] = action_one_hot
        else:
            out_df = pd.concat([out_df]*len(action_options_df), axis=0).reset_index(drop=True)
            # for i in range(len(action_options_df) - 1):
            #     idx = len(out_df)
            #     out_df.loc[idx] = out_df.loc[0].copy()

            # for i in range(len(action_options_df)):
            #     out_df.loc[i, cav.DFIDs.A_ONE_HOT] = action_options_df.loc[i,:]

            out_df.loc[:, cav.DFIDs.A_ONE_HOT] = action_options_df.loc[:,:]

        self.logger.debug('in_game_transform: A_ONE_HOT done')

        return out_df, av

    def predict_max_qq(self, model, ldf):
        r = np.zeros((len(ldf), len(cav.DFIDs.A_ONE_HOT)))
        i = 0
        for name, onehot in action_options:
            ldf.loc[:,cav.DFIDs.A_ONE_HOT] = onehot
            r[:,i] = model.predict(ldf)
            i += 1

        r = np.max(r,axis=1)
        return r


    def augment_predictions(self, in_file_names, model):
        transform_time = 0.0
        predict_time   = 0.0
        write_time     = 0.0
        for fn in in_file_names:
            with pd.HDFStore(fn) as s:
                keys = s.keys()
                for key in keys:
                    if s[key][cav.DFIDs.QQ_Pred_Max].isnull().values.any():
                        ldf_ = s[key]
                        time1 = time.time()
                        t = cav.FeatureSelectionTransformation0(ldf_, size=self.size)
                        ldf = t.transform(correct_survival_qq_next_state_max=False)[self.input_columns]
                        time2 = time.time()
                        y = self.predict_max_qq(model, ldf) #model.predict(ldf)
                        time3 = time.time()
                        ldf_.loc[:,cav.DFIDs.QQ_Pred_Max] = y
                        s.put(key, ldf_)
                        time4 = time.time()
                        transform_time += time2 - time1
                        predict_time   += time3 - time2
                        write_time     += time4 - time3
                        # ldf_ = s[key]
                        # pass
        self.logger.debug('augment_predictions: transform_time: {:.3f}, predict_time: {:.3f}, write_time: {:.3f}'.format(transform_time, predict_time, write_time))


    def batch_transform(self, id, use_cached=True, augment_with_penalty_moves=False, model=None):
        file_name_pattern = './hdf5_training_data/{}-{}.h5'
        out_file_name = file_name_pattern.format(id, self.get_name())
        rdf  = None
        rdf_ = None
        if use_cached and os.path.isfile(out_file_name):
            self.logger.info('Loading transformed data from file: {}'.format(out_file_name))
            with pd.HDFStore(out_file_name, mode='r') as s:
                rdf = s['df'].copy()
        else:
            in_file_names = [file_name_pattern.format(id, i) for i in range(4)]

            if model is not None:
                time1 = time.time()
                self.augment_predictions(in_file_names, model)
                time2 = time.time()
                self.logger.debug('batch_transform augment_predictions {:.3f} ms'.format((time2 - time1) * 1000.0))
            else:
                raise Exception('You need to provide a model to make sure that all QQNextStateMax values can be filled!')

            time1 = time.time()
            postprocess = cav.PostProcessGame(in_file_names, out_file_name, size=self.size)
            postprocess.process()
            self.logger.debug(postprocess.time_info)
            del postprocess
            time2 = time.time()
            self.logger.debug('batch_transform: PostProcessGame took {:.3f} ms'.format((time2 - time1) * 1000.0))

            with pd.HDFStore(out_file_name) as s:
                rdf = s['df'].copy()

        if augment_with_penalty_moves:
            time1 = time.time()
            penalty_moves_transform = cav.AugmentGameDataWithPenaltyMoves(rdf)
            rdf_ = penalty_moves_transform.process()
            time2 = time.time()
            del penalty_moves_transform
            self.logger.debug('AugmentGameDataWithPenaltyMoves took {:.3f} ms'.format((time2 - time1) * 1000.0))

        return rdf, rdf_

    def batch_transform_X_y(self, in_df):
        X = in_df[self.input_columns]
        y = in_df['QQ']
        return X, y




def get_model_load_path(name, suffix=DEFAULT_MODEL_SUFFIX, selected_id=None, latest=True):
    glob_list = glob.glob('{}/*-symbol.json'.format(model_directory))
    # print(glob_list)

    model_options = []
    for g in glob_list:
        r = re.search(r'^.*/(\d+)-(.*?)-(.*?)-symbol\.json$', g)
        if r:
            id         = r.group(1)
            agent_name = r.group(2)
            suffix     = r.group(3)
            model_options += [(id, agent_name, suffix)]
        else:
            raise Exception('The glob does not match the pattern: {}'.format(g))

    # print(model_options)

    # print(name, suffix)
    model_id_options = [int(id) for id, an, sfx in model_options if an == name and sfx == suffix]
    # print(model_id_options)
    if len(model_id_options) == 0:
        return None
    id = None
    if latest:
        id = np.max(model_id_options)
    else:
        if int(selected_id) not in model_id_options:
            raise Exception('The selected_id is not in the model_id_options: {}, {}'.format(selected_id, model_id_options))
        id = int(selected_id)

    return '{}/{}'.format(model_directory, BaseModel.MODEL_NAME_PATTERN.format(id, name, suffix))

class BaseModel(object):

    MODEL_NAME_PATTERN = '{}-{}-{}'

    def __init__(self, name, fn, logger, transform, auto_save=True):
        self.logger = logger
        if '-' in name:
            raise Exception('Model names must not contain "-" characters: {}'.format(name))

        self.num_gpus  = mx.context.num_gpus()
        self.ctx       = mx.gpu(0) if mx.context.num_gpus() > 0 else mx.cpu(0)
        self.logger.info('number of gpus: {}, ctx: {}'.format(self.num_gpus, self.ctx))

        self.transform = transform
        self.auto_save = auto_save
        if fn is not None:
            self.model     = gw.GluonRegressor(model_fn=fn, batch_size=8 * 256, model_ctx=self.ctx, epochs=2, auto_save=auto_save)

        self.name = name
        self.path = self.create_model_save_path()
        self.action_options = action_options
        self.trainer = None


    def get_transform(self):
        return self.transform

    def create_model_save_path(self, suffix=DEFAULT_MODEL_SUFFIX):
        id = int(time.mktime(datetime.datetime.now().timetuple()))
        file_name = BaseModel.MODEL_NAME_PATTERN.format(id, self.name, suffix)
        return '{}/{}'.format(model_directory, file_name)

    def fit(self, X_train, y_train, model_save_path=None, epochs=2, batchsize=8 * 256, **kwargs):
        if model_save_path is None:
            model_save_path = self.create_model_save_path()
        self.path = model_save_path
        self.logger.info('model_save_path: {}'.format(model_save_path))

        return self.model.fit(X_train, y_train, batch_size=batchsize, epochs=epochs, verbose=1, validation_split=0.1, model_save_path=model_save_path)#, callbacks=model_callback_list

    def save(self):
        self.model.save(self.path)

    def load(self, id=None, suffix=None, latest=True):
        model_load_path_ = get_model_load_path(self.name, selected_id=id, latest=latest)
        model_load_path = '{}-symbol.json'.format(model_load_path_)
        # model_params_path = '{}-0000.params'.format(model_load_path_)

        self.logger.info("Trying to load model: {}".format(model_load_path))

        if model_load_path and os.path.isfile(model_load_path):
            self.model.load(model_load_path) # , model_params_path
            self.logger.info("Model loaded: {}".format(model_load_path))
        else:
            self.logger.info("No model is found")

    def predict(self, X):
        return self.model.predict(X)

    def evaluate(self, X, y):
        return self.model.evaluate(X, y)

def create_net_(base=mx.gluon.nn.Sequential):
    ACTIVATION = 'relu'
    net = base(prefix='MLP_')
    with net.name_scope():
        net.add(
            # mx.gluon.nn.Flatten(),
            mx.gluon.nn.Dense(300, activation=ACTIVATION, prefix='fc-1_'),
            mx.gluon.nn.Dense(100, activation=ACTIVATION, prefix='fc-2_'),
            mx.gluon.nn.Dense(1 , activation=None       , prefix='predictions')
        )
    return net

def very_simple():
    net = create_net_(base=mx.gluon.nn.HybridSequential)
    net.hybridize(static_shape=True, static_alloc=True)
    return net

class VerySimple5(BaseModel):
    def __init__(self, logger):
        self.transform = BaseTransform(logger, size=5)
        super().__init__("VerySimpleMX5", very_simple, logger, self.transform)


class NCHWTransform(Transform):
    def __init__(self, logger, name='nchw-transform', size=11):
        super().__init__(logger, name=name)
        logger.debug('setup')
        self.size = size
        self.action_option_dict = dict(action_options)

        self.base_transform = BaseTransform(logger, size=size)
        self.input_columns  = self.base_transform.input_columns

    def get_name(self):
        return '{}-size{}'.format(self.name, self.size)

    def in_game_transform(self, game_state, action=None):
        self.logger.debug('in_game_transform: start')

        ldf = self.base_transform.in_game_transform(game_state, action=action)

        out_df = ldf
        return out_df

    def batch_transform(self, id, use_cached=True, augment_with_penalty_moves=False, model=None):

        rdf, rdf_ = self.base_transform.batch_transform(id, use_cached=use_cached, augment_with_penalty_moves=augment_with_penalty_moves, model=model)

        if augment_with_penalty_moves:
            t = cav.FeatureSelectionTransformationNCHW(rdf_)
        else:
            t = cav.FeatureSelectionTransformationNCHW(rdf)
        rxds = t.transform()

        return rxds

    def batch_transform_iter(self, id, use_cached=True, augment_with_penalty_moves=False, batch_size = 600000, model=None):
        rdf, rdf_ = self.base_transform.batch_transform(id, use_cached=use_cached, augment_with_penalty_moves=augment_with_penalty_moves, model=model)

        if augment_with_penalty_moves:
            rdf = rdf_

        n = len(rdf)
        rnd_idx =np.random.permutation(np.arange(0, n))
        rdf = rdf.iloc[rnd_idx,:]

        for i in range(0,n,batch_size):
            e = min(n, i+batch_size)

            ldf = rdf.iloc[i:e,:]
            t = cav.FeatureSelectionTransformationNCHW(ldf)
            rxds = t.transform()
            yield rxds





# def vgg_(base=mx.gluon.nn.Sequential):
#
#     ACTIVATION = 'relu'
#
#     def make_features(layers, filters, batch_norm=True):
#
#         featurizer = mx.gluon.nn.HybridSequential(prefix='')
#         for i, num in enumerate(layers):
#             for _ in range(num):
#                 featurizer.add(mx.gluon.nn.Conv2D(filters[i], kernel_size=3, padding=1,
#                                                   weight_initializer=mx.init.Xavier(rnd_type='gaussian', factor_type='out', magnitude=2), bias_initializer='zeros'))
#                 if batch_norm:
#                     featurizer.add(mx.gluon.nn.BatchNorm())
#                 featurizer.add(mx.gluon.nn.Activation(ACTIVATION))
#             featurizer.add(mx.gluon.nn.MaxPool2D(strides=2))
#
#         return featurizer
#
#     net = base(prefix='VGG_')
#
#     layers, filters = vgg_spec[5]
#
#     # ND = 4096
#     ND = 512
#     with net.name_scope():
#         features = make_features(layers, filters)
#         features.add(mx.gluon.nn.Flatten())
#         features.add(mx.gluon.nn.Dense(ND, activation=ACTIVATION, weight_initializer='normal', bias_initializer='zeros', prefix='fc-1'))
#         features.add(mx.gluon.nn.Dropout(rate=0.5))
#         features.add(mx.gluon.nn.Dense(ND, activation=ACTIVATION, weight_initializer='normal', bias_initializer='zeros', prefix='fc-2'))
#         features.add(mx.gluon.nn.Dropout(rate=0.5))
#         output = mx.gluon.nn.Dense(1, activation=None, weight_initializer='normal', bias_initializer='zeros', prefix='predictions')
#         net.add(features)
#         net.add(output)
#
#     return net
#



vgg_spec = {
    # 5: ([2,2,3,3], [32, 64, 128, 256]),
    8: ([3, 2], [32, 64]),
    11: ([1, 1, 2, 2, 2], [64, 128, 256, 512, 512]),
    13: ([2, 2, 2, 2, 2], [64, 128, 256, 512, 512]),
}

# https://github.com/apache/incubator-mxnet/blob/master/python/mxnet/gluon/model_zoo/vision/vgg.py

class VGG(mx.gluon.nn.HybridBlock):

    ACTIVATION = 'relu'

    # https://gluon.mxnet.io/chapter03_deep-neural-networks/mlp-dropout-gluon.html
    def __init__(self, layers, filters, batch_norm=True, **kwargs):
        super().__init__(**kwargs)
        log.debug('VGG start')
        assert len(layers) == len(filters)
        # self.ND = 4096
        self.ND = 512
        with self.name_scope():
            self.features = self._make_features(layers, filters, batch_norm=batch_norm)
            self.features.add(mx.gluon.nn.Flatten())
            self.features.add(mx.gluon.nn.Dense(self.ND, activation=VGG.ACTIVATION, weight_initializer='normal', bias_initializer='zeros', prefix='fc-1'))
            self.features.add(mx.gluon.nn.Dropout(rate=0.5))
            self.features.add(mx.gluon.nn.Dense(self.ND, activation=VGG.ACTIVATION, weight_initializer='normal', bias_initializer='zeros', prefix='fc-2'))
            self.features.add(mx.gluon.nn.Dropout(rate=0.5))
            self.output = mx.gluon.nn.Dense(1, activation=None, weight_initializer='normal', bias_initializer='zeros',prefix='predictions')

        self.features.hybridize(static_shape=True, static_alloc=True)
        log.debug('VGG end')

    # https://discuss.mxnet.io/t/pooling-and-convolution-with-same-mode/528/3
    # http://deeplearning.net/software/theano/tutorial/conv_arithmetic.html
    def _make_features(self, layers, filters, batch_norm):
        log.debug('_make_features')

        # self.input_shape  = (None,8,11,11)
        self.kernel_size  = (3,3)
        self.strides      = (1,1)
        self.dilation     = (1,1)

        # Ensures padding = 'SAME' for ODD kernel selection
        self.padding_mode = 'same'
        p0 = self.dilation[0] * (self.kernel_size[0] - 1) // 2
        p1 = self.dilation[1] * (self.kernel_size[1] - 1) // 2
        self.padding      = (p0,p1)
        #self.padding      = (0,0)

        # padding, is_slice, out_size = keras.backend.mxnet_backend._preprocess_padding_mode(padding_mode,input_shape,kernel,strides,dilation)

        # out        =        15    + 0          -1 * (3-1) - 1
        # out_height = floor((height+2*padding[0]-dilation[0]*(kernel_size[0]-1)-1)/stride[0])+1
        # out_width  = floor((width+2*padding[1]-dilation[1]*(kernel_size[1]-1)-1)/stride[1])+1

        featurizer = mx.gluon.nn.HybridSequential(prefix='')
        for i, num in enumerate(layers):
            for _ in range(num):
                l = mx.gluon.nn.Conv2D(filters[i], kernel_size=self.kernel_size, padding=self.padding, strides=self.strides, dilation=self.dilation, weight_initializer=mx.init.Xavier(rnd_type='gaussian', factor_type='out', magnitude=2), bias_initializer='zeros')
                log.debug('l: {}'.format(l))
                featurizer.add(l)
                if batch_norm:
                    featurizer.add(mx.gluon.nn.BatchNorm())
                featurizer.add(mx.gluon.nn.Activation(VGG.ACTIVATION))
            featurizer.add(mx.gluon.nn.MaxPool2D(strides=2))

        return featurizer

    def hybrid_forward(self, F, x, *args, **kwargs):

        # https://mxnet.incubator.apache.org/api/python/gluon/nn.html#mxnet.gluon.nn.Conv2D
        # height = x.shape[2]
        # width  = x.shape[3]
        # out_height = np.floor((height+2*self.padding[0]-self.dilation[0]*(self.kernel_size[0]-1)-1)/self.strides[0])+1
        # out_width  = np.floor((width+2*self.padding[1]-self.dilation[1]*(self.kernel_size[1]-1)-1)/self.strides[1])+1
        # log.debug('hybrid_forward: x0.shape: {}; expected output shape: {}'.format(x.shape, (out_height, out_width)))


        x = self.features(x)
        # log.debug('hybrid_forward: x1.shape: {}'.format((x.shape)))
        x = self.output(x)
        # log.debug('hybrid_forward: x2.shape: {}'.format((x.shape)))
        # x = mx.nd.array([1.2], dtype=np.float32, ctx=mx.cpu())
        return x


def vgg():
    log.debug('vgg')
    layers, filters = vgg_spec[8]
    net = VGG(layers, filters)
    # net = mx.gluon.nn.Sequential()
    # with net.name_scope():
    #     net.add(VGG(layers, filters))
    # net = vgg_(base=mx.gluon.nn.HybridSequential)
    net.hybridize(static_shape=True, static_alloc=True)
    return net

class VGGModel(BaseModel):
    def __init__(self, logger, auto_save=False):
        self.transform = NCHWTransform(logger, size=11)
        super().__init__("VGG5", vgg, logger, self.transform, auto_save=auto_save)


class VGGPlusBlock(mx.gluon.nn.HybridBlock):

    ACTIVATION = 'relu'

    # https://gluon.mxnet.io/chapter03_deep-neural-networks/mlp-dropout-gluon.html
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        log.debug('VGGPlusBlock start')

        self.ND = 512

        featurizer = mx.gluon.nn.HybridSequential(prefix='')

        layers      = [4, 3]
        filters     = [32, 64]
        max_padding = [(1,1), (0,0)]

        self.dense = mx.gluon.nn.HybridSequential(prefix='dense')

        with self.name_scope():
            i = 0
            self.f1 = self._make_features(layers[i:i+1], filters[i:i+1], max_padding[i:i+1], prefix='f1')
            i += 1
            self.f2 = self._make_features(layers[i:i+1], filters[i:i+1], max_padding[i:i+1], prefix='f2')
            self.f3 = mx.gluon.nn.Flatten()

            self.dense.add(mx.gluon.nn.Dense(self.ND, activation=VGG.ACTIVATION, weight_initializer='normal', bias_initializer='zeros', prefix='fc-1'))
            self.dense.add(mx.gluon.nn.Dropout(rate=0.5))
            self.dense.add(mx.gluon.nn.Dense(self.ND, activation=VGG.ACTIVATION, weight_initializer='normal', bias_initializer='zeros', prefix='fc-2'))
            self.dense.add(mx.gluon.nn.Dropout(rate=0.5))
            self.output = mx.gluon.nn.Dense(1, activation=None, weight_initializer='normal', bias_initializer='zeros',prefix='predictions')

        log.debug('VGGPlusBlock end')

    # https://discuss.mxnet.io/t/pooling-and-convolution-with-same-mode/528/3
    # http://deeplearning.net/software/theano/tutorial/conv_arithmetic.html
    def _make_features(self, layers, filters, max_padding, batch_norm=True, prefix=''):
        log.debug('VGGPlusBlock._make_features')

        assert len(layers) == len(filters)
        assert len(layers) == len(max_padding)

        # self.input_shape  = (None,8,11,11)
        self.kernel_size  = (3,3)
        self.strides      = (1,1)
        self.dilation     = (1,1)

        # Ensures padding = 'SAME' for ODD kernel selection
        self.padding_mode = 'same'
        p0 = self.dilation[0] * (self.kernel_size[0] - 1) // 2
        p1 = self.dilation[1] * (self.kernel_size[1] - 1) // 2
        self.padding      = (p0,p1)

        featurizer = mx.gluon.nn.HybridSequential(prefix=prefix)
        with featurizer.name_scope():
            for i, num in enumerate(layers):
                for _ in range(num):
                    l = mx.gluon.nn.Conv2D(filters[i], kernel_size=self.kernel_size, padding=self.padding, strides=self.strides, dilation=self.dilation, weight_initializer=mx.init.Xavier(rnd_type='gaussian', factor_type='out', magnitude=2), bias_initializer='zeros')
                    log.debug('l: {}'.format(l))
                    featurizer.add(l)
                    if batch_norm:
                        featurizer.add(mx.gluon.nn.BatchNorm())
                    featurizer.add(mx.gluon.nn.Activation(VGG.ACTIVATION))
                featurizer.add(mx.gluon.nn.MaxPool2D(strides=2,padding=max_padding[i]))

        return featurizer

    def hybrid_forward(self, F, x1, x2, *args, **kwargs):
        x2 = self.f1(x2)
        x2 = self.f2(x2)
        x2 = self.f3(x2)

        # x = mx.nd.concat(x1, x2, dim=1)
        x = F.concat(x1, x2, dim=1)

        x = self.dense(x)
        x = self.output(x)
        return x




class VGGPlusModel(BaseModel):

    def __init__(self, logger, hybridize=True, auto_save=True, model_ctx=None, name="VGG5Plus"):
        self.transform = NCHWTransform(logger, size=11)
        # logger.debug('self.transform.base_transform.input_columns: {}'.format(self.transform.base_transform.input_columns))
        super().__init__(name, None, logger, self.transform, auto_save=auto_save)
        self.model = None # we don't use the gluon_wrapper classes

        self.features = [f for f in self.transform.base_transform.input_columns if not f.startswith('cav_')]
        self.channels = [c for c in cav.FeatureSelectionTransformationNCHW.channels if c != 'origin']


        # self.logger = logger
        # if '-' in name:
        #     raise Exception('Model names must not contain "-" characters: {}'.format(name))
        #
        # self.num_gpus  = mx.context.num_gpus()
        # self.ctx       = mx.gpu(0) if mx.context.num_gpus() > 0 else mx.cpu(0)
        # self.logger.info('number of gpus: {}, ctx: {}'.format(self.num_gpus, self.ctx))
        #
        # self.transform = transform
        # self.model     = gw.GluonRegressor(model_fn=fn, batch_size=8 * 256, model_ctx=self.ctx, epochs=2, auto_save=auto_save)
        #
        # self.name = name
        # self.path = self.create_model_save_path()
        # self.action_options = action_options

        self.loss_function = mx.gluon.loss.L2Loss()
        # self.loss_function = mx.gluon.loss.HuberLoss(rho=5)
        self.init_function = mx.init.Xavier(factor_type='in', magnitude=2*3)
        self.optimizer = mx.optimizer.Adam(learning_rate=0.005)
        self.epochs = 2
        self.num_workers = 1
        self.batch_size    = 512
        if model_ctx is None:
            self.model_ctx = mx.gpu(0) if mx.context.num_gpus() > 0 else mx.cpu(0)
        else:
            self.model_ctx = model_ctx
        log.debug('model_ctx    : {}'.format(self.model_ctx))

        self.model = VGGPlusBlock()

        self.auto_save = auto_save
        if self.auto_save:
            if hybridize == False:
                logger.warn('Setting hybridize to true, because auto_save is true!')
            hybridize = True

        if hybridize:
            self.model.hybridize(static_shape=True, static_alloc=True)

        self.model.collect_params().initialize(self.init_function, ctx=self.model_ctx)
        self.model_loaded = False

        self.trainer = mx.gluon.Trainer(self.model.collect_params(), self.optimizer)
        self.trainer_load_path = None

        self.init_progress_metric_df()

        log.debug('OS           : {}'.format(sys.platform))
        log.debug('Python       : {}'.format(sys.version))
        log.debug('MXNet        : {}'.format(mx.__version__))
        log.debug('Numpy        : {}'.format(np.__version__))
        # log.debug('GPU          : {}'.format(gw.get_gpu_name()))
        log.debug('CPU cores    : {}'.format(multiprocessing.cpu_count()))
        # log.debug(gw.get_cuda_version())
        # log.debug('CuDNN Version: {}'.format(gw.get_cudnn_version()))


    def init_progress_metric_df(self):
        self.progress_metric_df = pd.DataFrame(columns=['epoch', 'last_batch_l2loss', 'mse_train', 'mse_val'])

    def init_progress_metric_df(self):
        self.progress_metric_df = pd.DataFrame(columns=['epoch', 'last_batch_l2loss', 'mse_train', 'mse_val'])

    def to_ndarray_iter(self, train_x_nf, train_x_nchw, train_y, batch_size):
        # log.debug('using mx.io.NDArrayIter implementation')
        l = len(train_x_nf)
        if l < batch_size:
            batch_size = l

        data = dict(data0=train_x_nf, data1=train_x_nchw)
        label      = None
        label_name = None
        if train_y is not None:
            label      =  dict(lin_reg_label=train_y)
            label_name = 'lin_reg_label'

        itr = mx.io.NDArrayIter(data, label, batch_size, shuffle=False, label_name=label_name, last_batch_handle='pad')
        # itr = gw.DataIterLoader(itr)
        return itr

    # nf   = N x F: F == features
    # nchw = N x C x H x W : C = channels, H = height, W =  width
    def fit_(self, train_x_nf, train_x_nchw, train_y, eval_on_train=False, batch_size=256, epochs=2, verbose=1, validation_split=0.1, model_save_path=None):
        # self.logger.debug('train_x_nf.shape: {}, train_x_nchw.shape: {}'.format(train_x_nf.shape, train_x_nchw.shape))

        if self.auto_save and model_save_path is None:
            model_save_path = self.create_model_save_path()
            self.path = model_save_path
        self.logger.info('model_save_path: {}'.format(model_save_path))

        self.batch_size       = batch_size
        self.epochs           = epochs
        self.verbose          = verbose
        self.validation_split = validation_split


        # hampers dropout!
        # seed = 43
        # mx.random.seed(seed)
        # np.random.seed(seed)
        self.init_progress_metric_df()

        n =  train_x_nf.shape[0]

        if validation_split is not None:

            if n * validation_split > 10000:
                validation_split = 10000.0 / n

            train_y_dummy = np.arange(0,n)
            train_x_dummy = train_y_dummy.reshape(-1,1)
            _, _, y_train_idx, y_test_idx = sklearn.model_selection.train_test_split(train_x_dummy, train_y_dummy, shuffle=True, test_size = validation_split) # , random_state = seed

            X_test_nf   = train_x_nf[y_test_idx,:]
            X_test_nchw = train_x_nchw[y_test_idx,:,:,:]
            y_test      = train_y[y_test_idx]

        else:
            y_train_idx = np.random.permutation(np.arange(0,n))

        X_train_nf   = train_x_nf[y_train_idx,:]
        X_train_nchw = train_x_nchw[y_train_idx,:,:,:]
        y_train      = train_y[y_train_idx]


        loss_function = self.loss_function


        train_iter = self.to_ndarray_iter(X_train_nf, X_train_nchw, y_train, batch_size)

        # self.trainer = mx.gluon.Trainer(self.model.collect_params(), self.optimizer)

        nr_batches = len(X_train_nf) // self.batch_size
        total = self.epochs * (nr_batches + 1)

        with tqdm.tqdm(total=total) as pbar:
            for e in range(self.epochs):
                batch_loss      = []
                last_batch_loss = None
                train_iter.reset()
                for i, db in enumerate(train_iter):
                    pbar.update(1)
                    # if e == 0 and i == 1 and self.trainer_load_path is not None:
                    #     self.logger.debug('loading trainer states: {}'.format(self.trainer_load_path))
                    #     if os.path.isfile(self.trainer_load_path):
                    #         self.trainer.load_states(self.trainer_load_path)
                    #     self.trainer_load_path = None

                    x_nf_   = db.data[0]
                    x_nchw_ = db.data[1]
                    y_      = db.label[0]

                    x_nf = x_nf_.as_in_context(self.model_ctx)
                    x_nchw = x_nchw_.as_in_context(self.model_ctx)
                    y = y_.as_in_context(self.model_ctx)
                    if self.num_workers > 1:
                        mx.nd.waitall()
                    with mx.autograd.record():
                        output = self.model(x_nf, x_nchw)
                        loss = loss_function(output, y)

                    loss.backward()
                    batch_loss     += [loss]
                    last_batch_loss = mx.nd.mean(loss).asscalar()
                    self.trainer.step(x_nf.shape[0])

                if self.num_workers > 1:
                    mx.nd.waitall()
                if eval_on_train:
                    s_train = self.score_(X_train_nf, X_train_nchw, y_train)
                else:
                    s_train = np.concatenate([a.asnumpy() for a in batch_loss])
                    s_train = np.mean(s_train)

                if validation_split is not None:
                    s_val = self.score_(X_test_nf, X_test_nchw, y_test)
                else:
                    s_val = 0.0
                self.progress_metric_df.loc[len(self.progress_metric_df)] = [e, last_batch_loss, s_train, s_val]

        if self.auto_save and model_save_path is not None:
            self.save(model_save_path)
        return self

    def predict_(self, x_nf, x_nchw, **kwargs):
        # self.logger.debug('X_nf.shape: {}, X_nchw.shape: {}'.format(x_nf.shape, x_nchw.shape))

        itr = self.to_ndarray_iter(x_nf, x_nchw, None, self.batch_size)

        y_pred  = mx.nd.zeros(x_nf.shape[0])
        for i, db in enumerate(itr):
            l = np.asscalar(np.min([self.batch_size, x_nf.shape[0] - i *self.batch_size]))
            x_nf_ = db.data[0].as_in_context(self.model_ctx)
            x_nchw_ = db.data[1].as_in_context(self.model_ctx)

            output = self.model(x_nf_, x_nchw_)
            y_pred[i * self.batch_size : i * self.batch_size + l] = output[:l,0]

        r = y_pred.asnumpy()
        return r

    def score_(self, x_nf, x_nchw, y):
        y_pred = self.predict_(x_nf, x_nchw)
        s = sklearn.metrics.mean_squared_error(y, y_pred)
        return s

    def fit(self, X_train, y_train, eval_on_train=False, batch_size=256, epochs=2, verbose=1, validation_split=0.1, model_save_path=None):
        if not isinstance(X_train, pd.DataFrame):
            raise Exception('The input X_train must be a dataframe')

        if not np.any(X_train.columns == 'QQ'):
            X_train = X_train.copy()
            X_train.loc[:,'QQ'] = y_train

        # self.logger.debug('self.transform.base_transform.input_columns: {}'.format(self.transform.base_transform.input_columns))
        input_columns = ['QQ'] + list(self.transform.base_transform.input_columns)
        X_train = X_train[input_columns]
        t = cav.FeatureSelectionTransformationNCHW(X_train)
        rxds = t.transform()

        return self.fit_xds(rxds, eval_on_train=eval_on_train, batch_size=batch_size, epochs=epochs, verbose=verbose, validation_split=validation_split, model_save_path=model_save_path)

    def fit_xds(self, xds, eval_on_train=False, batch_size=256, epochs=2, verbose=1, validation_split=0.1, model_save_path=None, gamma=None):
        rxds = xds
        if gamma is None:
            gamma = cav.None_FT0.discount_rate

        X_train_nf, X_train_nchw = self.xds_to_nf_and_nchw(rxds)
        # y_train = rxds['base'].loc[dict(base_fields='QQ')].astype(np.float64).values
        q = rxds['base'].loc[dict(base_fields='Q')].astype(np.float64).values
        qqnsm = rxds['base'].loc[dict(base_fields='QQNextStateMax')].astype(np.float64).values
        y_train = q + gamma * qqnsm

        return self.fit_(X_train_nf, X_train_nchw, y_train, eval_on_train=eval_on_train, batch_size=batch_size, epochs=epochs, verbose=verbose, validation_split=validation_split, model_save_path=model_save_path)

    def fit_file(self, id, augment_with_penalty_moves=False, eval_on_train=False, batch_size=256, epochs=2, verbose=1, validation_split=0.1, model_save_path=None):
        rxds = self.transform.batch_transform(id, augment_with_penalty_moves=augment_with_penalty_moves)
        return self.fit_xds(rxds, eval_on_train=eval_on_train, batch_size=batch_size, epochs=epochs, verbose=verbose, validation_split=validation_split, model_save_path=model_save_path)

    def predict_xds(self, xds):
        rxds = xds

        X_nf, X_nchw = self.xds_to_nf_and_nchw(rxds)
        return self.predict_(X_nf, X_nchw)

    def predict(self, X):
        # self.logger.debug('self.transform.base_transform.input_columns: {}'.format(self.transform.base_transform.input_columns))
        if type(X).__module__ == np.__name__:
            X = pd.DataFrame(X, columns=self.transform.base_transform.input_columns)

        if not isinstance(X, pd.DataFrame):
            raise Exception('The input X_train must be a dataframe')

        X = X[self.transform.base_transform.input_columns]
        t = cav.FeatureSelectionTransformationNCHW(X)
        rxds = t.transform()

        X_nf, X_nchw = self.xds_to_nf_and_nchw(rxds)

        return self.predict_(X_nf, X_nchw)

    def xds_to_nf_and_nchw(self, rxds):
        X_nf = rxds['base'].loc[dict(base_fields=self.features)].astype(np.float32).values
        X_nchw = rxds['cav'].loc[dict(channel=self.channels)].values
        return X_nf, X_nchw

    def save(self, file_name):
        self.model.export(file_name)
        if self.model_loaded:
            self.trainer.save_states(file_name + '-trainer.states')


    def disable_matplotlib_loggers(self):
        d = logging.Logger.manager.loggerDict
        keys = d.keys()
        self.active_matplotlib_loggers        = [key for key in keys if key.startswith('matplotlib')]
        self.active_matplotlib_loggers_status = [logging.getLogger(key).disabled for key in self.active_matplotlib_loggers]
        for key in self.active_matplotlib_loggers:
            logging.getLogger(key).disabled = True

    def plot(self):
        self.disable_matplotlib_loggers()
        ldf = self.progress_metric_df

        plt.figure(figsize=(15, 8), dpi=80, facecolor='w', edgecolor='k')
        ax = plt.subplot(1, 1, 1)
        ax.plot(ldf['mse_train'].values)
        ax.plot(ldf['mse_val'].values)
        ax.set_ylabel('loss')
        ax.set_xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')

        # return fig

    def load(self, id=None, suffix=None, latest=True):
        model_load_path_ = get_model_load_path(self.name, selected_id=id, latest=latest)
        model_load_path = '{}-symbol.json'.format(model_load_path_)
        # model_params_path = '{}-0000.params'.format(model_load_path_)

        self.logger.info("Trying to load model: {}".format(model_load_path))

        if model_load_path and os.path.isfile(model_load_path):
            self.load_(model_load_path) # , model_params_path
            self.model_loaded = True
            self.logger.info("Model loaded: {}".format(model_load_path))
            self.trainer_load_path = model_load_path_ + '-trainer.states'
            if os.path.isfile(self.trainer_load_path):
                self.trainer.load_states(self.trainer_load_path)
                self.logger.info('Trainer states loaded: {}'.format(self.trainer_load_path))
        else:
            self.logger.info("No model is found")

    def load_(self, model_load_path, model_params_path=None):

        if model_params_path is None:
            r = re.search(r'^(.*/\d+-.*?-.*?)-symbol\.json$', model_load_path)
            if r:
                file_base_name = r.group(1)
            else:
                raise Exception('The glob does not match the pattern: {}'.format(model_load_path))
            model_params_path = '{}-0000.params'.format(file_base_name)

        self.model = mx.gluon.SymbolBlock.imports(model_load_path, ['data0','data1'], model_params_path, self.model_ctx)
        self.trainer = mx.gluon.Trainer(self.model.collect_params(), self.optimizer)

# https://mlexplained.com/2018/01/10/an-intuitive-explanation-of-why-batch-normalization-really-works-normalization-in-deep-learning-part-1/
# https://mlexplained.com/2018/01/13/weight-normalization-and-layer-normalization-explained-normalization-in-deep-learning-part-2/

class VGG20190317Block(mx.gluon.nn.HybridBlock):

    ACTIVATION = 'relu'

    # https://gluon.mxnet.io/chapter03_deep-neural-networks/mlp-dropout-gluon.html
    def __init__(self, logger=None, **kwargs):
        super().__init__(**kwargs)
        self.logger = logger
        if self.logger is None:
            self.logger = log
        self.logger.debug('VGG20190317Block start')

        self.ND = 512

        layers      = [3, 3, 2]
        filters     = [64, 128, 256]
        max_padding = [(1,1), (0,0), (1,1)]

        self.dense = mx.gluon.nn.HybridSequential(prefix='dense')

        with self.name_scope():
            i = 0
            self.f1 = self._make_features(layers[i:i+1], filters[i:i+1], max_padding[i:i+1], prefix='f1', kernel_size=(5,5)) # , ,,in_channels=7
            i += 1
            self.f2 = self._make_features(layers[i:i+1], filters[i:i+1], max_padding[i:i+1], prefix='f2')
            i += 1
            self.f3 = self._make_features(layers[i:i+1], filters[i:i+1], max_padding[i:i+1], prefix='f3')
            self.f4 = mx.gluon.nn.Flatten()

            self.dense.add(mx.gluon.nn.Dense(self.ND, weight_initializer='normal', bias_initializer='zeros', prefix='fc-1')) # activation=VGG20190317Block.ACTIVATION,
            self.dense.add(mx.gluon.nn.BatchNorm())
            self.dense.add(mx.gluon.nn.ELU())
            self.dense.add(mx.gluon.nn.Dense(self.ND, weight_initializer='normal', bias_initializer='zeros', prefix='fc-2')) # activation=VGG20190317Block.ACTIVATION,
            self.dense.add(mx.gluon.nn.BatchNorm())
            self.dense.add(mx.gluon.nn.ELU())
            self.dense.add(mx.gluon.nn.Dense(self.ND, weight_initializer='normal', bias_initializer='zeros', prefix='fc-3')) # activation=VGG20190317Block.ACTIVATION,
            self.dense.add(mx.gluon.nn.BatchNorm())
            self.dense.add(mx.gluon.nn.ELU())
            self.output = mx.gluon.nn.Dense(1, activation=None, weight_initializer='normal', bias_initializer='zeros',prefix='predictions')

        self.logger.debug('VGG20190317Block end')

    # https://discuss.mxnet.io/t/pooling-and-convolution-with-same-mode/528/3
    # http://deeplearning.net/software/theano/tutorial/conv_arithmetic.html
    def _make_features(self, layers, filters, max_padding, batch_norm=True, prefix='', kernel_size=(3,3)):
        self.logger.debug('VGG20190317Block._make_features')

        assert len(layers) == len(filters)
        assert len(layers) == len(max_padding)

        # self.input_shape  = (None,8,11,11)
        self.kernel_size  = kernel_size
        self.strides      = (1,1)
        self.dilation     = (1,1)

        # Ensures padding = 'SAME' for ODD kernel selection
        self.padding_mode = 'same'
        p0 = self.dilation[0] * (self.kernel_size[0] - 1) // 2
        p1 = self.dilation[1] * (self.kernel_size[1] - 1) // 2
        self.padding      = (p0,p1)

        # mx.init.Xavier(factor_type='in', magnitude=2*3)

        featurizer = mx.gluon.nn.HybridSequential(prefix=prefix)
        with featurizer.name_scope():
            for i, num in enumerate(layers):
                for lnr in range(num):
                    l = mx.gluon.nn.Conv2D(filters[i], kernel_size=self.kernel_size, padding=self.padding, strides=self.strides, dilation=self.dilation,
                                           weight_initializer=mx.init.Xavier(rnd_type='gaussian', factor_type='out', magnitude=2), bias_initializer='zeros' )  # , activation='relu'
                    self.logger.debug('l: {}'.format(l))
                    featurizer.add(l)
                    if batch_norm:
                        featurizer.add(mx.gluon.nn.BatchNorm())
                    # featurizer.add(mx.gluon.nn.Activation(VGG20190317Block.ACTIVATION))
                    featurizer.add(mx.gluon.nn.ELU())
                featurizer.add(mx.gluon.nn.MaxPool2D(strides=2,padding=max_padding[i]))

        return featurizer

    def hybrid_forward(self, F, x1, x2, *args, **kwargs):
        x2 = self.f1(x2)
        x2 = self.f2(x2)
        x2 = self.f3(x2)
        x2 = self.f4(x2)

        # x = mx.nd.concat(x1, x2, dim=1)
        x = F.concat(x1, x2, dim=1)

        x = self.dense(x)
        x = self.output(x)
        return x


class VGG20190317Model(VGGPlusModel):

    def __init__(self, logger, hybridize=True, auto_save=True, model_ctx=None):
        super().__init__(logger, hybridize=hybridize, auto_save=auto_save, model_ctx=model_ctx, name="VGG20190317")
        self.model = VGG20190317Block(logger=logger)

        if self.auto_save:
            if hybridize == False:
                logger.warn('Setting hybridize to true, because auto_save is true!')
            hybridize = True

        if hybridize:
            self.model.hybridize(static_shape=True, static_alloc=True)

        self.model.collect_params().initialize(self.init_function, ctx=self.model_ctx)
        self.model_loaded = False

        self.trainer = mx.gluon.Trainer(self.model.collect_params(), self.optimizer)
        self.trainer_load_path = None
