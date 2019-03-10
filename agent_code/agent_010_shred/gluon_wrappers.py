"""scikit-learn like wrapper for gluon in mxnet"""
import mxnet as mx
from mxnet import gluon, nd, autograd, metric
import sklearn, sklearn.pipeline, sklearn.model_selection, sklearn.preprocessing
from sklearn.base import BaseEstimator
import numpy as np, pandas as pd, matplotlib.pyplot as plt
import tqdm, logging, re
import sys,os,subprocess,glob,multiprocessing

def get_gpu_name():
    try:
        out_str = subprocess.run(["nvidia-smi", "--query-gpu=gpu_name", "--format=csv"], stdout=subprocess.PIPE).stdout
        out_list = out_str.decode("utf-8").split('\n')
        out_list = out_list[1:-1]
        return out_list
    except Exception as e:
        print(e)


def get_cuda_version():
    """Get CUDA version"""
    if sys.platform == 'win32':
        raise NotImplementedError("Implement this!")
        # This breaks on linux:
        #cuda=!ls "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA"
        #path = "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\" + str(cuda[0]) +"\\version.txt"
    elif sys.platform == 'linux' or sys.platform == 'darwin':
        path = '/usr/local/cuda/version.txt'
    else:
        raise ValueError("Not in Windows, Linux or Mac")
    if os.path.isfile(path):
        with open(path, 'r') as f:
            data = f.read().replace('\n','')
        return data
    else:
        return "No CUDA in this machine"

def get_cudnn_version():
    """Get CUDNN version"""
    if sys.platform == 'win32':
        raise NotImplementedError("Implement this!")
        # This breaks on linux:
        #cuda=!ls "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA"
        #candidates = ["C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\" + str(cuda[0]) +"\\include\\cudnn.h"]
    elif sys.platform == 'linux':
        candidates = ['/usr/include/x86_64-linux-gnu/cudnn_v[0-99].h',
                      '/usr/local/cuda/include/cudnn.h',
                      '/usr/include/cudnn.h']
    elif sys.platform == 'darwin':
        candidates = ['/usr/local/cuda/include/cudnn.h',
                      '/usr/include/cudnn.h']
    else:
        raise ValueError("Not in Windows, Linux or Mac")
    for c in candidates:
        file = glob.glob(c)
        if file: break
    if file:
        with open(file[0], 'r') as f:
            version = ''
            for line in f:
                if "#define CUDNN_MAJOR" in line:
                    version = line.split()[-1]
                if "#define CUDNN_MINOR" in line:
                    version += '.' + line.split()[-1]
                if "#define CUDNN_PATCHLEVEL" in line:
                    version += '.' + line.split()[-1]
        if version:
            return version
        else:
            return "Cannot find CUDNN version"
    else:
        return "No CUDNN in this machine"

log = logging.getLogger('gluon_wrappers')

class DataIterLoader():
    def __init__(self, data_iter):
        self.data_iter = data_iter

    def __iter__(self):
        self.data_iter.reset()
        return self

    def __next__(self):
        batch = self.data_iter.__next__()
        # print('batch.label: {}'.format(batch.label))
        if batch.label is not None and len(batch.label) > 0:
            assert len(batch.data) == len(batch.label) == 1
            label = batch.label[0]
        else:
            label = None
        data = batch.data[0]
        return data, label

    def next(self):
        return self.__next__() # for Python 2


def to_gluon_iter(x_in, y_in, batch_size=256, workers=1): # (multiprocessing.cpu_count()//2)
    if workers <= 1:
        log.debug('using mx.io.NDArrayIter implementation')
        itr = mx.io.NDArrayIter(x_in, y_in, batch_size, shuffle=None, label_name='lin_reg_label')
        itr = DataIterLoader(itr)
    else:
        log.debug('mx.gluon.data.DataLoader implementation with {} workers'.format(workers))
        x_nd = nd.array(x_in)
        y_nd = nd.array(y_in)
        dataset = mx.gluon.data.ArrayDataset(x_nd, y_nd)

        itr = mx.gluon.data.DataLoader(dataset, batch_size = batch_size, shuffle = None, num_workers=workers)# , last_batch = 'rollover'
    return itr


class GluonRegressor(BaseEstimator):

    def __init__(self, model_fn, loss_function=mx.gluon.loss.L2Loss(), init_function=mx.init.Xavier(), batch_size=512, model_ctx=mx.cpu(), epochs=2, optimizer=mx.optimizer.Adam(), num_workers=1, auto_save=True):
        self.batch_size    = batch_size
        self.model_ctx     = model_ctx
        self.epochs        = epochs
        self.model_fn      = model_fn
        self.model         = model_fn()
        self.optimizer     = optimizer
        self.loss_function = loss_function
        self.init_function = init_function
        self.num_workers   = num_workers
        self.auto_save     = auto_save

        self.model.collect_params().initialize(self.init_function, ctx=self.model_ctx)
        self.init_progress_metric_df()

        log.debug('OS           : {}'.format(sys.platform))
        log.debug('Python       : {}'.format(sys.version))
        log.debug('MXNet        : {}'.format(mx.__version__))
        log.debug('Numpy        : {}'.format(np.__version__))
        log.debug('GPU          : {}'.format(get_gpu_name()))
        log.debug('CPU cores    : {}'.format(multiprocessing.cpu_count()))
        log.debug(get_cuda_version())
        log.debug('CuDNN Version: {}'.format(get_cudnn_version()))

    def init_progress_metric_df(self):
        self.progress_metric_df = pd.DataFrame(columns=['epoch', 'last_batch_l2loss', 'mse_train', 'mse_val'])

    def to_train_iter(self, train_x, train_y, **kwargs):
        train_iter = to_gluon_iter(train_x, train_y, batch_size=self.batch_size, workers=self.num_workers)
        return train_iter

    def fit(self, train_x, train_y, eval_on_train=False, batch_size=256, epochs=2, verbose=1, validation_split=0.1, model_save_path=None):
        self.batch_size       = batch_size
        self.epochs           = epochs
        self.verbose          = verbose
        self.validation_split = validation_split
        seed = 43
        mx.random.seed(seed)
        np.random.seed(seed)
        self.init_progress_metric_df()

        if isinstance(train_x, pd.DataFrame):
            train_x = train_x.values

        if isinstance(train_y, pd.DataFrame):
            train_y = train_y.values.reshape(-1)
        elif isinstance(train_y, pd.Series):
            train_y = train_y.values


        X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(train_x, train_y, test_size = validation_split, random_state = seed)

        loss_function = self.loss_function

        # trainer = gluon.Trainer(self.model.collect_params(), self.optimizer, {'learning_rate': self.learning_rate, **kwargs})
        trainer = gluon.Trainer(self.model.collect_params(), self.optimizer)
        train_iter = self.to_train_iter(X_train, y_train)

        current_loss = np.nan

        nr_batches = len(X_train) // self.batch_size
        total = self.epochs * (nr_batches + 1)
        # if nr_batches * self.batch_size == len(train_x):
        #     total = self.epochs * nr_batches - 1
        # else:
        #     total = self.epochs * (nr_batches + 1) - 1
        # print(total, nr_batches * self.batch_size, nr_batches, self.batch_size)
        with tqdm.tqdm(total=total) as pbar:
            for e in range(self.epochs):
                batch_loss      = []
                last_batch_loss = None
                for i, (x_, y_) in enumerate(train_iter):
                    pbar.update(1)
                    # log.debug('x_.shape: {}, x_.dtype: {}'.format(x_.shape, x_.dtype))
                    # log.debug('y_.shape: {}, y_.dtype: {}'.format(y_.shape, y_.dtype))
                    x = x_.as_in_context(self.model_ctx)
                    y = y_.as_in_context(self.model_ctx)
                    # log.debug('x.shape: {}, x.dtype: {}'.format(x.shape, x.dtype))
                    # log.debug('y.shape: {}, y.dtype: {}'.format(y.shape, y.dtype))
                    if self.num_workers > 1:
                        nd.waitall()
                    with autograd.record():
                        output = self.model(x)
                        # log.debug('output.shape: {}, output.dtype: {}'.format(output.shape, output.dtype))
                        loss = loss_function(output, y)

                    loss.backward()
                    batch_loss     += [loss]
                    last_batch_loss = nd.mean(loss).asscalar()
                    trainer.step(x.shape[0])

                if self.num_workers > 1:
                    nd.waitall()
                # y_pred = self.model(self.train_x).asnumpy()
                # mse = sklearn.metrics.mean_squared_error(train_y, y_pred)
                if eval_on_train:
                    s_train = self.score(X_train, y_train)
                else:
                    s_train = np.concatenate([a.asnumpy() for a in batch_loss])
                    s_train = np.mean(s_train)
                s_val   = self.score(X_test , y_test)
                self.progress_metric_df.loc[len(self.progress_metric_df)] = [e, last_batch_loss, s_train, s_val]
                if self.auto_save and model_save_path is not None:
                    self.save(model_save_path)

        return self

    def predict(self, x, **kwargs):
        dataset = mx.gluon.data.ArrayDataset(nd.array(x))
        iter    = mx.gluon.data.DataLoader(dataset, batch_size = self.batch_size)
        y_pred  = nd.zeros(x.shape[0])
        for i, (data) in enumerate(iter):
            data   = data.as_in_context(self.model_ctx)
            output = self.model(data)
            y_pred[i * self.batch_size : i * self.batch_size + output.shape[0]] = output[:,0]
        return y_pred.asnumpy()

    def save(self, file_name):
        self.model.export(file_name)

    def load(self, model_load_path, model_params_path=None):

        if model_params_path is None:
            r = re.search(r'^(.*/\d+-.*?-.*?)-symbol\.json$', model_load_path)
            if r:
                file_base_name = r.group(1)
            else:
                raise Exception('The glob does not match the pattern: {}'.format(model_load_path))
            model_params_path = '{}-0000.params'.format(file_base_name)

        self.model = mx.gluon.SymbolBlock.imports(model_load_path, ['data'], model_params_path, self.model_ctx)

    def score(self, x, y):
        y_pred = self.predict(x)
        s = sklearn.metrics.mean_squared_error(y, y_pred)
        return s

    # https://www.electricmonk.nl/log/2017/08/06/understanding-pythons-logging-module/
    # https://stackoverflow.com/questions/12158048/changing-loggings-basicconfig-which-is-already-set
    # https://github.com/ansible/ansible/issues/47347

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

        fig = plt.figure(figsize=(15, 8), dpi=80, facecolor='w', edgecolor='k')
        ax = plt.subplot(1, 1, 1)
        ax.plot(ldf['mse_train'].values)
        ax.plot(ldf['mse_val'].values)
        ax.set_ylabel('loss')
        ax.set_xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')

        # return fig

