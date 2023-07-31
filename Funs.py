
import numpy as np
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt

def dice_coef(y_true, y_pred, smooth=1):
    import tensorflow as tf
    import keras.backend as K
    y_true = tf.cast(y_true, dtype='float32')
    y_pred = tf.cast(y_pred, dtype='float32')
    """
    Dice = (2*|X & Y|)/ (|X|+ |Y|)
         =  2*sum(|A*B|)/(sum(A^2)+sum(B^2))
    ref: https://arxiv.org/pdf/1606.04797v1.pdf
    """
    intersection = K.sum(y_true * y_pred)
    return (2. * intersection + smooth) / (K.sum(y_true) + K.sum(y_pred) + smooth)

def dice_coef_loss(y_true, y_pred):
    return 1-dice_coef(y_true, y_pred)

def two_loss_func(y_true, y_pred):
    from tensorflow.keras.losses import binary_crossentropy
    return 0.5*binary_crossentropy(y_true, y_pred)+(1-0.5)*dice_coef_loss(y_true, y_pred)

def calibration_plot(test_y, test_y_pred, title_plot, name_file):
    from sklearn.calibration import calibration_curve
    import matplotlib.lines as mlines
    plot_y, plot_x = calibration_curve(test_y, test_y_pred, n_bins = 20)

    fig, ax = plt.subplots(1, 1, figsize = (6,5))
    ax.hist(test_y_pred[test_y == 0], weights = np.ones_like(test_y_pred[test_y == 0])/len(test_y[test_y == 0]), alpha = 0.4,  label = "Negative", bins = 50)
    ax.hist(test_y_pred[test_y == 1], weights=np.ones_like(test_y_pred[test_y == 1]) / len(test_y[test_y == 1]),
            alpha=0.4, label="Positive", bins = 50)
    ax.plot(plot_x, plot_y, marker = 'o', linewidth = 1, label = "Model")
    line = mlines.Line2D([0,1], [0,1], color = "blue", label = "Ideally calibrated")
    transform = ax.transAxes
    line.set_transform(transform)
    ax.add_line(line)
    fig.suptitle(title_plot)
    ax.set_xlabel("Predicted positive probability")
    ax.set_ylabel("Observed positive proportion")
    plt.legend(loc = "upper left")
    plt.show()
    fig.savefig(name_file + ".pdf", dpi=720)

# FROM https://towardsdatascience.com/get-confidence-intervals-for-any-model-performance-metrics-in-machine-learning-f9e72a3becb2
def one_boot(*data_args):
    """Usage: (t, p) = one_boot(true, pred) with true, pred, t, p arrays of same length
    """
    length = len(data_args[0])
    index = np.random.randint(0, length, size=length)  # apply same sampled index to all args:
    return [ pd.Series(arg.values[index], name=arg.name)  # .reset_index() is slower
             if isinstance(arg, pd.Series) else arg[index]   for  arg in data_args
           ]
import re
def calc_metrics(metrics, *data_args):
    """Return a list of calculated values for each metric applied to *data_args
    where metrics is a metric func or iterable of funcs e.g. [m1, m2, m3, m4]
    """
    metrics=_fix_metrics(metrics)
    mname = metrics.__name__ if hasattr(metrics, '__name__') else "Metric"
    return pd.Series\
     ([m(*data_args) for m in metrics], index=[_metric_name(m) for m in metrics], name=mname)
def _metric_name(metric):  # use its prettified __name__
    name = re.sub(' score$', '', metric.__name__.replace('_',' ').strip())
    return name.title() if name.lower()==name else name
def _fix_metrics(metrics_): # allow for single metric func or any iterable of metric funcs
    if callable(metrics_): metrics_=[metrics_]  # single metric func to list of one
    return pd.Series(metrics_)  # in case iterable metrics_ is generator, generate & store

def raw_metric_samples(metrics, *data_args, nboots):
    """Return dataframe containing metric(s) for nboots boot sample datasets
    where metrics is a metric func or iterable of funcs e.g. [m1, m2, m3]
    """
    metrics=_fix_metrics(metrics)
    cols=[calc_metrics(metrics, *boot_data)   for boot_data  in _boot_generator\
           (*data_args, nboots=nboots)  if len(np.unique(boot_data[0])) >1  # >1 for log Loss, ROC
         ]#end of list comprehension
    return pd.DataFrame\
      ( {iboot: col for iboot,col in enumerate(cols)}#end of dict comprehension
      ).rename_axis("Boot", axis="columns").rename_axis(cols[0].name)
def _boot_generator(*data_args, nboots): #return Gener of boot sampl datasets, not huge list!
    return (one_boot(*data_args) for _ in range(nboots)) # generator expression

def ci_auto( metrics, *data_args, alpha=0.05, nboots=None, Sample, HistoryYears, Threshold, Output, PredictionWindow):
    """Return Pandas data frame of bootstrap confidence intervals.
    PARAMETERS:
    metrics : a metric func or iterable of funcs e.g. [m1, m2, m3]
    data_args : 1+ (often 2, e.g. ytrue,ypred) iterables for metric
    alpha: = 1 - confidence level; default=0.05 i.e. confidence=0.95
    nboots (optional!): # boots drawn from data; dflt None ==> calc. from alpha
    """
    #alpha, nboots = _get_alpha_nboots(alpha, nboots)
    metrics=_fix_metrics(metrics)
    result=raw_metric_samples(metrics, *data_args, nboots=nboots)
    nb=result.shape[1]  # num boots we ended up with
    if nb<nboots:
        t = f'Note: {nboots-nb} boot sample datasets dropped\n'
        print(t + f'(out of {nboots}) because all vals were same in 1st data arg.')
    result = result.apply(lambda row: row.quantile([0.5*alpha, 1 - 0.5*alpha]), axis=1)
    result.columns = [f'{x*100:.4g}%ile' for x in (0.5*alpha, 1 - 0.5*alpha)]
    result.insert(0, "Observed", calc_metrics(metrics, *data_args)) #col for obs (point estim)
    result = result.rename_axis(f"%ile for {nb} Boots", axis="columns")
    result["Sample"] = Sample
    result["HistoryYears"] = HistoryYears
    result["Threshold"] = Threshold
    result["Output"] = Output
    result["Time"] = PredictionWindow
    return result

