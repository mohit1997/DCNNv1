import argparse
import os
import random
import warnings
from glob import glob

import numpy as np
import torch as th
import torch.nn.functional as F
from utils import strided_app
from torch_utils import to_variable
import matplotlib.pyplot as plt
from cluster import cluster
from metrics import corrected_naylor_metrics
from torch.autograd import Variable

from savemodel import Saver
from stacked_pymodel_middle import MyNet as SELUNet
# INFO: Set random seeds
np.random.seed(42)
th.manual_seed(42)
th.cuda.manual_seed_all(42)
random.seed(42)


result = 'test_idr_aplawdtoslt.txt'

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--speechfolder',
        type=str,
        default='slt_speech_raw',
        help='data directory containing speech files')

    parser.add_argument(
        '--peaksfolder',
        type=str,
        default='slt_peaks',
        help='data directory containing peak files')
    parser.add_argument(
        '--window',
        type=int,
        default=200,
        help='window size for the overlapping sub arrays')
    parser.add_argument(
        '--stride', type=int, default=1, help='stride of the moving window')
    parser.add_argument(
        '-n',
        '--model_name',
        type=str,
        default='bce_epoch_15.pt',
        help='checkpoint file containing the model to use for prediction')
    parser.add_argument(
        '--model_dir',
        default='checkpoints/multispeaker_aplawd_200',
        type=str,
        help='Directory containing checkpoint files')
    parser.add_argument(
        '--use_cuda', type=bool, default=True, help='use gpu for inference')
    parser.add_argument(
        '--threshold',
        type=float,
        default=0,
        help='threshold for discerning peaks')
    args = parser.parse_args()
    return args


def create_dataset(speechfolder,
                   peaksfolder,
                   window,
                   stride,
                   file_slice=slice(0, 10)):
    speechfiles = sorted(glob(os.path.join(speechfolder, '*.npy')))[file_slice]
    peakfiles = sorted(glob(os.path.join(peaksfolder, '*.npy')))[file_slice]

    # speechfiles = speechfiles[:10]
    # peakfiles = peakfiles[:10]
    speech_data = [np.load(f) for f in speechfiles]
    peak_data = [np.load(f) for f in peakfiles]

    speech_data = np.concatenate(speech_data)
    peak_data = np.concatenate(peak_data)
    indices = np.arange(len(speech_data))

    speech_windowed_data = strided_app(speech_data, window, stride)
    peak_windowed_data = strided_app(peak_data, window, stride)
    indices = strided_app(indices, window, stride)
    return speech_windowed_data, indices, peak_data


def main():
    args = parse_args()

    saver = Saver(args.model_dir)
    model = SELUNet()
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        if args.use_cuda:
            model = model.cuda()
        model, _, params_dict = saver.load_checkpoint(
            model, file_name=args.model_name)

    model.eval()
    filespan = 10
    numfiles = len(glob(os.path.join(args.speechfolder, '*.npy')))
    # numfiles = 20
    print('Models and Files Loaded')

    metrics_list = []

    for i in range(0, numfiles, filespan):
        if (i + filespan) > numfiles:
            break
        speech_windowed_data, indices, actual_gci_locations = create_dataset(
            args.speechfolder, args.peaksfolder, args.window, args.stride,
            slice(i, i + filespan))

        # input = to_variable(
        #     th.from_numpy(
        #         np.expand_dims(speech_windowed_data, 1).astype(np.float32)),
        #     args.use_cuda, True)

        input = Variable(th.from_numpy(
                 np.expand_dims(speech_windowed_data, 1).astype(np.float32))).cuda()

        with warnings.catch_warnings():
            with th.no_grad():
                prediction = model(input)
                prediction = prediction.cpu()


        predicted_peak_indicator = F.sigmoid(prediction[:, 0]).data.numpy()
        predicted_peak_distance = (prediction[:, 1]).data.numpy().astype(
            np.int32)


        del input
        del prediction
        # tem = (predicted_peak_indicator > 0.8) * 1.0
        # print(np.sum(tem))
        # print(predicted_peak_distance)

        predicted_peak_indicator_indices = predicted_peak_indicator > args.threshold

        predicted_peak_indicator = predicted_peak_indicator[
            predicted_peak_indicator_indices].ravel()
        predicted_peak_distance = predicted_peak_distance[
            predicted_peak_indicator_indices].ravel()
        indices = indices[predicted_peak_indicator_indices]

        assert (len(indices) == len(predicted_peak_distance))
        assert (len(predicted_peak_distance) == len(predicted_peak_indicator))

        positive_distance_indices = predicted_peak_distance < args.window

        positive_peak_distances = predicted_peak_distance[
            positive_distance_indices]
        postive_predicted_peak_indicator = predicted_peak_indicator[
            positive_distance_indices]

        # print(indices.shape, indices)


        gci_locations = [
            indices[i, d] for i, d in enumerate(positive_peak_distances)
        ]


        locations_true = np.nonzero(actual_gci_locations)[0]
        xaxes = np.zeros(len(actual_gci_locations))
        xaxes[locations_true] = 1

        ground_truth = np.row_stack((np.arange(len(actual_gci_locations)),
                                     xaxes))
        predicted_truth = np.row_stack((gci_locations,
                                        postive_predicted_peak_indicator))

        gx = ground_truth[0, :]
        gy = ground_truth[1, :]


        px = predicted_truth[0, :]
        py = predicted_truth[1, :]

        fs = 16000

        # gci = np.array(
        #     cluster(
        #         px,
        #         py,
        #         threshold=0.5,
        #         samples_per_bin=5,
        #         histogram_count_threshold=3))

        #####Aplawd_Male
        # gci = np.array(
        #     cluster(
        #         px,
        #         py,
        #         threshold=0.8,
        #         samples_per_bin=5,
        #         histogram_count_threshold=5))

        #####Aplawd_Female


        threshold = 0.5
        samples_per_bin = 5
        histogram_count_threshold = 1

        gci = np.array(
            cluster(
                px,
                py,
                threshold=threshold,
                samples_per_bin=samples_per_bin,
                histogram_count_threshold=histogram_count_threshold))

        predicted_gci_time = gci / fs
        target_gci_time = np.nonzero(gy)[0] / fs

        gci = np.round(gci).astype(np.int64)
        gcilocs = np.zeros_like(gx)
        gcilocs[gci] = 1

        metric = corrected_naylor_metrics(target_gci_time, predicted_gci_time)
        print(metric)
        metrics_list.append(metric)

    idr = np.mean([
        v for m in metrics_list for k, v in m.items()
        if k == 'identification_rate'
    ])
    mr = np.mean(
        [v for m in metrics_list for k, v in m.items() if k == 'miss_rate'])
    far = np.mean([
        v for m in metrics_list for k, v in m.items()
        if k == 'false_alarm_rate'
    ])
    se = np.mean([
        v for m in metrics_list for k, v in m.items()
        if k == 'identification_accuracy'
    ])

    print('IDR: {:.5f} MR: {:.5f} FAR: {:.5f} IDA: {:.5f}'.format(
        idr, mr, far, se))

    with open(result, 'w') as f:
        f.write('IDR: {:.5f} MR: {:.5f} FAR: {:.5f} IDA: {:.5f}\n'.format(
            idr, mr, far, se))
        f.write('threshold: {:.5f} samples_per_bin: {:.5f} histogram_count_threshold: {:.5f}'.format(
            threshold, samples_per_bin, histogram_count_threshold))


if __name__ == "__main__":
    main()
