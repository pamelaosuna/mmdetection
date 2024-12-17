# Copyright (c) OpenMMLab. All rights reserved.
import datetime
import itertools
import os.path as osp
import tempfile
from collections import OrderedDict
from typing import Dict, List, Optional, Sequence, Union

import numpy as np
import torch
from mmengine.evaluator import BaseMetric
from mmengine.fileio import dump, get_local_path, load
from mmengine.logging import MMLogger
from terminaltables import AsciiTable

from mmdet.datasets.api_wrappers import COCO, COCOeval, COCOevalMP
from mmdet.registry import METRICS
from mmdet.structures.mask import encode_mask_results
from ..functional import eval_recalls
from mmdet.evaluation.metrics.coco_metric import CocoMetric


@METRICS.register_module()
class SpineMetric(CocoMetric):
    """Spine evaluation metric.

    Evaluate precision, recall and F1-score for spine detection, which is based on
    box detection.

    Args:
    ann_file (str, optional): Path to the coco format annotation file.
        If not specified, ground truth annotations from the dataset will
        be converted to coco format. Defaults to None.
    proposal_nums (Sequence[int]): Numbers of proposals to be evaluated.
        Defaults to (100, 300, 1000).
    iou_thrs (float, optional): IoU threshold to compute precision and recall. 
        Default to 0.5.
    metric_items (List[str], optional): Metric result names to be
        recorded in the evaluation result. Defaults to None.
    format_only (bool): Format the output results without perform
        evaluation. It is useful when you want to format the result
        to a specific format and submit it to the test server.
        Defaults to False.
    outfile_prefix (str, optional): The prefix of json files. It includes
        the file path and the prefix of filename, e.g., "a/b/prefix".
        If not specified, a temp file will be created. Defaults to None.
    file_client_args (dict, optional): Arguments to instantiate the
        corresponding backend in mmdet <= 3.0.0rc6. Defaults to None.
    backend_args (dict, optional): Arguments to instantiate the
        corresponding backend. Defaults to None.
    collect_device (str): Device name used for collecting results from
        different ranks during distributed training. Must be 'cpu' or
        'gpu'. Defaults to 'cpu'.
    prefix (str, optional): The prefix that will be added in the metric
        names to disambiguate homonymous metrics of different evaluators.
        If prefix is not provided in the argument, self.default_prefix
        will be used instead. Defaults to None.
    """
    default_prefix: Optional[str] = 'coco'

    def __init__(self,
                 ann_file: Optional[str] = None,
                 metric: Union[str, List[str]] = 'bbox',
                 classwise: bool = False,
                 proposal_nums: Sequence[int] = (100,300, 1000),
                 iou_thrs: Optional[Union[float, Sequence[float]]] = [0.5],
                 metric_items: Optional[Sequence[str]] = None,
                 format_only: bool = False,
                 outfile_prefix: Optional[str] = None,
                 file_client_args: dict = None,
                 backend_args: dict = None,
                 collect_device: str = 'cpu',
                 prefix: Optional[str] = None,
                 sort_categories: bool = False,
                 use_mp_eval: bool = False) -> None:
        super().__init__(
            ann_file=ann_file,
            metric=metric,
            classwise=classwise,
            proposal_nums=proposal_nums,
            iou_thrs=iou_thrs,
            metric_items=metric_items,
            format_only=format_only,
            outfile_prefix=outfile_prefix,
            file_client_args=file_client_args,
            backend_args=backend_args,
            collect_device=collect_device,
            prefix=prefix,
            sort_categories=sort_categories,
            use_mp_eval=use_mp_eval)
        
    def compute_metrics(self, results: list) -> Dict[str, float]:
        """Compute the metrics from processed results.

        Args:
            results (list): The processed results of each batch.

        Returns:
            Dict[str, float]: The computed metrics. The keys are the names of
            the metrics, and the values are corresponding results.
        """

        logger: MMLogger = MMLogger.get_current_instance()

        # split gt and prediction list
        gts, preds = zip(*results)

        tmp_dir = None
        if self.outfile_prefix is None:
            tmp_dir = tempfile.TemporaryDirectory()
            outfile_prefix = osp.join(tmp_dir.name, 'results')
        else:
            outfile_prefix = self.outfile_prefix

        if self._coco_api is None:
            # use converted gt json file to initialize coco api
            logger.info('Converting ground truth to coco format...')
            coco_json_path = self.gt_to_coco_json(
                gt_dicts=gts, outfile_prefix=outfile_prefix)
            self._coco_api = COCO(coco_json_path)

        # handle lazy init
        if self.cat_ids is None:
            self.cat_ids = self._coco_api.get_cat_ids(
                cat_names=self.dataset_meta['classes'])
        if self.img_ids is None:
            self.img_ids = self._coco_api.get_img_ids()

        # convert predictions to coco format and dump to json file
        result_files = self.results2json(preds, outfile_prefix)

        eval_results = OrderedDict()
        if self.format_only:
            logger.info('results are saved in '
                        f'{osp.dirname(outfile_prefix)}')
            return eval_results

        for metric in self.metrics:
            logger.info(f'Evaluating {metric}...')

            # evaluate proposal, bbox and segm
            iou_type = 'bbox' if metric == 'proposal' else metric
            if metric not in result_files:
                raise KeyError(f'{metric} is not in results')
            try:
                predictions = load(result_files[metric])
                coco_dt = self._coco_api.loadRes(predictions)

            except IndexError:
                logger.error(
                    'The testing results of the whole dataset is empty.')
                break

            coco_eval = COCOeval(self._coco_api, coco_dt, iou_type)

            coco_eval.params.catIds = self.cat_ids
            coco_eval.params.imgIds = self.img_ids
            coco_eval.params.maxDets = list(self.proposal_nums)
            coco_eval.params.iouThrs = self.iou_thrs

            coco_eval.evaluate()
            coco_eval.accumulate()
            coco_eval.summarize()

            ## custom 
            # mapping of cocoEval.stats
            coco_metric_names = {
                'precision': 0,
                'recall': 1, 
                'f1': 2
            }
            metric_items = self.metric_items
            if metric_items is not None:
                for metric_item in metric_items:
                    if metric_item not in coco_metric_names:
                        raise KeyError(
                            f'metric item "{metric_item}" is not supported')

            precisions = coco_eval.eval['precision']
            recalls = coco_eval.eval['recall']

            results_per_category = []
            for idx, cat_id in enumerate(self.cat_ids):
                t = []
                # area range index 0: all area ranges
                # max dets index -1: typically 100 per image
                nm = self._coco_api.loadCats(cat_id)[0]
                # precision: (iou, recall, cls, area range, max dets)
                prec = precisions[:, :, idx, 0, 0]
                prec = prec[prec > -1]

                rec = recalls[:, idx, 0, 0]
                rec = rec[rec > -1]
                
                f1s = 2 * (prec * rec) / (prec + rec + 1e-6)

                if prec.size:
                    mean_prec = np.mean(prec)
                    mean_rec = np.mean(rec)
                    mean_f1 = np.mean(f1s)
                else:
                    mean_prec = float('nan')
                    mean_rec = float('nan')
                    mean_f1 = float('nan')
                
                t.append(f'{nm["name"]}')
                t.append(f'{round(mean_prec, 3)}')
                eval_results[f'{nm["name"]}_precision'] = round(mean_prec, 3)

                t.append(f'{round(mean_rec, 3)}')
                eval_results[f'{nm["name"]}_recall'] = round(mean_rec, 3)

                t.append(f'{round(mean_f1, 3)}')
                eval_results[f'{nm["name"]}_f1'] = round(mean_f1, 3)

                results_per_category.append(tuple(t))

            num_columns = len(results_per_category[0])
            results_flatten = list(
                itertools.chain(*results_per_category))
            headers = [
                'category', 'precision', 'recall', 'f1'
            ]
            results_2d = itertools.zip_longest(*[
                    results_flatten[i::num_columns]
                    for i in range(num_columns)]
                    )
            table_data = [headers]
            table_data += [result for result in results_2d]
            table = AsciiTable(table_data)
            logger.info('\n' + table.table)
            if metric_items is None:
                metric_items = [
                    'precision', 'recall', 'f1'
                ]

            for metric_item in metric_items:
                key = f'{metric}_{metric_item}'
                val = coco_eval.stats[coco_metric_names[metric_item]]
                eval_results[key] = float(f'{round(val, 3)}')
            
            prec_rec_f1 = coco_eval.stats[:3]
            logger.info(f'{metric}: {prec_rec_f1[0]:.3f} '
                        f'{prec_rec_f1[1]:.3f} {prec_rec_f1[2]:.3f} ')
            
        if tmp_dir is not None:
            tmp_dir.cleanup()
        return eval_results