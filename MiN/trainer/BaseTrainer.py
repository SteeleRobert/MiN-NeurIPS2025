import sys
import logging
import torch
from utils.factory import get_model
from data_process.data_manger import DataManger
import os
import datetime
import json


def train(args):
    _train(args)


_DATASET_TO_BENCHMARK = {
    'cifar224': 'cifar_100',
    'imagenetr': 'imagenet_r',
    'imageneta': 'imagenet_a',
    'cub': 'cub_200',
    'omnibenchmark': 'omnibenchmark',
    'vtab': 'vtab',
    'objectnet': 'objectnet',
    'ifood101': 'food_101',
}


def _compute_cil_metrics(task_accs_history):
    """Compute avg_acc, avg_inc_acc, bwt, task0_final from per-step per-task accuracies."""
    T = len(task_accs_history)
    final_accs = task_accs_history[-1]
    avg_acc = sum(final_accs.values()) / len(final_accs)

    bwt_sum, bwt_count = 0.0, 0
    for t in range(T - 1):
        bwt_sum += task_accs_history[-1].get(t, 0) - task_accs_history[t].get(t, 0)
        bwt_count += 1
    bwt = bwt_sum / bwt_count if bwt_count > 0 else 0.0

    task0_final = task_accs_history[-1].get(0, 0)

    inc_accs = [sum(task_accs_history[s].values()) / len(task_accs_history[s])
                for s in range(T)]
    avg_inc_acc = sum(inc_accs) / len(inc_accs)

    return {'avg_acc': avg_acc, 'avg_inc_acc': avg_inc_acc,
            'bwt': bwt, 'task0_final': task0_final}


def _save_compcont_results(args, task_accs_history):
    """Save results in compcont-bench JSON format under ~/qz-compcont-learning/results/."""
    metrics = _compute_cil_metrics(task_accs_history)

    benchmark_key = _DATASET_TO_BENCHMARK.get(args['dataset'], args['dataset'])
    backbone = args.get('backbone_type', 'unknown')
    label = 'MiN ({} init={} inc={})'.format(
        args['dataset'], args['init_class'], args['increment'])

    result = {
        'label': label,
        'metrics': metrics,
        'task_accs_history': {
            str(step): {str(t): acc for t, acc in step_accs.items()}
            for step, step_accs in enumerate(task_accs_history)
        },
        'classifier_stats': {},
        'timings': [],
        'seed': args.get('seed', 0),
        'backbone': backbone,
        'classifier': 'MiN',
        'benchmark': benchmark_key,
    }

    results_root = os.environ.get(
        'COMPCONT_RESULTS_DIR',
        os.path.join(os.path.expanduser('~'), 'qz-compcont-learning', 'results'),
    )
    bdir = os.path.join(results_root, benchmark_key)
    os.makedirs(bdir, exist_ok=True)
    path = os.path.join(bdir, 'MiN_{}.json'.format(backbone))
    with open(path, 'w') as f:
        json.dump(result, f, indent=2)

    print('  Saved: {}'.format(path))
    print('  FINAL: avg_acc={:.1f}%  avg_inc_acc={:.1f}%  bwt={:+.3f}  task0={:.1f}%'.format(
        metrics['avg_acc'] * 100, metrics['avg_inc_acc'] * 100,
        metrics['bwt'], metrics['task0_final'] * 100))
    return path


def _train(args):
    now_time = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    logs_name = "logs/{}/{}/{}_{}/{}".format(args["dataset"], args["model"], args["init_class"], args["increment"],
                                             now_time)
    workdir = os.path.join(logs_name, 'work_dir')
    if not os.path.exists(logs_name):
        os.makedirs(logs_name)
    if not os.path.exists(workdir):
        os.makedirs(workdir)
    with open(os.path.join(workdir, 'configs.json'), 'w', encoding='utf-8') as json_file:
        json.dump(args, json_file, indent=2)

    logfilename = "logs/{}/{}/{}_{}/{}/{}_{}_{}_{}_{}".format(
        args["dataset"],
        args["model"],
        args["init_class"],
        args["increment"],
        now_time,
        args["dataset"],
        args["model"],
        args["init_class"],
        args["increment"],
        args["backbone_type"]
    )

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(filename)s] => %(message)s",
        handlers=[
            logging.FileHandler(filename=logfilename + ".log"),
        ],
    )

    # _set_random()
    _set_device(args)
    print_args(args)

    datamanger = DataManger(args['dataset'], args['device'], args)

    model = get_model(args, logging)

    task_accs_history = []

    model.init_train(data_manger=datamanger)
    eval_res = model.after_train(data_manger=datamanger)
    if eval_res is not None and 'all_task_accy' in eval_res:
        task_accs_history.append(dict(eval_res['all_task_accy']))
    save_path = os.path.join(workdir, 'task_0_check_point.pth')
    if args["save_all_checkpoint"] is True:
        model.save_check_point(save_path)

    for i in range(datamanger.task_size):
        save_path = os.path.join(workdir, 'task_{}_check_point.pth'.format(i+1))
        model.increment_train(data_manger=datamanger)
        eval_res = model.after_train(data_manger=datamanger)
        if eval_res is not None and 'all_task_accy' in eval_res:
            task_accs_history.append(dict(eval_res['all_task_accy']))
        if args["save_all_checkpoint"] is True:
            model.save_check_point(save_path)
    save_path = os.path.join(logs_name, 'Last_check_point.pth')
    model.save_check_point(save_path)

    if task_accs_history:
        _save_compcont_results(args, task_accs_history)


def _set_device(args):
    device_type = args["device"]
    gpus = []

    for device in device_type:
        if device_type == -1:
            device = torch.device("cpu")
        else:
            device = torch.device("cuda:{}".format(device))
        gpus.append(device)

    args["device"] = gpus[0]


def print_args(args):
    for key, value in args.items():
        logging.info("{}: {}".format(key, value))


def _set_random(seed: int = 0):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
