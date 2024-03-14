import os
import pickle
import argparse
import numpy as np
import torch

import utils
import attacks
import random

def get_args():
    parser = argparse.ArgumentParser()
    utils.add_shared_args(parser)

    parser.add_argument('--data-mode', type=str, default='whole',
                        choices=['whole', 'mix'],
                        help='mix = clear + unlearnable data, clear = clear data only')

    parser.add_argument('--filter', type=str, default=None,
                        choices=['averaging', 'gaussian', 'median', 'bilateral'],
                        help='select the low pass filter; only works in [mix] mode')

    parser.add_argument('--man-data-path', type=str, default=None,
                        help='set the path to the manual dataset')
    parser.add_argument('--noise-path', type=str, default=None,
                        help='set the path to the train images noises')
    parser.add_argument('--poi-idx-path', type=str, default=None,
                        help='set the path to the poisoned indices')
    parser.add_argument('--resume-path', type=str, default=None,
                        help='set where to resume the model')

    parser.add_argument('--perturb-freq', type=int, default=1,
                        help='set the perturbation frequency')
    parser.add_argument('--report-freq', type=int, default=500,
                        help='set the report frequency')
    parser.add_argument('--save-freq', type=int, default=5000,
                        help='set the checkpoint saving frequency')
    parser.add_argument('--adversarial', type=int, default=1)
    parser.add_argument('--clean', default=0, type=int)
    parser.add_argument('--wandb', action='store_true')
    parser.add_argument('--wandb-project', type=str, default='gen-noise')
    parser.add_argument('--wandb-name', type=str, default='noise')
    parser.add_argument('--wandb-notes', type=str, default='')
    parser.add_argument('--random', type=int, default=42)

    return parser.parse_args()


def get_manual_loader(dataset, data_path, batch_size):
    with open(data_path, 'rb') as f:
        man_dataset = pickle.load(f)
    trans = utils.get_transforms(dataset, train=True, is_tensor=False)

    man_dataset = utils.Dataset( man_dataset['x'], man_dataset['y'].astype(np.int64), trans )
    loader = utils.Loader(man_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    return loader


def save_checkpoint(save_dir, save_name, model, optim, log):
    torch.save({
        'model_state_dict': utils.get_model_state(model),
        'optim_state_dict': optim.state_dict(),
        }, os.path.join(save_dir, '{}-model.pkl'.format(save_name)))
    with open(os.path.join(save_dir, '{}-log.pkl'.format(save_name)), 'wb') as f:
        pickle.dump(log, f, protocol=4)

def random_seed(seed):
    random.seed(seed)
    os.environ["PYTHONSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main(args, logger):
    ''' init model / optim / dataloader / loss func '''
    model = utils.get_arch(args.arch, args.dataset)

    if args.resume_path is not None:
        state_dict = torch.load(args.resume_path, map_location=torch.device('cpu'))
        model.load_state_dict( state_dict['model_state_dict'] )
        del state_dict

    criterion = torch.nn.CrossEntropyLoss()

    if not args.cpu:
        model.cuda()
        criterion = criterion.cuda()

    if args.parallel:
        model = torch.nn.DataParallel(model)
        
    optim = utils.get_optim(
        args.optim, model.parameters(),
        lr=args.lr, weight_decay=args.weight_decay, momentum=args.momentum)

    if args.data_mode == 'whole':
        train_loader = utils.get_whole_loader(
            args.dataset, batch_size=args.batch_size, root=args.data_dir, train=True,
            noise_path=args.noise_path, clean=args.clean, fitr=args.filter)
    elif args.data_mode == 'mix':
        train_loader = utils.get_mix_loader(
            args.dataset, batch_size=args.batch_size, root=args.data_dir, train=True,
            clean=args.clean, noise_path=args.noise_path, poisoned_indices_path=args.poi_idx_path, fitr=args.filter
        )
    else:
        raise NotImplementedError

    test_loader = utils.get_poisoned_loader(
        args.dataset, batch_size=args.batch_size, root=args.data_dir, train=False)

    attacker = attacks.PGDAttacker(
        radius = args.pgd_radius,
        steps = args.pgd_steps,
        step_size = args.pgd_step_size,
        random_start = args.pgd_random_start,
        norm_type = args.pgd_norm_type,
        ascending = True,
    )

    log = dict()
    
    if args.wandb:
        import wandb
        wandb.init(
            project = args.wandb_project,
            config  = args,
            name    = args.wandb_name + ('-AT' if args.adversarial else ''),
            notes   = args.wandb_notes
        )
        
    # if args.arch == 'vit' or args.arch == 'swin':
    #     scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim)

    for step in range(args.train_steps):
        lr = args.lr * (args.lr_decay_rate ** (step // args.lr_decay_freq))
        for group in optim.param_groups:
            group['lr'] = lr

        x, y = next(train_loader)
        if not args.cpu:
            x, y = x.cuda(), y.cuda()

        if args.adversarial:
            adv_x = attacker.perturb(model, criterion, x, y)
        else:
            adv_x = x

        model.train()
        _y = model(adv_x)
        adv_acc = (_y.argmax(dim=1) == y).sum().item() / len(x)
        adv_loss = criterion(_y, y)
        optim.zero_grad()
        adv_loss.backward()
        optim.step()

        utils.add_log(log, 'adv_acc', adv_acc)
        utils.add_log(log, 'adv_loss', adv_loss.item())
        
        if args.wandb:
            wandb.log({
                # 'lr': lr,
                'adv_acc': adv_acc,
                'adv_loss': adv_loss
            })

        if (step+1) % args.save_freq == 0:
            save_checkpoint(
                args.save_dir, '{}-ckpt-{}'.format(args.save_name, step+1),
                model, optim, log)

        if (step+1) % args.report_freq == 0:
            test_acc, test_loss = utils.evaluate(model, criterion, test_loader, args.cpu)
            utils.add_log(log, 'test_acc', test_acc)
            utils.add_log(log, 'test_loss', test_loss)

            logger.info('step [{}/{}]:'.format(step+1, args.train_steps))
            logger.info('adv_acc {:.2%} \t adv_loss {:.3e}'
                        .format( adv_acc, adv_loss.item() ))
            logger.info('test_acc  {:.2%} \t test_loss  {:.3e}'
                        .format( test_acc, test_loss ))
            logger.info('')
            
            if args.wandb:
                wandb.log({
                    'test_acc': test_acc,
                    'test_loss': test_loss
                })

    # save_checkpoint(args.save_dir, '{}'.format(args.save_name), model, optim, log)

    return


if __name__ == '__main__':
    # random_seed(42)
    args = get_args()
    logger = utils.generic_init(args)
    random_seed(args.random)

    try:
        main(args, logger)
    except Exception as e:
        logger.exception('Unexpected exception! %s', e)
