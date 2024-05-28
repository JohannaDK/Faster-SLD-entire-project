"""
changed optimizer for new multi-head model
"""

from datetime import datetime
import logging
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from inference import *
from dataloader.indoor6 import *
from dataloader.backbone_dataloader import *
from models.efficientlitesld import EfficientNetSLD
from models.backbone_model import *
from utils.heatmap import generate_heat_maps_gpu

# unchanged from train.py, just plots losses and (I think) validation errors
def plotting(ROOT_FOLDER, scene):
    data = pickle.load(open('%s/stats_%s.pkl' % (ROOT_FOLDER,scene), 'rb'))
    fig, axs = plt.subplots(4, 1)

    t = 0
    s = []
    epoch = 0
    for i in range(len(data[scene]['train'])-1):
        if data[scene]['train'][i+1]['ep'] == epoch + 1:
            epoch += 1
        else:
            t += 1
            s.append(data[scene]['train'][i]['loss'])

    t = np.arange(0, t)
    s = np.array(s)
    s = np.convolve(s, np.ones(10)/10., mode='same')

    axs[0].plot(t, np.log(s))
    axs[0].set(xlabel='iterations', ylabel='loss', title='')
    axs[0].grid()

    max_grad = np.array([data[scene]['train'][i]['max_grad'] for i in range(len(data[scene]['train']))])
    axs[1].plot(np.arange(0, len(max_grad)), np.log10(max_grad))
    axs[1].set(xlabel='iterations', ylabel='max gradient', title='')
    axs[1].grid()

    t = np.array([data[scene]['eval'][i]['ep'] for i in range(len(data[scene]['eval']))])
    s = np.array([np.median(data[scene]['eval'][i]['pixel_error']) for i in range(len(data[scene]['eval']))])
    axs[2].plot(t, s)
    axs[2].set(xlabel='epoch', ylabel='Pixel error', title='')
    axs[2].grid()
    axs[2].set_yticks(np.arange(0, 20, 5), minor=False)
    axs[2].set_ylim(0, 20)

    r = np.array([data[scene]['eval'][i]['recall'] for i in range(len(data[scene]['eval']))])
    axs[3].plot(t, r)
    axs[3].set(xlabel='epoch', ylabel='recall', title='')
    axs[3].grid()

    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.8, hspace=1.0)
    plt.close()
    fig.savefig('%s/curve_train_test_%s.png' % (ROOT_FOLDER,scene), format='png', dpi=120)

def train_patches(opt):

    if not os.path.exists(opt.output_folder):
        os.makedirs(opt.output_folder)

    logging.basicConfig(filename='%s/training.log' % opt.output_folder, filemode='a', level=logging.DEBUG, format='')
    logging.info("Scene Landmark Detector Training Patches")
    print('Start training ...')

    backbone_scenes = ["scene1","scene2a", "scene4a", "scene5", "scene6"]
    backbone_train_dataset_list = []

    stats_pkl_logging = {}
    for scene in backbone_scenes:
        stats_pkl_logging[scene] = {'train': [], 'eval': []}

    device = opt.gpu_device

    assert len(opt.landmark_indices) == 0 or len(opt.landmark_indices) == 2, "landmark indices must be empty or length 2"
    for scene in backbone_scenes:
        backbone_train_dataset_list.append(Indoor6Patches(landmark_idx=np.arange(opt.landmark_indices[0],
                                                    opt.landmark_indices[1]) if len(opt.landmark_indices) == 2 else [None],
                                scene_id=scene,
                                mode='train',
                                root_folder=opt.dataset_folder,
                                input_image_downsample=2,
                                landmark_config=opt.landmark_config,
                                visibility_config=opt.visibility_config,
                                skip_image_index=1))
    backbone_train_dataset = CombinedDataset(backbone_train_dataset_list,shuffle=True)
    backbone_train_sampler = HomogeneousBatchSampler(backbone_train_dataset, opt.training_batch_size, shuffle=True)
    backbone_train_dataloader = DataLoader(dataset = backbone_train_dataset, num_workers=4, batch_sampler=backbone_train_sampler,pin_memory=True)
    
    ## Save the trained landmark configurations
    for i,scene in enumerate(backbone_scenes):
        np.savetxt(os.path.join(opt.output_folder, 'landmarks_{}.txt'.format(scene)), backbone_train_dataset_list[i].landmark)
        np.savetxt(os.path.join(opt.output_folder, 'visibility_{}.txt'.format(scene)), backbone_train_dataset_list[i].visibility, fmt='%d')

    num_landmarks = backbone_train_dataset_list[0].landmark.shape[1]

    if opt.model == 'efficientnet-backbonev1':
        backbone = BackboneV1(model="efficientnet", output_downsample=opt.output_downsample).to(device=device)
        feats = backbone.output_channels
        heads = {}
        models = {}
        for scene in backbone_scenes:
            heads[scene] = SceneHeadV1(num_landmarks=num_landmarks, features=feats).to(device=device)
            models[scene] = nn.Sequential(backbone, heads[scene])
    elif opt.model == 'resnet-backbonev1':
        backbone = BackboneV1(model="resnet", output_downsample=opt.output_downsample).to(device=device)
        feats = backbone.output_channels
        heads = {}
        models = {}
        for scene in backbone_scenes:
            heads[scene] = SceneHeadV1(num_landmarks=num_landmarks, features=feats).to(device=device)
            models[scene] = nn.Sequential(backbone, heads[scene])
    elif opt.model == 'efficientnet-backbonev2':
        backbone = BackboneV2(model="efficientnet", output_downsample=opt.output_downsample).to(device=device)
        feats = backbone.output_channels
        heads = {}
        models = {}
        for scene in backbone_scenes:
            heads[scene] = SceneHeadV2(num_landmarks=num_landmarks, features=feats).to(device=device)
            models[scene] = nn.Sequential(backbone, heads[scene])
    elif opt.model == 'resnet-bbv2headv3':
        feats = 512
        model = MultiHeadModel(bb_model="resnetv2",head_version="v3",scenes=backbone_scenes,features=feats,num_landmarks=num_landmarks).to(device=device)
    elif opt.model == 'effnet-bbv2headv3':
        feats = 320
        model = MultiHeadModel(bb_model="efficientnetv2",head_version="v3",scenes=backbone_scenes,features=feats,num_landmarks=num_landmarks).to(device=device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, betas=(0.9, 0.999), eps=1e-4, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)

    lowest_median_angular_error = 1e6

    for epoch in range(opt.num_epochs):
        # Training
        training_loss = 0
        for idx, batch in enumerate(tqdm(backbone_train_dataloader)):
            assert all(sc == batch['scene'][0] for sc in batch['scene'])
            cur_scene = batch['scene'][0]
            model.train()

            B1, B2, _, H, W = batch['patches'].shape
            B = B1 * B2
            patches = batch['patches']
            visibility = batch['visibility']
            landmark2d = batch['landmark2d']

            # highest supported precision for MPS is FP32
            if device.lower() == 'mps':
                patches = patches.float()
                visibility = visibility.float()
                landmark2d = landmark2d.float()

            patches = patches.reshape(B, 3, H, W).to(device=device)
            visibility = visibility.reshape(B, num_landmarks).to(device=device)
            landmark2d = landmark2d.reshape(B, 2, num_landmarks).to(device=device)

            # Batch randomization
            input_batch_random = np.random.permutation(B)
            landmark2d_rand = [landmark2d[input_batch_random[b:b + 1]] for b in range(B)]
            patches_rand = [patches[input_batch_random[b:b + 1]] for b in range(B)]
            visibility_rand = [visibility[input_batch_random[b:b + 1]] for b in range(B)]

            landmark2d_rand = torch.cat(landmark2d_rand, dim=0)
            patches_rand = torch.cat(patches_rand, dim=0)
            visibility_rand = torch.cat(visibility_rand, axis=0)

            # Resolution configure
            landmark2d_rand /= opt.output_downsample
            heat_map_size = [H // opt.output_downsample, W // opt.output_downsample]

            gt = generate_heat_maps_gpu(landmark2d_rand,
                                        visibility_rand,
                                        heat_map_size,
                                        sigma=torch.tensor([20. / opt.output_downsample], dtype=torch.float, device=device, requires_grad=False))
            gt.requires_grad = False

            # Clear gradient
            optimizer.zero_grad()

            # CNN forward pass
            pred = model(patches_rand,cur_scene)

            # Compute loss and do backward pass

            losses = torch.sum((pred[visibility_rand != 0.5] - gt[visibility_rand != 0.5]) ** 2)

            training_loss += losses.detach().clone().item()
            losses.backward()

            m = torch.tensor([0.0]).to(device)
            for p in model.parameters():
                if p.grad is not None:
                    m = torch.max(torch.max(torch.abs(p.grad.data)), m)

            ## Ignore batch with large gradient element
            if epoch == 0 or (epoch > 0 and m < 1e4):
                optimizer.step()
            else:
                model.load_state_dict(torch.load('%s/model-best_median.ckpt' % (opt.output_folder)))
                model.to(device=device)

            logging.info('epoch %d, iter %d, loss %4.4f' % (epoch, idx, losses.item()))
            stats_pkl_logging[cur_scene]['train'].append({'ep': epoch, 'iter': idx, 'loss': losses.item(), 'max_grad': m.cpu().numpy()})

        path = '{}/model-latest.ckpt'.format(opt.output_folder)
        torch.save(model.state_dict(),path)

        if scheduler.get_last_lr()[-1] > 5e-5:
            scheduler.step()

        if epoch % 5 == 0:
            mean_median_angular_error = 0

            for scene in backbone_scenes:
                opt.pretrained_model = [path]
                opt.scene_id = scene
                eval_stats = inference(opt, opt_tight_thr=1e-3, minimal_tight_thr=1e-3, mode='val')

                mean_median_angular_error += np.median(eval_stats['angular_error']) / len(backbone_scenes)

                # date time
                ts = datetime.now().timestamp()
                dt = datetime.fromtimestamp(ts)
                datestring = dt.strftime("%Y-%m-%d_%H-%M-%S")

                # Print, log and update plot
                stats_pkl_logging[scene]['eval'].append(
                    {'ep': epoch,
                    'angular_error': eval_stats['angular_error'],
                    'pixel_error': eval_stats['pixel_error'],
                    'recall': eval_stats['r5p5']
                    })


                try:
                    str_log = 'scene %s' \
                            'epoch %3d: [%s] ' \
                            'tr_loss= %10.2f, ' \
                            'lowest_median= %8.4f deg. ' \
                            'recall= %2.4f ' \
                            'angular-err(deg.)= [%7.4f %7.4f %7.4f]  ' \
                            'pixel-err= [%4.3f %4.3f %4.3f] [mean/med./min] ' % (scene, epoch, datestring, training_loss,
                                                                                    lowest_median_angular_error,
                                                                                    eval_stats['r5p5'],
                                                                                    np.mean(eval_stats['angular_error']),
                                                                                    np.median(eval_stats['angular_error']),
                                                                                    np.min(eval_stats['angular_error']),
                                                                                    np.mean(eval_stats['pixel_error']),
                                                                                    np.median(eval_stats['pixel_error']),
                                                                                    np.min(eval_stats['pixel_error']))
                    print(str_log)
                    logging.info(str_log)
                except ValueError:  #raised if array is empty.
                    str_log = 'epoch %3d: [%s] ' \
                                'tr_loss= %10.2f, ' \
                                'No correspondences found' % (epoch, datestring, training_loss)
                    print(str_log)
                    logging.info(str_log)

                with open('%s/stats_%s.pkl' % (opt.output_folder,scene) , 'wb') as f:
                    pickle.dump(stats_pkl_logging, f)
                plotting(opt.output_folder, scene)
            
            path = '%s/model-best_median.ckpt' % (opt.output_folder)
            if (mean_median_angular_error < lowest_median_angular_error):
                lowest_median_angular_error = mean_median_angular_error
                torch.save(model.state_dict(), path)
                path = path = '%s/bb-best_median.ckpt' % (opt.output_folder)
                torch.save(model.bb.state_dict(), path)
        
            if (~os.path.exists(path) and len(eval_stats['angular_error']) == 0):
                torch.save(model.state_dict(), path)
