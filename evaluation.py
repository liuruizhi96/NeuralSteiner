import os
import pickle
import torch
import wandb
# import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
from utils_steiner.loss import dice_coeff, FocalCoverLoss
np.set_printoptions(linewidth=np.inf)
np.set_printoptions(threshold=np.inf)

def save_as_pickle(filename, data):
    completeName = os.path.join("./results/", filename)
    with open(completeName, 'wb') as output:
        pickle.dump(data, output)

@torch.inference_mode()
def evaluate(vis, epoch, exp_name, model, val_loader, vis_loader, criterion, dice_loss, cover_loss,other_loss):

    model.eval()
    # switch model to evaluation mode.
    # This is necessary for layers like dropout, batchNorm etc. which behave differently in training and evaluation mode
    val_loss  = 0
    val_dice  = 0
    val_cover = 0
    num_val_batches = len(val_loader)
    if vis: 
        resultlist = []

    with torch.no_grad():
        for batch in tqdm(val_loader, total=num_val_batches, desc='Validation round', unit='batch', leave=False):
        # for batch_data in val_loader:
            # Load the input features and labels from the test dataset
            # pin, cost, stp = batch
            pin, cost, stp, size = batch

            input_img = torch.stack([pin, cost],dim=1).to(device='cuda:0', dtype=torch.float32)

            stp_label = stp.to(device='cuda:0', dtype=torch.float32)
            size_net = size.to(device='cuda:0', dtype=torch.int16)

            # Make predictions: Pass image data from test dataset,
            # make predictions about class image belongs to(0-9 in this case)
            # predict the mask
            stp_pred = model(input_img)
            if model.module.out_channels == 1:
                assert stp_label.min() >= 0 and stp_label.max() <= 1, 'True mask indices should be in [0, 1]'
                batch_loss = criterion(stp_pred.squeeze(1), stp_label.float())
                
                # compute the Dice score
                
                batch_loss += dice_loss(torch.sigmoid(stp_pred.squeeze(1)), stp_label.float(), size_net)
                loss_cover,cover_rate = cover_loss(torch.sigmoid(stp_pred.squeeze(1)), stp_label.float(), size_net)
                batch_loss += loss_cover
                loss_other = other_loss(torch.sigmoid(stp_pred.squeeze(1)), stp_label.float(), size_net)
                batch_loss += loss_other
                dice_score = dice_coeff(torch.sigmoid(stp_pred.squeeze(1)), stp_label.float(), size_net, reduce_batch_first=False)
                # dice_score += cover_rate
                

            # Compute the loss sum up batch loss
            val_loss += batch_loss.item()
            val_dice += dice_score.mean().item()
            val_cover += cover_rate

        #可视化
        if vis:
            vis_cover_loss = FocalCoverLoss(gamma=1, focal=False, reduction='vis')

            for batch in vis_loader:
            # for batch_data in val_loader:
                # Load the input features and labels from the test dataset
                pin_vis_, cost_vis_, stp_vis_, size_vis_   = batch

                input_vis_img = torch.stack([pin_vis_, cost_vis_],dim=1).to(device='cuda:0', dtype=torch.float32)

                stp_vis_label = stp_vis_.to(device='cuda:0', dtype=torch.float32)
                size_vis_net  = size_vis_.to(device='cuda:0', dtype=torch.int16)

                # Make predictions: Pass image data from test dataset,
                # make predictions about class image belongs to(0-9 in this case)

                # predict the mask
                stp_vis_pred = model(input_vis_img)
                if model.module.out_channels == 1:
                    assert stp_vis_label.min() >= 0 and stp_vis_label.max() <= 1, 'True mask indices should be in [0, 1]'
                    batch_loss = criterion(stp_vis_pred.squeeze(1), stp_vis_label.float())
                    if vis:
                        batch_loss += dice_loss(torch.sigmoid(stp_vis_pred.squeeze(1)), stp_vis_label.float(), size_vis_net)
                        dice_score = dice_coeff(torch.sigmoid(stp_vis_pred.squeeze(1)), stp_vis_label.float(), size_vis_net, reduce_batch_first=False)
                        _, cover_rate_vis = vis_cover_loss(torch.sigmoid(stp_vis_pred.squeeze(1)), stp_vis_label.float(), size_vis_net)
                        dice_list = dice_score.detach().cpu().numpy()
                        cover_rate_list = cover_rate_vis.detach().cpu().numpy()
                    # compute the Dice score
                pin_vis_list   = pin_vis_.numpy()
                cost_vis_list  = cost_vis_.numpy()
                stp_vis_list   = stp_vis_.numpy()
                dicescore = np.round(dice_list, 5)
                CoverRate = np.round(cover_rate_list, 5)
                predlist  = torch.sigmoid(stp_vis_pred.squeeze(1)).detach().float().cpu().numpy()
                predlist  = np.round(predlist, 5)
                # cost_sum = np.sum(predlist*cost_vis_list)
                resultlist.append([pin_vis_list,cost_vis_list,stp_vis_list,predlist,dicescore, CoverRate])
            if not os.path.exists("./results/{}".format(exp_name)):
                os.makedirs("./results/{}".format(exp_name))
                os.makedirs("./pic_results/{}".format(exp_name))
            save_as_pickle("{}/{}_epoch_{}".format(exp_name,exp_name,epoch), resultlist)

        

    model.train()
    return val_loss / max(num_val_batches, 1), val_dice / max(num_val_batches, 1), val_cover / max(num_val_batches, 1)

# 2023 1028 vs. ispd flute
@torch.inference_mode()
def ai_steiner_inference_flute(vis, exp_name, model, infer_loader, vis_loader):
    
    model.eval()
    # switch model to evaluation mode.
    # This is necessary for layers like dropout, batchNorm etc. which behave differently in training and evaluation mode

    num_infer_batches = len(infer_loader)
    all_batch_results = []

    if vis: 
        resultlist = []

    with torch.no_grad():
        for batch in tqdm(infer_loader, total=num_infer_batches, desc='Inference round', unit='batch', leave=False):
        # for batch_data in infer_loader:
            # Load the input features and labels from the test dataset
            pin, cost, idx = batch

            input_img = torch.stack([pin, cost],dim=1).to(device='cuda:0', dtype=torch.float32)


            # Make predictions: Pass image data from test dataset,
            # make predictions about class image belongs to(0-9 in this case)
            # predict the mask
            stp_pred = model(input_img)
            pin_list   = pin.numpy()
            cost_list  = cost.numpy()
            idx_list   = idx.numpy()
            print(stp_pred.shape)
            # predlist  = torch.sigmoid(stp_pred.squeeze(1)).detach().float().cpu().numpy()
            predlist  = stp_pred.detach().float().cpu().numpy()
            # print(len(predlist), 'jjjjjjjj')
            # input()
            predlist  = np.round(predlist, 5)

            for i in range(len(cost_list)):
                cost_i = cost_list[i]
                if (cost_i[:,0] >= 2).any():
                    padding_index_x = (cost_i >= 2).argmax(axis=0)
                else:
                    padding_index_x = [cost_i.shape[0]]

                if (cost_i[0] >= 2).any():
                    padding_index_y = (cost_i >= 2).argmax(axis=1)
                else:
                    padding_index_y = [cost_i.shape[1]]

                cost_map = cost_list[i][0:padding_index_x[0],0:padding_index_y[0]]
                pin_map  = pin_list[i][0:padding_index_x[0],0:padding_index_y[0]]
                try:
                    pred_stp = predlist[i][0:padding_index_x[0],0:padding_index_y[0]]
                    # print('a')
                    # print(predlist.shape)
                except:
                    print('b')
                    # print(predlist.shape)

                # all_batch_results.append([pin_list[i],cost_list[i],predlist[i],idx_list[i]])
                all_batch_results.append([pin_map,cost_map,pred_stp,idx_list[i]])
            # print(len(all_batch_results), 'hhhhhh')
            # input()

        #可视化
        if vis:

            for batch in vis_loader:
            # for batch_data in vis_loader:
                # Load the input features and labels from the test dataset
                pin_vis_, cost_vis_, idx_vis_ = batch

                input_vis_img = torch.stack([pin_vis_, cost_vis_],dim=1).to(device='cuda:0', dtype=torch.float32)


                # Make predictions: Pass image data from test dataset,
                # make predictions about class image belongs to(0-9 in this case)

                # predict the mask
                stp_vis_pred = model(input_vis_img)
                    # compute the Dice score
                pin_vis_list   = pin_vis_.numpy()
                cost_vis_list  = cost_vis_.numpy()
                idx_vis_list = idx_vis_.numpy()

                predlist  = torch.sigmoid(stp_vis_pred.squeeze(1)).detach().float().cpu().numpy()
                predlist  = np.round(predlist, 5)
                # cost_sum = np.sum(predlist*cost_vis_list)
                resultlist.append([pin_vis_list,cost_vis_list,predlist, idx_vis_list])
            if not os.path.exists("./results/{}".format(exp_name)):
                os.makedirs("./results/{}".format(exp_name))
                os.makedirs("./pic_results/{}".format(exp_name))
            save_as_pickle("{}/{}".format(exp_name,exp_name), resultlist)

        

    return all_batch_results

## 2023 0928
@torch.inference_mode()
def ai_steiner_inference(vis, exp_name, model, infer_loader, vis_loader, criterion, dice_loss, cover_loss):
    
    model.eval()
    # switch model to evaluation mode.
    # This is necessary for layers like dropout, batchNorm etc. which behave differently in training and evaluation mode
    infer_loss  = 0
    infer_dice  = 0
    infer_cover = 0
    num_infer_batches = len(infer_loader)
    all_batch_results = []
    if vis: 
        resultlist = []

    with torch.no_grad():
        for batch in tqdm(infer_loader, total=num_infer_batches, desc='Inference round', unit='batch', leave=False):
        # for batch_data in infer_loader:
            # Load the input features and labels from the test dataset
            # pin, cost, stp = batch
            pin, cost, stp, size = batch

            input_img = torch.stack([pin, cost],dim=1).to(device='cuda:0', dtype=torch.float32)

            stp_label = stp.to(device='cuda:0', dtype=torch.float32)
            size_net = size.to(device='cuda:0', dtype=torch.int16)

            # Make predictions: Pass image data from test dataset,
            # make predictions about class image belongs to(0-9 in this case)
            # predict the mask
            stp_pred = model(input_img)
            
            if model.module.out_channels == 1:
                assert stp_label.min() >= 0 and stp_label.max() <= 1, 'True mask indices should be in [0, 1]'
                batch_loss = criterion(stp_pred.squeeze(1), stp_label.float())
                
                # compute the Dice score
                
                batch_loss += dice_loss(torch.sigmoid(stp_pred.squeeze(1)), stp_label.float(), size_net)
                loss_cover,cover_rate = cover_loss(torch.sigmoid(stp_pred.squeeze(1)), stp_label.float(), size_net)
                batch_loss += loss_cover
                dice_score = dice_coeff(torch.sigmoid(stp_pred.squeeze(1)), stp_label.float(), size_net, reduce_batch_first=False)
                # dice_score += cover_rate
                

            # Compute the loss sum up batch loss
            infer_loss += batch_loss.item()
            infer_dice += dice_score.mean().item()
            infer_cover += cover_rate
            all_batch_results.append(stp_pred.detach().cpu())

        #可视化
        if vis:
            vis_cover_loss = FocalCoverLoss(gamma=1, focal=False, reduction='vis')

            for batch in vis_loader:
            # for batch_data in vis_loader:
                # Load the input features and labels from the test dataset
                pin_vis_, cost_vis_, stp_vis_, size_vis_   = batch

                input_vis_img = torch.stack([pin_vis_, cost_vis_],dim=1).to(device='cuda:0', dtype=torch.float32)

                stp_vis_label = stp_vis_.to(device='cuda:0', dtype=torch.float32)
                size_vis_net  = size_vis_.to(device='cuda:0', dtype=torch.int16)

                # Make predictions: Pass image data from test dataset,
                # make predictions about class image belongs to(0-9 in this case)

                # predict the mask
                stp_vis_pred = model(input_vis_img)
                if model.module.out_channels == 1:
                    assert stp_vis_label.min() >= 0 and stp_vis_label.max() <= 1, 'True mask indices should be in [0, 1]'
                    batch_loss = criterion(stp_vis_pred.squeeze(1), stp_vis_label.float())
                    if vis:
                        batch_loss += dice_loss(torch.sigmoid(stp_vis_pred.squeeze(1)), stp_vis_label.float(), size_vis_net)
                        dice_score = dice_coeff(torch.sigmoid(stp_vis_pred.squeeze(1)), stp_vis_label.float(), size_vis_net, reduce_batch_first=False)
                        _, cover_rate_vis = vis_cover_loss(torch.sigmoid(stp_vis_pred.squeeze(1)), stp_vis_label.float(), size_vis_net)
                        dice_list = dice_score.detach().cpu().numpy()
                        cover_rate_list = cover_rate_vis.detach().cpu().numpy()
                    # compute the Dice score
                pin_vis_list   = pin_vis_.numpy()
                cost_vis_list  = cost_vis_.numpy()
                stp_vis_list   = stp_vis_.numpy()
                dicescore = np.round(dice_list, 5)
                CoverRate = np.round(cover_rate_list, 5)
                predlist  = torch.sigmoid(stp_vis_pred.squeeze(1)).detach().float().cpu().numpy()
                predlist  = np.round(predlist, 5)
                # cost_sum = np.sum(predlist*cost_vis_list)
                resultlist.append([pin_vis_list,cost_vis_list,stp_vis_list,predlist,dicescore, CoverRate])
            if not os.path.exists("./results/{}".format(exp_name)):
                os.makedirs("./results/{}".format(exp_name))
                os.makedirs("./pic_results/{}".format(exp_name))
            save_as_pickle("{}/{}".format(exp_name,exp_name), resultlist)

        

#     return infer_loss / max(num_infer_batches, 1), infer_dice / max(num_infer_batches, 1), infer_cover / max(num_infer_batches, 1), all_batch_results

#             # Get the index of the max log-probability
#             pred = output.max(1, keepdim=True)[1]
#             correct += pred.eq(target.view_as(pred)).sum().item()

#             # Log images in your test dataset automatically,
#             # along with predicted and true labels by passing pytorch tensors with image data into wandb.
#             example_images.append(wandb.Image(
#                 data[0], caption="Pred:{} Truth:{}".format(classes[pred[0].item()], classes[target[0]])))

#    # wandb.log(a_dict) logs the keys and values of the dictionary passed in and associates the values with a step.
#    # You can log anything by passing it to wandb.log(),
#    # including histograms, custom matplotlib objects, images, video, text, tables, html, pointclounds and other 3D objects.
#    # Here we use it to log test accuracy, loss and some test images (along with their true and predicted labels).
#     wandb.log({
#         "Examples": example_images,
#         "Test Accuracy": 100. * correct / len(infer_loader.dataset),
#         "Test Loss": test_loss
#     })



# @torch.inference_mode()
# def evaluate(net, val_loader, device, amp):
#     net.eval()
#     num_val_batches = len(val_loader)
#     dice_score = 0

#     # iterate over the validation set
#     with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=amp):
#         for batch in tqdm(val_loader, total=num_val_batches, desc='Validation round', unit='batch', leave=False):
#             image, mask_true = batch['image'], batch['mask']

#             # move images and labels to correct device and type
#             image = image.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
#             mask_true = mask_true.to(device=device, dtype=torch.long)

#             # predict the mask
#             mask_pred = net(image)

#             if net.n_classes == 1:
#                 assert mask_true.min() >= 0 and mask_true.max() <= 1, 'True mask indices should be in [0, 1]'
#                 mask_pred = (F.sigmoid(mask_pred) > 0.5).float()
#                 # compute the Dice score
#                 dice_score += dice_coeff(mask_pred, mask_true, reduce_batch_first=False)
#             else:
#                 assert mask_true.min() >= 0 and mask_true.max() < net.n_classes, 'True mask indices should be in [0, n_classes]'
#                 # convert to one-hot format
#                 mask_true = F.one_hot(mask_true, net.n_classes).permute(0, 3, 1, 2).float()
#                 mask_pred = F.one_hot(mask_pred.argmax(dim=1), net.n_classes).permute(0, 3, 1, 2).float()
#                 # compute the Dice score, ignoring background
#                 dice_score += multiclass_dice_coeff(mask_pred[:, 1:], mask_true[:, 1:], reduce_batch_first=False)

#     net.train()
#     return dice_score / max(num_val_batches, 1)