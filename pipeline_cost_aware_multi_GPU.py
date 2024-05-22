import random
import argparse
import torch
import torch.nn as nn
import logging
import pickle
from diff_data_generate import save_as_pickle
from utils_steiner.steiner_dataset import load_pickle

from train_stp_multi_GPU import train_model
from infer_stp import infer_model_flute

from net.unet_small import UNet_Small_4, UNet_Small_5, UNet_Small_4_ccnet, UNet_Small_5_ccnet
from net.res_ccanet import ResNet,ResCcaNet, ResCcaNet_34
from net.models.routenet import RouteNet

if __name__ == '__main__':
    

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-e','--epoch',
        help='num of epoch',
        type=int,
        default=50,
    )
    parser.add_argument(
        '-test','--test_case',
        help='idx of test case',
        type=int,
        default=1,
    )
    parser.add_argument(
        '-b','--batchsize',
        help='batchsize',
        type=int,
        default=32,
    )
    parser.add_argument(
        '-lr','--learning_rate',
        help='learning rate',
        type=float,
        default=1e-3,
    )
    parser.add_argument(
        '-gc','--gradient_clip',
        help='gradient_clip',
        type=float,
        default=1.0,
    )
    parser.add_argument(
        '-m','--momentum',
        help='momentum',
        type=float,
        default=0.9,
    )
    parser.add_argument(
        '-beta2','--beta2',
        help='beta2',
        type=float,
        default=0.999,
    )
    parser.add_argument(
        '-shuf','--shuffle',
        help='shuffle',
        type=str,
        choices=['F', 'T'],
    )
    parser.add_argument(
        '-v','--visualization',
        help='visualization',
        type=str,
        default='F',
    )
    parser.add_argument(
        '-t','--train_mode',
        help='train_mode',
        type=str,
        default='T',
    )
    parser.add_argument(
        '-data','--datasize',
        help='datasize',
        type=str,
        choices=['small', 'big', 'large'],
    )
    parser.add_argument(
        '-di','--diceloss',
        help='diceloss',
        type=float,
        default=4.0,
    )
    parser.add_argument(
        '-c','--costloss',
        help='costloss',
        type=float,
        default=1.0,
    )
    parser.add_argument(
        '-bce','--bceloss',
        help='bceloss',
        type=float,
        default=1.0,
    )
    parser.add_argument(
        '-p','--posweight',
        help='posweight',
        type=float,
        default=1.0,
    )
    parser.add_argument(
        '-step','--lr_step',
        help='lr_step',
        type=float,
        default=0.5,
    )
    parser.add_argument(
        '-wd','--weight_decay',
        help='weight_decay',
        type=float,
        default=1e-4,
    )
    parser.add_argument(
        '-pin','--max_pin',
        help='max pin',
        type=int,
        default=5,
    )
    parser.add_argument(
        '-u','--unet_layer',
        help='unet layer num',
        type=int,
        default=4,
    )

    
    args = parser.parse_args()

    beta2             = args.beta2
    epochs            = args.epoch
    shuf              = args.shuffle
    vis               = args.visualization
    train_flag        = args.train_mode
    pin               = args.max_pin
    unet              = args.unet_layer
    momentum          = args.momentum
    datasize          = args.datasize
    bceloss           = args.bceloss
    diceloss          = args.diceloss
    costloss          = args.costloss
    lr_step           = args.lr_step
    batch_size        = args.batchsize
    posweight         = args.posweight
    learning_rate     = args.learning_rate
    weight_decay      = args.weight_decay
    gradient_clipping = args.gradient_clip
    test              = args.test_case
    
    if shuf == 'F':
        shuffle = False
    else:
        shuffle = True

    if vis == 'F':
        visual = False
    else:
        visual = True

    if train_flag == 'F':
        train_flag = False
    else:
        train_flag = True


    if train_flag == True:
        # datalist_raw = []
        # datalist_raw = load_pickle('datasets_0529/0608_new_modified_CA_size100_2-10pin_stp')
        datalist_raw = load_pickle('0905_dataset/0905_ispd18_even_CA_size150_upto30pin_stp')

        # if datasize == 'big':
        #     train_num = 102400
        #     # train_num = 204800
        #     val_num   = 10240
        #     vis_num = 128

        if datasize == 'big':
            train_num = 51200
            # train_num = 204800
            val_num   = 5120
            vis_num = 256

        
        elif datasize == 'small':
            train_num = 40960
            val_num   = 4096
            vis_num = 128

        else:
            train_num = 204800
            val_num   = 10240
        # 
        # random.shuffle(datalist_raw)
        data_list = datalist_raw[0 : 0+train_num+val_num+vis_num]
        random.seed(329)
        random.shuffle(data_list)


        data_train = data_list[0 : train_num]
        data_val   = data_list[train_num : train_num+val_num]


        count_20 = 0
        count_40 = 0
        count_80 = 0
        count_100 = 0

        for net in data_val:
            if net[0].shape[0]+net[0].shape[1] <= 20:
                count_20 += 1 
            elif net[0].shape[0]+net[0].shape[1] <= 40:
                count_40 += 1 
            elif net[0].shape[0]+net[0].shape[1] <= 80:
                count_80 += 1
            elif net[0].shape[0]+net[0].shape[1] <= 400:
                count_100 += 1

        print(count_20, count_40, count_80, count_100)

        # for i in range(len(data_val)):
        #     if np.sum(data_val[i][2]) < 0.5:
        #         print(data_val[i][2])

        #     # net[2] = np.maximum(data_val[i][2]-1, 0)
        #     # print(net[2])
        #         print('-----------------------------')
        #         input()

        # 固定可视化测试集
    
        # data_vis = load_pickle("datasets_0529/vis_data_small128_medium64_large64_stp_0608")
        data_vis = load_pickle("0905_dataset/vis_data_small128_medium64_large64_size150_upto30pin_stp_0905_ispd18_even")

    else: 
        # infer_dataname = f'1028_cugr_data/split/ispd18/metal5_test{test}/1028_metal5_test{test}_iter0_SmallHighR.pkl'
        # data_infer = load_pickle(infer_dataname)
        data_vis = load_pickle(f"1028_cugr_data/split/ispd19/ispd19_test7_metal5/1106_ispd19_test7_metal5_iter0_MediumHighR_150.pkl")
        # data_vis = load_pickle(f"1028_cugr_data/split/ispd19/ispd19_test7_metal5/1106_ispd19_test7_metal5_iter0_LargeLowR_150_pooled_(4, 4).pkl")



    if unet == 4:
        model = UNet_Small_4(in_channels=2, out_channels=1, bilinear=False)

    if unet == 18:
        model = ResCcaNet(in_channel=2, num_blocks=[2,2,2,2], out_channels=1, bilinear=False)

    elif unet == 34:
        model = ResCcaNet_34(in_channel=2, num_blocks=[2,2,2,2,2,2,2,2], out_channels=1, bilinear=False)
        # model = ResCcaNet(in_channel=2, device=device, num_blocks=[2,2,2,2], out_channels=1, bilinear=False)
        model.load_state_dict(torch.load('./exp_para/0911_node9_multiGPU_cover_cost ispd18_even vis_even epoch60 h+w150 R-CCA-34: 256*8 kernel-5 rcca-2*2(3,4,5,6) dropout ADAM pat-1 Pin2-30 big lr0.0004 FBCE bs64 gc1.0 posw1.0 m0.95 b20.99 wd0.0 c1.0 di1.0 bce2.0 lrstep0.3/0911_node9_multiGPU_cover_cost ispd18_even vis_even epoch60 h+w150 R-CCA-34: 256*8 kernel-5 rcca-2*2(3,4,5,6) dropout ADAM pat-1 Pin2-30 big lr0.0004 FBCE bs64 gc1.0 posw1.0 m0.95 b20.99 wd0.0 c1.0 di1.0 bce2.0 lrstep0.3_epoch_27.pth',map_location=torch.device('cuda:0')))

    elif unet == 2018:
        model = RouteNet(in_channel=2, out_channels=1)
        model.load_state_dict(torch.load('./exp_para/1022_node9_multiGPU_RouteNet AllLoss ispd18_even vis_even epoch60 h+w150 ADAM pat-1 Pin2-30 big lr0.0005 FBCE bs64 gc1.0 posw1.0 m0.95 b20.99 wd0.0 c1.0 di1.0 bce2.0 lrstep0.3/1022_node9_multiGPU_RouteNet AllLoss ispd18_even vis_even epoch60 h+w150 ADAM pat-1 Pin2-30 big lr0.0005 FBCE bs64 gc1.0 posw1.0 m0.95 b20.99 wd0.0 c1.0 di1.0 bce2.0 lrstep0.3_epoch_29.pth',map_location=torch.device('cuda:0')))

    model = model.to(memory_format=torch.channels_last)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)  # 将模型移到cuda:0上

    device_ids = [0]
    # device_ids = [3]
    if torch.cuda.device_count() > 1:
        # print(f"Using {torch.cuda.device_count()} GPUs!")
        print(f"Using {device_ids} GPUs!")
    # model = nn.DataParallel(model)
    model = nn.DataParallel(model, device_ids=device_ids)
    

    logging.info(f'Network:\n'
                 f'\t{model.module.in_channels} input channels\n'
                 f'\t{model.module.out_channels} output channels (classes)\n')

    # model.to(device=device)
    # model = model.cuda()  # 将模型发送到 GPU 上

    # large case:
    # for test in range(2,11):
    #     infer_dataname = f'1028_cugr_data/split/ispd18/metal5_test{test}/1030_metal5_test{test}_iter0_LargeHighR_pooled_(3, 3).pkl'
    #     data_infer = load_pickle(infer_dataname)
    #     print('Length of Data: ', len(data_infer))
    #     '1028_cugr_data/split/ispd18/test2/1030_test2_iter0_LargeLowR_pooled_(3, 3).pkl'



    # all medium+small:

    # for test in range(8,10):
    # # for test in {5,8,10}:
    #     infer_dataname = f'1028_cugr_data/split/ispd19/ispd19_test{test}_metal5/1106_ispd19_test{test}_metal5_iter0_SmallLowR_30.pkl'
    #     data_infer = load_pickle(infer_dataname)
    #     print('Length of Data: ', len(data_infer))
    #     infer_dataname = f'1028_cugr_data/split/ispd19/ispd19_test{test}_metal5/1106_ispd19_test{test}_metal5_iter0_SmallHighR_30.pkl'
    #     data_infer += load_pickle(infer_dataname)
    #     print('Length of Data: ', len(data_infer))
    #     infer_dataname = f'1028_cugr_data/split/ispd19/ispd19_test{test}_metal5/1106_ispd19_test{test}_metal5_iter0_MediumLowR_150.pkl'
    #     data_infer += load_pickle(infer_dataname)
    #     print('Length of Data: ', len(data_infer))
    #     infer_dataname = f'1028_cugr_data/split/ispd19/ispd19_test{test}_metal5/1106_ispd19_test{test}_metal5_iter0_MediumHighR_150.pkl'
    #     data_infer += load_pickle(infer_dataname)
    #     print('Length of Data: ', len(data_infer))
    

    # large metal5:
    for test in range(7,8):
    # for test in {5,8,10}:
        infer_dataname = f'1028_cugr_data/split/ispd19/ispd19_test{test}_metal5/1106_ispd19_test{test}_metal5_iter0_LargeLowR_150_pooled_(5, 5).pkl'
        # infer_dataname = f'1028_cugr_data/split/ispd19/ispd19_test{test}/1106_ispd19_test{test}_iter0_LargeHighR_pooled_(4, 4).pkl'
        # # infer_dataname = f'1028_cugr_data/split/ispd18/test{test}/1030_test{test}_iter0_LargeLowR_pooled_(4, 4).pkl'
        # infer_dataname = f'1028_cugr_data/split/ispd19/ispd19_test{test}_metal5/1106_ispd19_test{test}_metal5_iter0_LargeLowR_150.pkl'
        # infer_dataname = f'1028_cugr_data/split/ispd19/ispd19_test{test}/1106_ispd19_test{test}_iter0_LargeHighR.pkl'
        # infer_dataname = f'1028_cugr_data/split/ispd18/test{test}/1028_test{test}_iter0_LargeLowR.pkl'
        data_infer = load_pickle(infer_dataname)
        print('Length of Data: ', len(data_infer))

    
    # # large:
    # # for test in range(1,11):
    # for test in {7,8,9}:
    #     infer_dataname = f'1028_cugr_data/split/ispd19/ispd19_test{test}/1106_ispd19_test{test}_iter0_LargeLowR.pkl'
    #     data_infer = load_pickle(infer_dataname)
    #     print('Length of Data: ', len(data_infer))



        # infer_dataname = f'1028_cugr_data/split/ispd19/ispd19_test{test}_metal5/1106_ispd19_test{test}_metal5_iter0_MediumHighR_150.pkl'
        # data_infer += load_pickle(infer_dataname)
        # print('Length of Data: ', len(data_infer))

        # if test < 11:
        #     data_vis = load_pickle(f"1028_cugr_data/split/ispd18/metal5_test{test}/Vis_small_low")
        # else:
        #     data_vis = load_pickle(f"1028_cugr_data/split/ispd18/metal5_test{test}/Vis_medium_low")

        print(f'Running iteration {test}')

        def complex_shape_key(item):
            # print(item)
            """
            This function will return a tuple of keys for sorting.
            The first key is the size (area) of the matrix.
            The second key is the ratio of the number of rows to the number of columns.
            """
            pin_map, _, _,_ = item
            pin_map_area = pin_map.shape[0] * pin_map.shape[1]
            pin_map_ratio = pin_map.shape[0] / pin_map.shape[1]


            # The composite key is then a tuple consisting of areas and shape ratios.
            # If you want to prioritize the area over the shape ratio, you can switch the tuple ordering.
            return (pin_map_area , pin_map_ratio)

        # Sort the datalist using the composite shape key
        data_infer.sort(key=complex_shape_key)
        print(len(data_infer))
        data_infer = data_infer[0:int(1.0*len(data_infer))]
        print(len(data_infer))


        if train_flag == True:
            train_model(
                model,
                data_train,
                data_val,
                data_vis,
                num_class  = 1,
                shuffle    = shuffle,
                epochs     = epochs,
                costloss_coef = costloss,
                diceloss_coef = diceloss,
                bceloss_coef  = bceloss,
                posweight     = posweight,
                batch_size    = batch_size,
                learning_rate = learning_rate,
                amp           = True,
                weight_decay  = weight_decay, #adam
                momentum      = momentum,
                beta2         = beta2,
                gradient_clip = gradient_clipping,
                lr_step      = lr_step,
                # exp_name      = f'Cost-Aware Unet{unet}-CCNet: rcca2 conv-5 dropout patient-1 ADAM Pin5-MaxPin_{pin} {datasize} R5 h+w lr{learning_rate} FBCE bs{batch_size} gc{gradient_clipping} posw{posweight} m{momentum} b2{beta2} wd{weight_decay} c{costloss} di{diceloss} bce{bceloss} lrstep{lr_step}',
                # exp_name      = f'0711_CoverLoss ispd18test13579 test_246810 epoch100 h+w100 RESCCANET-{unet}: 256*8 kernel-5 rcca-2*2(4,5) dropout ADAM pat-1 Pin2-{pin} {datasize} lr{learning_rate} FBCE bs{batch_size} gc{gradient_clipping} posw{posweight} m{momentum} b2{beta2} wd{weight_decay} c{costloss} di{diceloss} bce{bceloss} lrstep{lr_step}',
                # exp_name      = f'0909_node9_multiGPU_OrigCostLoss_OtherLoss ispd18_even vis_even epoch60 h+w150 R-CCA-{unet}: 256*8 kernel-5 rcca-2*2(3,4,5,6) dropout ADAM pat-1 Pin2-{pin} {datasize} lr{learning_rate} FBCE bs{batch_size} gc{gradient_clipping} posw{posweight} m{momentum} b2{beta2} wd{weight_decay} c{costloss} di{diceloss} bce{bceloss} lrstep{lr_step}',
                exp_name      = f'1022_node9_multiGPU_RouteNet AllLoss ispd18_even vis_even epoch60 h+w150 ADAM pat-1 Pin2-{pin} {datasize} lr{learning_rate} FBCE bs{batch_size} gc{gradient_clipping} posw{posweight} m{momentum} b2{beta2} wd{weight_decay} c{costloss} di{diceloss} bce{bceloss} lrstep{lr_step}',
                # exp_name      = f'1022_node9_multiGPU_cost_nocover smallnet ispd18_even vis_even epoch60 h+w150 R-CCA-{unet}: 256*4 kernel-7 rcca-2*2(2,3) dropout ADAM pat-1 Pin2-{pin} {datasize} lr{learning_rate} FBCE bs{batch_size} gc{gradient_clipping} posw{posweight} m{momentum} b2{beta2} wd{weight_decay} c{costloss} di{diceloss} bce{bceloss} lrstep{lr_step}',
                
                # exp_name      = f'cut memory RESNET-{unet}: 256*4 kernel-5 rcca-4_every1 patient-1 ADAM Pin5-MaxPin_{pin} {datasize} R5 h+w lr{learning_rate} FBCE bs{batch_size} gc{gradient_clipping} posw{posweight} m{momentum} b2{beta2} wd{weight_decay} c{costloss} di{diceloss} bce{bceloss} lrstep{lr_step}',
                # exp_name      = f'Unet{unet} maxpool-1 dropout patient-1 ADAM Pin5-MaxPin_{pin} {datasize} R5 h+w lr{learning_rate} FBCE bs{batch_size} gc{gradient_clipping} posw{posweight} m{momentum} b2{beta2} wd{weight_decay} c{costloss} di{diceloss} bce{bceloss} lrstep{lr_step}',
                train_flag    = train_flag,
                visual        = visual
            )
        else:
            infer_model_flute(
                model,
                # data_train,
                data_infer,
                data_vis,
                num_class  = 1,
                shuffle    = shuffle,
                epochs     = epochs,
                costloss_coef = costloss,
                diceloss_coef = diceloss,
                bceloss_coef  = bceloss,
                posweight     = posweight,
                batch_size    = batch_size,
                learning_rate = learning_rate,
                amp           = True,
                weight_decay  = weight_decay, #adam
                momentum      = momentum,
                beta2         = beta2,
                gradient_clip = gradient_clipping,
                lr_step       = lr_step,
                # exp_name      = f'1106-node4_multi_padding_load0911_ep27 No Shuffle ispd19_test{test} RESNET-{unet} h+w30_150 Low+HighR RESCCA-34:kernel-5 rcca-2*2(4,5) bs{batch_size}',
                # exp_name      = f'1107-2_multi_padding_load0911_ep27 No Shuffle ispd19_test{test}_metal5 RESNET-{unet} h+w30_150 Low+HighR RESCCA-34:kernel-5 rcca-2*2(4,5) bs{batch_size}',
                # exp_name      = f'1030_load0911_ep27 ispd18_metal5_test{test} RESNET-{unet} h+w Large Pooled HighR RESCCA-34: 256*8 kernel-5 rcca-2*2(4,5) bs{batch_size}',
                # exp_name      = f'1103_load1022_ep29 ispd18_metal5_test{test} RESNET-{unet} h+w30-150 Low+HighR RouteNet bs{batch_size}',
                # exp_name      = f'1107-2_multi_padding_load0911_ep27 NoShuffle ispd19_test{test}_metal5 RESNET-{unet} h+w Large LowR RESCCA-34:kernel-5 rcca-2*2(4,5) bs{batch_size}',
                exp_name      = f'1123-pooling(5,5)_19_t{test}_m5_load0911_ep27 NoShuffle RESNET-{unet} h+w Large150 LowR RESCCA-34:kernel-5 rcca-2*2(4,5) bs{batch_size}',
                # exp_name      = f'1123-pooling(4,4)_19_t{test}_load0911_ep27 NoShuffle RESNET-{unet} h+w Large150 HighR RESCCA-34:kernel-5 rcca-2*2(4,5) bs{batch_size}',
                # # exp_name      = f'1123-pooling(4,4)_18_t{test}_load0911_ep27 NoShuffle RESNET-{unet} h+w Large150 LowR RESCCA-34:kernel-5 rcca-2*2(4,5) bs{batch_size}',
                # exp_name      = f'1123_19_t{test}_m5_load0911_ep27 NoShuffle RESNET-{unet} h+w Large150 LowR RESCCA-34:kernel-5 rcca-2*2(4,5) bs{batch_size}',
                # exp_name      = f'1123_19_t{test}_load0911_ep27 NoShuffle RESNET-{unet} h+w Large150 HighR RESCCA-34:kernel-5 rcca-2*2(4,5) bs{batch_size}',
                # exp_name      = f'1123_18_t{test}_load0911_ep27 NoShuffle RESNET-{unet} h+w Large150 LowR RESCCA-34:kernel-5 rcca-2*2(4,5) bs{batch_size}',
                train_flag    = train_flag,
                visual        = visual,
                test          = test
            )
    


