from pyexpat import model
from sre_parse import parse_template
from statistics import mode
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import sys, os
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import argparse, time

from dataset.dataset import LEMMA, collate_func
# from MY_BERT.model.model import BERT
import model.cnn_bert as cnn_bert

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--basedir", type=str, default='cnn_bert_logs', 
                        help='where to store ckpts and logs')
    
    parser.add_argument("--train_data_file_path", type=str, 
                        default='data/formatted_train_qas_encode.json', 
                        )
    parser.add_argument("--test_data_file_path", type=str, 
                        default='data/formatted_test_qas_encode.json', 
                        )
    parser.add_argument("--val_data_file_path", type=str, 
                        default='data/formatted_val_qas_encode.json', 
                        )
    parser.add_argument('--answer_set_path', type=str, default='data/answer_set.txt')

    parser.add_argument("--batch_size", type=int, default=64, )
    parser.add_argument("--nepoch", type=int, default=5,  
                        help='num of total epoches')
    parser.add_argument("--lr", type=float, default=1e-3,  
                        help='')
    
    parser.add_argument("--i_val",   type=int, default=400, 
                        help='frequency of console printout and metric loggin')
    parser.add_argument("--i_test",   type=int, default=4000, 
                        help='frequency of console printout and metric loggin')
    parser.add_argument("--i_print",   type=int, default=6, 
                        help='frequency of console printout and metric loggin')
    parser.add_argument("--i_weight", type=int, default=50000, 
                        help='frequency of weight ckpt saving')

    parser.add_argument('--img_size', default=(224, 224))
    parser.add_argument('--num_frames_per_video', type=int, default=20)
    parser.add_argument('--cnn_modelname', type=str, default='resnet101')
    parser.add_argument('--cnn_pretrained', type=bool, default=True)
    parser.add_argument('--output_dim', type=int, default=1)
    parser.add_argument('--use_preprocessed_features', type=int, default=1)

    parser.add_argument('--test_only', default=False, type=bool)
    parser.add_argument('--reload_model_path', default='', type=str, help='model_path')

    args = parser.parse_args()
    return args

def train(args):
    device = args.device

    train_dataset = LEMMA(args.train_data_file_path, args.img_size, 'train', args.num_frames_per_video, args.use_preprocessed_features, all_qa_interval_path='/scratch/generalvision/LEMMA/vid_intervals.json')
    train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True, collate_fn=collate_func)
    
    val_dataset = LEMMA(args.val_data_file_path, args.img_size, 'val', args.num_frames_per_video, args.use_preprocessed_features, all_qa_interval_path='/scratch/generalvision/LEMMA/vid_intervals.json')
    val_dataloader = DataLoader(val_dataset, batch_size=64, shuffle=True, collate_fn=collate_func)

    test_dataset = LEMMA(args.test_data_file_path, args.img_size, 'test', args.num_frames_per_video, args.use_preprocessed_features, all_qa_interval_path='/scratch/generalvision/LEMMA/vid_intervals.json')
    test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=True, collate_fn=collate_func)
    
    with open(args.answer_set_path, 'r') as ansf:
        answers = ansf.readlines()
        args.output_dim = len(answers) # # output_dim == len(answers)

    cnn = cnn_bert.build_resnet(args.cnn_modelname, pretrained=args.cnn_pretrained).to(device=args.device)
    cnn.eval() # TODO ?

    bert = cnn_bert.BERT(output_dim=args.output_dim, question_pt_path='data/glove.pt',
                         feature_size=2048, maxlen=100, n_segments=3).to(args.device) # # vocab_size = glove_matrix.shape[0]

    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.Adam(bert.parameters(), lr=args.lr)

    global_step = 0
    TIMESTAMP = "{0:%Y-%m-%dT%H-%M-%S/}".format(datetime.now())
    log_dir = os.path.join(args.basedir, 'events', TIMESTAMP)
    os.makedirs(log_dir)
    with open(os.path.join(log_dir, 'argument.txt'), 'w') as f:
        for key, value in vars(args).items():
            f.write('%s:%s\n'%(key, value))
            print(key, value)

    writer = SummaryWriter(log_dir=log_dir)

    os.makedirs(os.path.join(args.basedir, 'ckpts'), exist_ok=True)
    pbar = tqdm(total=args.nepoch * len(train_dataloader))

    for epoch in range(args.nepoch):
        bert.train()
        for i, (frame_rgbs, question_encode, answer_encode, frame_features, _, question) in enumerate(train_dataloader):
            B, num_frame_per_video, C, H, W = frame_rgbs.shape
            
            frame_rgbs, question_encode, answer_encode = frame_rgbs.to(device), question_encode.to(device), answer_encode.to(device)
            cls_token = torch.ones(B, 1).long().to(device) # # add <CLS> token, assert vocab.index('<CLS<>') == 1
            question_encode = torch.cat((cls_token, question_encode), dim=1)

            if args.use_preprocessed_features:
                frame_features = frame_features.to(device)
            else:
                frame_features = cnn(frame_rgbs.reshape(-1, C, H, W))
                frame_features = frame_features.reshape(B, num_frame_per_video, -1)
            import pdb; pdb.set_trace()
            visual_segments = torch.ones(B, num_frame_per_video, dtype=torch.long)
            question_segments = torch.ones(B, question_encode.shape[1], dtype=torch.long) * 2
            question_segments[question_encode==0] = 0

            segments = torch.cat((question_segments[:, 0:1], visual_segments, question_segments[:, 1:,]), dim=1).to(device)
            
            logits = bert(question_encode, segments, frame_features)

            # gt = torch.zeros(B, args.output_dim).to(device)
            # for i in range(B):
            #     gt[i][answer_encode[i].long()] = 1

            loss = criterion(logits, answer_encode.long())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            pred = torch.argmax(logits, dim=1)
            train_acc = sum(pred == answer_encode) / B

            writer.add_scalar('train/loss', loss.item(), global_step)
            writer.add_scalar('learning rates', optimizer.param_groups[0]['lr'], global_step)
            writer.add_scalar('train/acc', train_acc, global_step)

            pbar.update(1)
            global_step += 1
            
            if global_step % args.i_print  == 0:
                print(f"global_step:{global_step}, train_loss:{loss.item()}, train_acc:{train_acc}")

            if (global_step) % args.i_val == 0:
                val_loss, val_acc = validate(cnn, bert, val_dataloader, epoch, args)
                writer.add_scalar('val/loss', val_loss.item(), global_step)
                writer.add_scalar('val/acc', val_acc, global_step)
            
            if (global_step) % args.i_test == 0:
                test_loss, test_acc = validate(cnn, bert, test_dataloader, epoch, args)
                writer.add_scalar('test/loss', test_loss.item(), global_step)
                writer.add_scalar('test/acc', test_acc, global_step)

            if (global_step) % args.i_weight == 0:
                torch.save({
                    'cnn_state_dict': cnn.state_dict(),
                    'bert_state_dict': bert.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss,
                    'global_step': global_step,
                }, os.path.join(args.basedir, 'ckpts', f"model_{global_step}.tar"))

            

def validate(cnn, bert, val_loader, epoch, args):
    bert.eval()
    all_acc = 0
    all_loss = 0
    batch_size = args.batch_size
    
    starttime = time.time()
    with torch.no_grad():
        for i, (frame_rgbs, question_encode, answer_encode, frame_features, _, question) in enumerate(val_loader):
            B, num_frame_per_video, C, H, W = frame_rgbs.shape
            frame_rgbs, question_encode, answer_encode = frame_rgbs.to(args.device), question_encode.to(args.device), answer_encode.to(args.device)
            if args.use_preprocessed_features:
                frame_features = frame_features.to(device)
            else:
                frame_features = cnn(frame_rgbs.reshape(-1, C, H, W))
                frame_features = frame_features.reshape(B, num_frame_per_video, -1)

            visual_segments = torch.ones(B, num_frame_per_video, dtype=torch.long)
            question_segments = torch.ones(B, question_encode.shape[1], dtype=torch.long) * 2
            question_segments[question_encode==0] = 0

            segments = torch.cat((question_segments[:, 0:1], visual_segments, question_segments[:, 1:,]), dim=1).to(device)
            
            logits = bert(question_encode, segments, frame_features)

            # gt = torch.zeros(B, args.output_dim).to(device)
            # for i in range(B):
            #     gt[i][answer_encode[i].long()] = 1

            all_loss += nn.CrossEntropyLoss().to(device)(logits, answer_encode.long())
            print(time.time() - starttime)
            pred = torch.argmax(logits, dim=1)
            test_acc = sum(pred == answer_encode) / B
            all_acc += test_acc

    all_loss /= len(val_loader)
    all_acc /= len(val_loader)
    bert.train()
    return all_loss, all_acc


def test(args):
    device = args.device
    

def reload(cnn, bert, optimizer, path):
    checkpoint = torch.load(path)
    cnn.load_state_dict(checkpoint['cnn_state_dict'])
    bert.load_state_dict(checkpoint['bert_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    global_step = checkpoint['global_step']
    cnn.eval()
    bert.eval()
    return global_step

if __name__ =='__main__':
    args = parse_args()

    device_id = 0
    device = torch.device(f"cuda:{device_id}" if torch.cuda.is_available() else "cpu")
    args.device = device

    if args.test_only:
        print('test only!')
        print('loading model from', args.reload_model_path)
        test(args)
    else:
        print('start training...')
        train(args)