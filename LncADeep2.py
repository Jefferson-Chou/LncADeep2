import argparse

def main():
    parser = argparse.ArgumentParser(usage='%(prog)s [options]', description='An ab initio lncRNA identification and functional annotation tool based on deep learning')
    parser.add_argument('-m', '--mode', type=str, required=True, choices=['identify', 'anno', 'train'], help='Mode of LncADeep2, choose identify, anno or train a identification model')

    parser.add_argument('-i', '--input', type=str, required=True, help='The path to the input fasta file')
    parser.add_argument('-o', '--output', type=str, help='The path to the output file')
    parser.add_argument('-th', '--thread', type=int, default=1, help='Number of threads, default=1')
    parser.add_argument('-d', '--device', type=str, default='cpu', help='The device used for prediction, e.g. cuda:0 , default=cpu')
    parser.add_argument('-fe', '--feature', type=int, default=0, choices=[0, 1], help='Saving features for identification, default=0, saved in the dir of -o option')
    parser.add_argument('-mf', '--model_file', type=str, default='./identification/models/model_93_0.1750_0.9440.pt', help='The path to the model file')
    parser.add_argument('-tl', '--training_labels', type=str, help='The txt file with labels corresponding to the training FASTA file')   
    
    args = parser.parse_args()

    if args.mode == 'identify':
        try:
            from identification.bin.identification import identify
            identify(filename=args.input, out_dir = args.output, hmmsearch_thread = args.thread, device = args.device, feat_out = args.feature, model_file = args.model_file)
        except Exception as e:
            print(e)
    elif args.mode == 'train':
        try:
            from identification.train import Train
            Train(filename=args.input, label_file = args.training_labels, device = args.device, hmmsearch_thread = args.thread)
        except Exception as e:
            print(e)
    else:
        try:
            from annotation.bin.prediction_bp import pred_go
            pred_go(filename=args.input, dev=args.device, r_thread = args.thread, anno_out = args.output)
        except Exception as e:
            print(e)

if __name__ == '__main__':
    main()
