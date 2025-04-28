import argparse, subprocess

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["preprocess","train","test"], required=True)
    p.add_argument("--epochs", type=int, help="for train")
    p.add_argument("--metric", choices=["basic","det","pca"], help="for test")
    p.add_argument("--checkpoint", help="for test")
    args = p.parse_args()

    if args.mode=="preprocess":
        subprocess.run("python preprocess.py", shell=True, check=True)
    elif args.mode=="train":
        if not args.epochs:
            p.error("--epochs is required for train")
        subprocess.run(f"python train.py {args.epochs}", shell=True, check=True)
    elif args.mode=="test":
        if not (args.metric and args.checkpoint):
            p.error("--metric and --checkpoint are required for test")
        subprocess.run(f"python test.py {args.metric} {args.checkpoint}", shell=True, check=True)

if __name__=="__main__":
    main()
