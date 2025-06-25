import argparse

def main():
    parser = argparse.ArgumentParser(description="practicas command dispatcher")
    parser.add_argument('command', choices=['train', 'predict'], help='Action to perform')
    args = parser.parse_args()

    if args.command == 'train':
        from . import train_model
        train_model.main()
    else:  # predict
        from . import inference_model
        inference_model.main()

if __name__ == '__main__':
    main()
