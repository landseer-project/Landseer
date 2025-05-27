import numpy as np
import os
import argparse
from diffprivlib.mechanisms import Laplace
#from tensorflow.keras.datasets import cifar10

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epsilon', type=float, default=1.0, 
                       help='Total privacy budget (default: 1.0)')
    parser.add_argument('--delta', type=float, default=1e-5,
                       help='Delta for (ε,δ)-DP (default: 1e-5)')
    parser.add_argument("--input-dir", default="/data",
                        help="Directory with data files (default: /data)")
    parser.add_argument('--output', type=str, default='/output',
                       help='Output directory (default: ./output)')
    return parser.parse_args()

def preprocess_images(images):
    images = images.astype(np.float32) / 255.0
    return np.transpose(images, (0, 3, 1, 2))  # [N, 3, 32, 32]

def apply_pixel_dp(images, epsilon):
    dp_images = np.zeros_like(images)
    pixel_epsilon = epsilon / (32 * 32 * 3)
    
    for i in range(images.shape[0]):
        for c in range(3):
            for h in range(32):
                for w in range(32):
                    laplace = Laplace(epsilon=pixel_epsilon, sensitivity=1.0)
                    dp_images[i,c,h,w] = np.clip(laplace.randomise(images[i,c,h,w]), 0, 1)
    return dp_images

def main():
    args = parse_args()
    #os.makedirs(args.output_dir, exist_ok=True)
    #(train_images, train_labels), (test_images, test_labels) = cifar10.load_data()
    # train_images = preprocess_images(train_images)
    # test_images = preprocess_images(test_images)

    # Load CIFAR-10 data from files
    train_images = np.load(os.path.join(args.input_dir, 'data.npy'))
    train_labels = np.load(os.path.join(args.input_dir, 'labels.npy'))
    test_images = np.load(os.path.join(args.input_dir, 'test_data.npy'))
    test_labels = np.load(os.path.join(args.input_dir, 'test_labels.npy'))
    train_images = preprocess_images(train_images)
    test_images = preprocess_images(test_images)

    print(f"Applying DP with ε={args.epsilon}, δ={args.delta}")
    dp_train_images = apply_pixel_dp(train_images, args.epsilon)

    # Save files
    np.save(f'{args.output}/data.npy', dp_train_images)
    np.save(f'{args.output}/labels.npy', train_labels)
    np.save(f'{args.output}/test_data.npy', test_images)
    np.save(f'{args.output}/test_labels.npy', test_labels)

    with open(f'{args.output}/privacy_metrics.txt', 'w') as f:
        f.write(f"epsilon={args.epsilon}\n")
        f.write(f"delta={args.delta}\n")

    print(f"Saved DP-processed CIFAR-10 to {args.output}")

if __name__ == "__main__":
    main()