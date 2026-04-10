import os
import subprocess

def extract_video_tasks():
    root = os.path.join('video_tasks', 'frames')
    all_bundles = list(filter(lambda x: '.tar.gz' in x, os.listdir(root)))
    for task in ['cls', 'qa', 'ret', 'mret']:
        print(f'Extracting {task} task ... ...')
        output_dir = os.path.join(root, f'video_{task}')
        os.makedirs(output_dir, exist_ok=True)
        bundles = sorted(list(filter(lambda x: task in x, all_bundles)))
        bundles = [os.path.join(root, x) for x in bundles]
        if len(bundles) > 1:
            cat = subprocess.Popen(['cat'] + bundles, stdout=subprocess.PIPE)
            cmd = ['tar', '-xzf', '-', '-C', output_dir]
            tar = subprocess.Popen(cmd, stdin=cat.stdout)
            cat_return_code = cat.wait()
            tar_return_code = tar.wait()
        elif len(bundles) == 1:
            tar = subprocess.run(['tar', '-xzf', bundles[0], '-C', output_dir])
        else:
            raise ValueError(f'Unexpected size: {len(bundles)}')
    print('All video tasks completed!')

def extract_image_tasks():
    root = 'image_tasks'
    for task in ['mmeb_v1', 'visdoc']:
        print(f'Extracting {task} task ... ...')
        tar = subprocess.run(['tar', '-xzf', os.path.join(root, f'{task}.tar.gz'), '-C', root])
    print('All image tasks completed!')

def main():
    extract_image_tasks()
    extract_video_tasks()

if __name__ == '__main__':
    main()